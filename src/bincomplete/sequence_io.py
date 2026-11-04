"""FASTA layout validation and bounded, byte-exact sequence window reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .resources import ResourceLimits

ALLOWED_BASES = b"ACGTNRYKMSWBDHVU.-"


@dataclass(frozen=True)
class SequenceRecord:
    """One validated sequence, its file span and optional regular line layout.

    Parameters
    ----------
    name : str
        First whitespace-delimited FASTA header token.
    sequence : bytes
        Uppercase bases, with sequence-line whitespace removed.
    start, end : int
        Byte span in the original uncompressed FASTA.
    line_bases, line_bytes : int
        Full-line base and byte widths, or zero for irregular wrapping.
    """

    name: str
    sequence: bytes
    start: int
    end: int
    line_bases: int
    line_bytes: int


def scan_records(path: str | Path, limits: ResourceLimits) -> Iterator[SequenceRecord]:
    """Stream validated records, accepting irregular FASTA through a safe fallback.

    Parameters
    ----------
    path : path-like
        Original plain FASTA. Gzip cannot provide these random-access offsets.
    limits : ResourceLimits
        Maximum allocation for one sequence buffer.

    Yields
    ------
    SequenceRecord
        Layout is regular only when every interior line has the same width,
        without embedded whitespace. The final line may be shorter.
    """
    with Path(path).open("rb") as stream:
        if stream.read(2) == b"\x1f\x8b":
            raise ValueError(
                "Random-access inputs must be plain FASTA; supply an existing uncompressed assembly"
            )
        stream.seek(0)
        parts: list[bytes] = []
        name, start, size = None, 0, 0
        line_bases = line_bytes = 0
        previous = None
        regular = True
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line or line.startswith(b">"):
                if name is not None:
                    if not size:
                        raise ValueError(f"Empty FASTA record: {name}")
                    layout = regular and previous is not None and previous[0] <= line_bases
                    yield SequenceRecord(
                        name,
                        b"".join(parts),
                        start,
                        offset,
                        line_bases if layout else 0,
                        line_bytes if layout else 0,
                    )
                if not line:
                    break
                tokens = line[1:].split()
                if not tokens:
                    raise ValueError("Empty FASTA identifier")
                name, start, parts, size = tokens[0].decode("utf-8"), stream.tell(), [], 0
                line_bases = line_bytes = 0
                previous, regular = None, True
            elif line.strip():
                if name is None:
                    raise ValueError("Sequence occurs before the first FASTA header")
                part = b"".join(line.split()).upper()
                if part.translate(None, ALLOWED_BASES):
                    raise ValueError(f"Invalid nucleotide sequence in {name}")
                size += len(part)
                limits.require(size * 4, f"Sequence buffer for {name}")
                if previous is not None and previous != (line_bases, line_bytes):
                    regular = False
                if not line_bases:
                    line_bases, line_bytes = len(part), len(line)
                if len(line.rstrip(b"\r\n")) != len(part):
                    regular = False
                previous = (len(part), len(line))
                parts.append(part)
            else:
                regular = False


def window_starts(bp: int, window_bp: int, count: int) -> np.ndarray:
    """Return the original uniform-window positions without changing rounding."""
    if type(window_bp) is not int or window_bp < 1 or type(count) is not int or count < 1:
        raise ValueError("Window size and count must be positive integers")
    return np.unique(np.linspace(0, max(0, bp - window_bp), count).astype(int))


def read_sequence(stream: BinaryIO, start: int, end: int, bp: int, limits: ResourceLimits) -> bytes:
    """Read one indexed sequence using the legacy whitespace/uppercase contract."""
    limits.require((end - start) * 3, "Indexed sequence buffer")
    stream.seek(start)
    value = b"".join(stream.read(end - start).split()).upper()
    if len(value) != bp:
        raise ValueError("Assembly sequence span changed")
    return value


def read_windows(
    stream: BinaryIO,
    record: tuple[int, int, int, int, int],
    window_bp: int,
    count: int,
    limits: ResourceLimits,
) -> list[bytes]:
    """Read exact requested windows, merging overlap and retaining input order.

    Parameters
    ----------
    stream : binary file
        Open original assembly; this function owns only its seek position.
    record : tuple of int
        Sequence start/end offsets, base length and validated line widths.
        Zero line widths select a full-sequence fallback for legacy or irregular
        indexes. No width is guessed from an unvalidated first line.
    window_bp, count : int
        The unchanged encoder window configuration.
    limits : ResourceLimits
        Maximum allocation for merged spans and returned windows.

    Returns
    -------
    list of bytes
        Byte-identical slices of the legacy uppercase full-sequence read.
    """
    start, end, bp, line_bases, line_bytes = record
    if bp < 1:
        raise ValueError("Indexed sequence length must be positive")
    if type(count) is int and count > 0:
        limits.require(count * 24, "Window-position workspace")
    starts = window_starts(bp, window_bp, count)
    if not line_bases or not line_bytes:
        sequence = read_sequence(stream, start, end, bp, limits)
        return [sequence[position : position + window_bp] for position in starts]
    if bp < 1 or line_bytes < line_bases:
        raise ValueError("Invalid regular FASTA layout")
    ranges = [(int(position), min(bp, int(position) + window_bp)) for position in starts]
    merged: list[tuple[int, int]] = []
    for left, right in ranges:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    retained = sum(right - left for left, right in ranges)
    buffers = []
    for left, right in merged:
        first = start + left // line_bases * line_bytes + left % line_bases
        last_base = right - 1
        last = start + last_base // line_bases * line_bytes + last_base % line_bases + 1
        if first < start or last > end:
            raise ValueError("Indexed window escaped its sequence span")
        limits.require(retained + (last - first) * 3, "Indexed sequence windows")
        stream.seek(first)
        value = b"".join(stream.read(last - first).split()).upper()
        if len(value) != right - left:
            raise ValueError("Assembly window layout changed")
        buffers.append((left, right, value))
    return [
        value[left - begin : right - begin]
        for left, right in ranges
        for begin, stop, value in buffers
        if begin <= left and right <= stop
    ]
