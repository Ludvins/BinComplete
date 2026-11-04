"""Window reads preserve bases, layout fallbacks, order and memory bounds."""

import io
from itertools import islice

import numpy as np
import pytest

from bincomplete import ResourceLimits, prepare_dataset
from bincomplete.sequence_io import read_windows, scan_records, window_starts


def prepared(tmp_path, payload):
    assembly = tmp_path / "assembly.fa"
    assembly.write_bytes(payload)
    (tmp_path / "members.tsv").write_text("contig_id\tbin_id\na\tb\n")
    (tmp_path / "coverage.tsv").write_text("contig_id\tmean\tvariance\na\t4\t1\nz\t5\t2\n")
    return prepare_dataset(
        tmp_path / "data",
        assembly=assembly,
        memberships=tmp_path / "members.tsv",
        coverage=tmp_path / "coverage.tsv",
    )


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_regular_layout_reads_identical_windows_and_duplicate_row_order(tmp_path, newline, final_newline):
    sequence = b"acgtnrykmswbdhvu.-" * 7
    lines = [sequence[i : i + 7] for i in range(0, len(sequence), 7)]
    data = prepared(
        tmp_path,
        b">a description"
        + newline
        + newline.join(lines)
        + newline
        + b">z"
        + newline
        + b"acgtn"
        + (newline if final_newline else b""),
    )
    limits = ResourceLimits()
    records = list(scan_records(tmp_path / "assembly.fa", limits))
    assert records[0].line_bases == 7
    assert records[0].line_bytes == 7 + len(newline)
    for width in [1, 6, 17, 1000]:
        for count in [1, 2, 4, 9]:
            actual = list(data.sequence_windows([1, 0, 1], limits, window_bp=width, count=count))
            expected = [
                [value[p : p + width] for p in window_starts(len(value), width, count)]
                for value in [b"ACGTN", sequence.upper(), b"ACGTN"]
            ]
            assert actual == expected
    with pytest.raises(IndexError):
        list(data.sequence_windows([2], limits, window_bp=4, count=4))


@pytest.mark.parametrize(
    "body",
    [
        b"ACG\nTA\nCGTA\n",
        b"\nACGT\nACGT\n",
        b"ACGT\n\nACGT\n",
        b"AC GT\tN\nACGT\n",
        b"ACGT\r\nACGT\nACGT\n",
        b"AC\nACGT\n",
    ],
)
def test_irregular_layout_uses_the_full_sequence_fallback(tmp_path, body):
    data = prepared(tmp_path, b">a\n" + body + b">z\nACGT\n")
    record = next(scan_records(tmp_path / "assembly.fa", ResourceLimits()))
    assert record.line_bases == record.line_bytes == 0
    sequence = b"".join(body.split()).upper()
    actual = next(data.sequence_windows([0], ResourceLimits(), window_bp=3, count=4))
    assert actual == [sequence[p : p + 3] for p in window_starts(len(sequence), 3, 4)]


def test_long_regular_contig_windows_fit_a_small_budget_without_full_reads(tmp_path):
    sequence = b"ACGT" * 500_000
    payload = (
        b">a\n" + b"\n".join(sequence[i : i + 70] for i in range(0, len(sequence), 70)) + b"\n>z\nACGT\n"
    )
    data = prepared(tmp_path, payload)
    small = ResourceLimits(memory_gib=0.001)
    with pytest.raises(MemoryError):
        next(data.sequences([0], small))
    windows = next(data.sequence_windows([0], small, window_bp=6000, count=4))
    assert windows == [sequence[p : p + 6000] for p in window_starts(len(sequence), 6000, 4)]


def test_overlapping_windows_read_one_merged_span(tmp_path):
    sequence = b"ACGT" * 2000
    path = tmp_path / "simple.fa"
    path.write_bytes(b">a\n" + sequence + b"\n")
    record = next(scan_records(path, ResourceLimits()))

    class Counted(io.BytesIO):
        total = 0

        def read(self, amount=-1):
            value = super().read(amount)
            self.total += len(value)
            return value

    stream = Counted(path.read_bytes())
    actual = read_windows(
        stream,
        (record.start, record.end, len(sequence), record.line_bases, record.line_bytes),
        6000,
        4,
        ResourceLimits(),
    )
    assert stream.total == len(sequence)
    assert actual == [sequence[p : p + 6000] for p in window_starts(len(sequence), 6000, 4)]


def test_window_pooling_preserves_tokenization_batches_and_float_reductions(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from bincomplete.representations.strand_pooling import pooled_vectors, pooled_windows

    data = prepared(tmp_path, b">a\nACGTACG\nTACGTAC\nGTACG\n>z\nTTTTACGT\n")

    class Tokenizer:
        all_special_ids = [0, 9]

        def __call__(self, values, **kwargs):
            def ids(s):
                return [9] + ["ACGT".index(c) + 1 for c in s] + [9]

            if isinstance(values, str):
                return {"input_ids": ids(values)}
            rows = [ids(value) for value in values]
            matrix = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.int64)
            for i, row in enumerate(rows):
                matrix[i, : len(row)] = torch.tensor(row)
            return {"input_ids": matrix, "attention_mask": matrix != 0}

    class Model:
        def __call__(self, **tokens):
            values = tokens["input_ids"].float()
            return SimpleNamespace(
                last_hidden_state=torch.stack(
                    [values, values**2, torch.arange(values.shape[1]).expand_as(values).float()], dim=-1
                )
            )

    space = dict(window_bp=12, windows=4, max_tokens=8, reverse_complement=True)
    execution = dict(device="cpu", batch_size=3, precision="float32")
    full = list(data.sequences([0, 1], ResourceLimits()))
    windows = list(data.sequence_windows([0, 1], ResourceLimits(), window_bp=12, count=4))
    np.testing.assert_array_equal(
        pooled_vectors(Tokenizer(), Model(), full, space, execution),
        pooled_windows(Tokenizer(), Model(), windows, space, execution),
    )


def test_scanner_rejects_invalid_or_empty_sequences(tmp_path):
    path = tmp_path / "invalid.fa"
    for body in [b">a\n", b">a\nACGTX\n", b"ACGT\n", b">\nACGT\n"]:
        path.write_bytes(body)
        with pytest.raises(ValueError):
            list(islice(scan_records(path, ResourceLimits()), 1))


def test_window_position_workspace_is_bounded_before_allocating():
    with pytest.raises(MemoryError, match="Window-position workspace"):
        read_windows(io.BytesIO(b"ACGT"), (0, 4, 4, 4, 4), 4, 1_000_000, ResourceLimits(memory_gib=0.001))
