"""Standalone DNABERT-S encoding and portable embedding artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .artifacts import complete, read, save, sha, verify_outputs
from .cache import DerivedCache
from .representations.dna import frozen_array, local_contract
from .representations.validation import validate_encoder_matrix
from .resources import FileLock, ResourceLimits
from .sequence_io import read_sequence, scan_records, window_starts

_BUNDLE_FILES = ("embeddings.npy", "embedding_ids.txt", "embedding_space.json")
_BUNDLE_KIND = "dnabert_s_embeddings"


def verify_bundle(
    assembly: str | Path,
    embeddings: str | Path,
    ids: str | Path,
    space: str | Path,
    *,
    input_hashes: Mapping[str, str] | None = None,
) -> tuple[Path, str] | None:
    """Verify an encode-produced bundle when supplied files carry its receipt."""
    files = tuple(Path(path).resolve() for path in (embeddings, ids, space))
    parents = {path.parent for path in files}
    for parent in sorted(parents):
        marker = parent / "complete.json"
        if not marker.is_file():
            continue
        raw = marker.read_bytes()
        proof = json.loads(raw)
        if not isinstance(proof, dict) or proof.get("kind") != _BUNDLE_KIND:
            continue
        if parents != {parent}:
            raise ValueError("Supply all three files from the same BinComplete embedding bundle")
        hashes, request = proof.get("outputs_sha256"), proof.get("request")
        if (
            proof.get("schema_version") != 1
            or not isinstance(request, dict)
            or not isinstance(hashes, dict)
            or set(hashes) != set(_BUNDLE_FILES)
        ):
            raise ValueError(f"Invalid BinComplete embedding bundle receipt: {marker}")

        def digest(path: Path) -> str:
            return sha(path) if input_hashes is None else input_hashes[str(path)]

        if request.get("assembly_sha256") != digest(Path(assembly).resolve()):
            raise ValueError("Embedding bundle belongs to a different assembly; encode this assembly first")
        for name, path in zip(_BUNDLE_FILES, files):
            if hashes[name] != digest(path):
                raise ValueError(f"Embedding bundle file changed: {path.name}")
        return marker, hashlib.sha256(raw).hexdigest()
    return None


class _Assembly:
    """Index only the fields needed by the shared frozen encoder, in FASTA order."""

    def __init__(self, path: Path, digest: str, limits: ResourceLimits):
        self.path, self.limits = path, limits
        self.manifest = {"assembly": str(path), "inputs_sha256": {str(path): digest}}
        self.ids: list[str] = []
        self.spans: list[tuple[int, int, int]] = []
        self.sequence_hashes: list[bytes] = []
        seen: set[str] = set()
        self.reserved_bytes = 0
        indexed = hashlib.sha256()
        for record in scan_records(path, limits, update_digest=indexed.update):
            if record.name in seen:
                raise ValueError(f"Duplicate FASTA identifier: {record.name}")
            if record.name != record.name.strip() or record.name.splitlines() != [record.name]:
                raise ValueError(
                    f"FASTA identifier cannot be stored in the embedding row IDs: {record.name!r}"
                )
            self.reserved_bytes += 768 + 4 * len(record.name)
            limits.require(self.reserved_bytes + 4 * len(record.sequence), "Assembly index and sequence")
            seen.add(record.name)
            self.ids.append(record.name)
            self.spans.append((record.start, record.end, len(record.sequence)))
            self.sequence_hashes.append(hashlib.sha256(record.sequence).digest())
        if indexed.hexdigest() != digest:
            raise ValueError("Assembly changed while indexing for encoding")
        if not self.ids:
            raise ValueError("Assembly FASTA has no contigs")

    def sequence_windows(self, rows: Sequence[int], *, window_bp: int, count: int) -> Iterator[list[bytes]]:
        self.limits.require(self.reserved_bytes + count * 24, "Assembly index and window positions")
        with self.path.open("rb") as stream:
            for row in rows:
                start, end, _ = self.spans[row]
                self.limits.require(self.reserved_bytes + (end - start) * 3, "Assembly index and sequence")
                sequence = read_sequence(stream, *self.spans[row], self.limits)
                if hashlib.sha256(sequence).digest() != self.sequence_hashes[row]:
                    raise ValueError(f"Assembly changed during encoding: {self.ids[row]}")
                windows = [
                    sequence[start : start + window_bp]
                    for start in window_starts(len(sequence), window_bp, count)
                ]
                del sequence
                yield windows


def encode(
    assembly: Path,
    output: Path,
    *,
    encoder_path: Path,
    encoder_options: dict[str, Any],
    cache_root: Path,
    limits: ResourceLimits,
) -> Path:
    """Publish an immutable, verified embedding bundle for one assembly."""
    assembly, output, encoder_path = (
        path.expanduser().resolve() for path in (assembly, output, encoder_path)
    )
    cache_root = cache_root.expanduser().resolve()
    if (
        assembly.is_relative_to(output)
        or output.is_relative_to(encoder_path)
        or encoder_path.is_relative_to(output)
    ):
        raise ValueError("Embedding output must be separate from the assembly and encoder directory")
    if (
        output.is_relative_to(cache_root)
        or cache_root.is_relative_to(output)
        or cache_root.is_relative_to(encoder_path)
        or encoder_path.is_relative_to(cache_root)
        or assembly.is_relative_to(cache_root)
    ):
        raise ValueError("Keep the cache separate from embedding outputs, the assembly and encoder weights")
    digest = sha(assembly)
    space, model_files = local_contract(encoder_path, encoder_options)
    execution = {
        "device": encoder_options.get("device", "cpu"),
        "batch_size": encoder_options.get("batch_size", 8),
        "precision": encoder_options.get("precision", "float32"),
        "storage_dtype": encoder_options.get("storage_dtype", "float32"),
    }
    request = {"assembly_sha256": digest, "space": space, "execution": execution}
    files = set(_BUNDLE_FILES)
    with FileLock(output.with_name(output.name + ".encode.lock")):
        marker = output / "complete.json"
        if marker.is_file():
            proof = read(marker)
            if (
                not isinstance(proof, dict)
                or proof.get("schema_version") != 1
                or proof.get("kind") != _BUNDLE_KIND
                or proof.get("request") != request
                or not isinstance(proof.get("outputs_sha256"), dict)
                or set(proof["outputs_sha256"]) != files
            ):
                raise ValueError("Embedding output belongs to another request; choose a new output directory")
            verify_outputs(output, proof)
            ids = (output / "embedding_ids.txt").read_text(encoding="utf-8").splitlines()
            matrix = np.load(output / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            try:
                validate_encoder_matrix(
                    matrix,
                    ids,
                    space["dimension"],
                    execution["storage_dtype"],
                    limits,
                    "completed embedding bundle",
                )
            finally:
                matrix._mmap.close()
            return output
        if output.exists():
            raise FileExistsError(f"Embedding output already exists; choose a new directory: {output}")

        store = _Assembly(assembly, digest, limits)

        def verify_sources() -> None:
            if sha(assembly) != digest:
                raise ValueError("Assembly changed during encoding")
            for path, expected in model_files.items():
                if sha(path) != expected:
                    raise ValueError(f"Encoder file changed during encoding: {path}")

        cache = DerivedCache(cache_root, limits, before_publish=verify_sources)
        context = SimpleNamespace(
            store=store,
            limits=limits,
            cache=cache,
            reserved_bytes=store.reserved_bytes,
            status=lambda **_: None,
        )
        spec = {
            "space": space,
            "model_path": str(encoder_path),
            "model_files": model_files,
            "execution": execution,
        }
        try:
            array = frozen_array(context, spec)
            if cache.hits:
                verify_sources()
            cached = Path(array.filename)
            encoding_identity = read(cached.with_name("complete.json"))["identity"]
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}.encode-", dir=output.parent
            ) as temporary:
                stage = Path(temporary)
                shutil.copyfile(cached, stage / "embeddings.npy")
                matrix = np.load(stage / "embeddings.npy", mmap_mode="r", allow_pickle=False)
                try:
                    validate_encoder_matrix(
                        matrix,
                        store.ids,
                        space["dimension"],
                        execution["storage_dtype"],
                        limits,
                        "embedding export",
                    )
                finally:
                    matrix._mmap.close()
                with (stage / "embedding_ids.txt").open("w", encoding="utf-8", newline="\n") as stream:
                    for name in store.ids:
                        stream.write(name + "\n")
                save(stage / "embedding_space.json", space)
                complete(
                    stage,
                    {
                        "schema_version": 1,
                        "kind": _BUNDLE_KIND,
                        "request": request,
                        "encoding_identity": encoding_identity,
                        "contigs": len(store.ids),
                    },
                )
                if output.exists() or output.is_symlink():
                    raise FileExistsError(f"Embedding output already exists: {output}")
                stage.rename(output)
        finally:
            cache.close()
    return output
