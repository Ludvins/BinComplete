"""Generic assembly inputs, indexed sequence access and explicit format adapters."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import complete, fingerprint, read, save, sha, verify_outputs
from .resources import FileLock, ResourceLimits


def fasta_records(path: str | Path, limits: ResourceLimits) -> Iterator[tuple[str, bytes, int, int]]:
    """Stream FASTA records and their byte spans from an uncompressed input.

    Parameters
    ----------
    path : path-like
        Plain FASTA; the first whitespace-delimited header token is the ID.
    limits : ResourceLimits
        Maximum sequence buffer allocation.

    Yields
    ------
    tuple
        Identifier, uppercase sequence, sequence-start offset and end offset.
    """
    from .sequence_io import scan_records

    for record in scan_records(path, limits):
        yield record.name, record.sequence, record.start, record.end


def _one(paths: Sequence[Path], label: str) -> Path:
    found = sorted(set(p.resolve() for p in paths if p.is_file()))
    if len(found) != 1:
        choices = ", ".join(str(p) for p in found) or "none"
        raise ValueError(
            f"Supply {label} explicitly; adapter found {len(found)} possible files: {choices}. "
            f"Use init --{label} PATH or datasets.sample.{label} in the experiment."
        )
    return found[0]


def bin_fasta_paths(root: str | Path) -> list[Path]:
    """Reject unsupported bin sequences before any original bin can be omitted."""
    entries = sorted(Path(root).iterdir())
    extensions = {".fa", ".fna", ".fasta"}
    # Compare suffix tokens, including a compressed FASTA's inner extension.
    unsupported = [
        p
        for p in entries
        if p.is_file()
        and any("." + s in extensions | {".fas", ".fsa"} for s in p.name.lower().split(".")[1:])
        and p.suffix.lower() not in extensions
    ]
    if unsupported:
        raise ValueError(
            "Unsupported bin FASTAs would be omitted: "
            + ", ".join(str(p) for p in unsupported)
            + ". Decompress to plain .fa/.fna/.fasta (or supply a complete membership TSV); "
            "all original bins must remain protected."
        )
    paths = [p for p in entries if p.is_file() and p.suffix.lower() in extensions]
    if not paths:
        raise ValueError("No plain bin FASTAs found; supply .fa/.fna/.fasta files or a membership TSV")
    return paths


def _primary_depth(root: Path) -> Path:
    candidates = sorted((root / "tmp/binning/mapping").glob("1-*.tsv"))
    primary = []
    for path in candidates:
        if path.stem.lower().endswith(("_tr1", "_tr2")):
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream, delimiter="\t"), [])
        if {"contigName", "contigLen", "totalAvgDepth"} <= set(header) and any(
            header[i + 1].endswith("-var") for i in range(3, len(header) - 1, 2)
        ):
            primary.append(path)
    if not primary:
        raise ValueError(
            "No primary JGI coverage table with measured variance found. Retain "
            "tmp/binning/mapping/1-<profile>.tsv alongside results/; "
            "use init --coverage PATH (and --coverage-profile for multiple samples). "
            "Derived _tr1/_tr2 tables and final quality summaries do not replace it. "
            "Inspected: " + (", ".join(str(p) for p in candidates) or "none")
        )
    return _one(primary, "coverage")


def mmlong2_inputs(root: str | Path, **overrides: Any) -> dict[str, Any]:
    """Discover mmlong2 final outputs without importing its workflow or quality data.

    Parameters
    ----------
    root : path-like
        Sample directory containing ``results`` and optional retained ``tmp``.
    **overrides
        Explicit ``assembly``, ``bins``, ``memberships`` or ``coverage`` paths.

    Returns
    -------
    dict
        Generic preparation arguments. Ambiguous coverage profiles are rejected.
    """
    root = Path(root).resolve()
    assembly = overrides.get("assembly") or _one(
        list((root / "results").glob("*_assembly.fasta")), "assembly"
    )
    bins = overrides.get("bins")
    memberships = overrides.get("memberships")
    if bins is None and memberships is None:
        bins = root / "results" / "bins"
        if not bins.is_dir():
            memberships = _one(list((root / "results").glob("*_contigs.tsv")), "memberships")
            bins = None
    coverage = overrides.get("coverage") or _primary_depth(root)
    return {"assembly": assembly, "bins": bins, "memberships": memberships, "coverage": coverage}


def _membership_rows(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        first = next(rows, None)
        if first is None:
            raise ValueError("Empty membership table")
        if "bin" in first and ("contig" in first or "contig_id" in first):
            ci = first.index("contig" if "contig" in first else "contig_id")
            bi = first.index("bin")
        elif "bin_id" in first and "contig_id" in first:
            ci, bi = first.index("contig_id"), first.index("bin_id")
        else:
            if len(first) != 2:
                raise ValueError("Membership TSV needs contig_id/bin_id columns or two headerless columns")
            ci, bi = 0, 1
            yield first[ci], first[bi]
        for row in rows:
            if row:
                yield row[ci], row[bi]


def _coverage(path: Path, ids: list[str], profile: str | None) -> np.ndarray:
    positions = {name: i for i, name in enumerate(ids)}
    values = np.full((len(ids), 2), np.nan, np.float64)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        header = next(rows)
        if "contig_id" in header and "mean" in header:
            ci, mi = header.index("contig_id"), header.index("mean")
            vi = header.index("variance") if "variance" in header else None
        elif "contigName" in header and "totalAvgDepth" in header:
            pairs = [(i, i + 1) for i in range(3, len(header) - 1, 2) if header[i + 1].endswith("-var")]
            if profile is not None:
                pairs = [pair for pair in pairs if header[pair[0]] == profile]
            if len(pairs) != 1:
                raise ValueError("Select exactly one JGI coverage profile by its mean-column name")
            ci, (mi, vi) = header.index("contigName"), pairs[0]
        else:
            raise ValueError("Coverage requires contig_id/mean[/variance] or JGI depth columns")
        seen = set()
        for row in rows:
            if not row or row[ci] not in positions:
                continue
            name = row[ci]
            if name in seen:
                raise ValueError(f"Duplicate coverage row: {name}")
            seen.add(name)
            values[positions[name], 0] = float(row[mi])
            if vi is not None:
                values[positions[name], 1] = float(row[vi])
    if not np.isfinite(values[:, 0]).all() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("Coverage means must be finite, nonnegative and present for every assembly contig")
    if vi is not None and not np.isfinite(values[:, 1]).all():
        raise ValueError("Supplied coverage variance must be finite and present for every contig")
    return values


def _composition(path: Path, ids: list[str], limits: ResourceLimits) -> np.ndarray:
    """Read an original canonical four-mer table without loading unrelated columns.

    Parameters
    ----------
    path : Path
        CSV or TSV with an ID column followed by columns named 0 through 135.
    ids : list of str
        Assembly row order; every ID must occur exactly once in the table.
    limits : ResourceLimits
        Explicit allocation budget for the compact float32 matrix.

    Returns
    -------
    numpy.ndarray
        Unnormalized supplied values. The method's declared numerical profile
        controls normalization; counts and frequencies are both supported.
    """
    limits.require(len(ids) * 136 * 4 + len(ids) * 768, "Supplied composition table")
    positions = {name: row for row, name in enumerate(ids)}
    values = np.empty((len(ids), 136), np.float32)
    seen = np.zeros(len(ids), bool)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream, delimiter="," if path.suffix.lower() == ".csv" else "\t")
        header = next(rows)
        try:
            columns = [header.index(str(i)) for i in range(136)]
        except ValueError as error:
            raise ValueError("Composition needs canonical four-mer columns named 0 through 135") from error
        for row in rows:
            if not row or row[0] not in positions:
                continue
            index = positions[row[0]]
            if seen[index]:
                raise ValueError(f"Duplicate composition row: {row[0]}")
            vector = np.asarray([row[column] for column in columns], np.float32)
            if not np.isfinite(vector).all() or (vector < 0).any() or not vector.sum(dtype=np.float64):
                raise ValueError(f"Invalid composition values: {row[0]}")
            values[index] = vector
            seen[index] = True
    if not seen.all():
        raise ValueError("Supplied composition is missing assembly contigs")
    return values


@dataclass(frozen=True)
class Dataset:
    """Reference to a prepared dataset; sequences and embedding arrays stay in place.

    Parameters
    ----------
    path : Path
        Directory containing the immutable dataset manifest and sequence index.
    """

    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the dataset manifest."""
        return read(self.path / "dataset.json")

    def verify(self, *, inputs: bool = True) -> dict[str, Any]:
        """Verify prepared outputs and, by default, external input identities."""
        proof = read(self.path / "complete.json")
        verify_outputs(self.path, proof)
        manifest = self.manifest
        self._index_owner(verify=True)
        if inputs:
            for path, digest in manifest["inputs_sha256"].items():
                if sha(path) != digest:
                    raise ValueError(f"Input changed since preparation: {path}")
        return manifest

    def _index_owner(self, filename: str = "sequences.sqlite", *, verify: bool = False) -> Dataset:
        """Resolve immutable shared metadata, checking source identities and cycles."""
        current = self
        visited = set()
        manifest = self.manifest
        assembly_sha = manifest["inputs_sha256"][manifest["assembly"]]
        while True:
            root = current.path.resolve()
            if root in visited:
                raise ValueError("Cyclic shared assembly-index reference")
            visited.add(root)
            metadata = current.manifest
            if metadata["inputs_sha256"][metadata["assembly"]] != assembly_sha:
                raise ValueError("Shared assembly index belongs to different FASTA bytes")
            reference = metadata.get("assembly_index")
            if reference is None or (
                filename not in {"ids.npy", "lengths.npy", "sequence_hashes.npy", "sequences.sqlite"}
                and filename not in metadata.get("shared_arrays", [])
            ):
                return current
            source = Dataset(Path(reference["path"]))
            marker = source.path / "complete.json"
            if sha(marker) != reference["receipt_sha256"]:
                raise ValueError("Shared assembly-index receipt changed")
            if verify:
                verify_outputs(source.path, read(marker))
            current = source

    def array_path(self, name: str) -> Path:
        """Resolve a local array or immutable shared assembly-metadata file.

        Parameters
        ----------
        name : str
            Exact metadata filename, such as ``ids.npy`` or ``coverage.npy``.

        Returns
        -------
        Path
            Assembly metadata may belong to another verified prepared dataset;
            matching raw numeric arrays can also be shared. Memberships, donor
            pools and cohort-specific inputs remain dataset-local.
        """
        if Path(name).name != name:
            raise ValueError("Array name must be a single filename")
        owner = (
            self._index_owner(name)
            if name in {"ids.npy", "lengths.npy", "sequence_hashes.npy", "sequences.sqlite"}
            or name in self.manifest.get("shared_arrays", [])
            else self
        )
        return owner.path / name

    def sequences(self, rows: Sequence[int], limits: ResourceLimits) -> Iterator[bytes]:
        """Read requested sequences in order using bounded indexed file spans."""
        from .sequence_io import read_sequence

        with sqlite3.connect(
            f"file:{self.array_path('sequences.sqlite').as_posix()}?mode=ro", uri=True
        ) as db:
            with Path(self.manifest["assembly"]).open("rb") as stream:
                for row in rows:
                    record = db.execute(
                        "SELECT start,end,bp FROM sequence WHERE row=?", (int(row),)
                    ).fetchone()
                    if record is None:
                        raise IndexError(row)
                    yield read_sequence(stream, *record, limits)

    def sequence_windows(
        self, rows: Sequence[int], limits: ResourceLimits, *, window_bp: int, count: int
    ) -> Iterator[list[bytes]]:
        """Stream exact encoder windows without loading regularly wrapped contigs.

        Parameters
        ----------
        rows : sequence of int
            Assembly rows, retaining caller order and duplicates.
        limits : ResourceLimits
            Read-buffer allocation budget.
        window_bp, count : int
            Unchanged uniform-window semantics of the configured encoder.

        Yields
        ------
        list of bytes
            Ordered uppercase windows. Legacy or irregular indexes use the full
            sequence fallback without modifying the source index.
        """
        from .sequence_io import read_windows

        with sqlite3.connect(
            f"file:{self.array_path('sequences.sqlite').as_posix()}?mode=ro", uri=True
        ) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(sequence)")}
            layout = {"line_bases", "line_bytes"} <= columns
            fields = "start,end,bp,line_bases,line_bytes" if layout else "start,end,bp,0,0"
            with Path(self.manifest["assembly"]).open("rb") as stream:
                for row in rows:
                    record = db.execute(f"SELECT {fields} FROM sequence WHERE row=?", (int(row),)).fetchone()
                    if record is None:
                        raise IndexError(row)
                    yield read_windows(stream, record, window_bp, count, limits)


def prepare_dataset(
    output: str | Path,
    *,
    assembly: str | Path | None = None,
    bins: str | Path | None = None,
    memberships: str | Path | None = None,
    coverage: str | Path | None = None,
    composition: str | Path | None = None,
    assembly_index: str | Path | None = None,
    coverage_profile: str | None = None,
    groups: str | Path | None = None,
    source_ids: str | Path | None = None,
    embeddings: str | Path | None = None,
    embedding_ids: str | Path | None = None,
    embedding_space: str | Path | None = None,
    encoder_path: str | Path | None = None,
    encoder_options: dict[str, Any] | None = None,
    donor_ids: str | Path | None = None,
    input_format: str = "generic",
    input_root: str | Path | None = None,
    limits: ResourceLimits | None = None,
) -> Dataset:
    """Index inputs in place and freeze membership, coverage and representation metadata.

    Parameters
    ----------
    output : path-like
        New prepared-dataset directory, or an identical existing dataset.
    assembly : path-like, optional
        Plain assembly FASTA containing both binned and candidate contigs.
    assembly_index : path-like, optional
        Existing immutable prepared dataset for identical FASTA bytes. Reuses its
        sequence index, IDs, lengths and sequence hashes without copying them.
        Identical coverage and supplied composition inputs also share arrays.
    bins, memberships : path-like, optional
        Exactly one bin FASTA directory or contig-to-bin membership TSV.
    coverage : path-like, optional
        Raw coverage mean and optional variance table; quality columns are ignored.
    composition : path-like, optional
        Original canonical four-mer CSV/TSV with ID and columns 0 through 135.
        Supplied values replace sequence-derived composition; quality and other
        columns are ignored. Declare the basis in the method's preprocessing.
    coverage_profile : str, optional
        Mean-column name selecting one sample from a multi-profile JGI table.
    groups : path-like, optional
        Two-column TSV mapping every bin ID to a related-group ID.
    source_ids : path-like, optional
        Two-column bin_id/source_id TSV preserving stable source RNG identities.
    embeddings, embedding_ids, embedding_space : path-like, optional
        Read-only float16/float32 NPY matrix, row IDs, and encoder-space JSON.
    encoder_path : path-like, optional
        Alternative local DNABERT-S weights and tokenizer. Inference is frozen,
        lazy and cached on the declared device; no weights are downloaded or copied.
    encoder_options : dict, optional
        Explicit frozen encoder pooling, window, storage and execution settings.
    donor_ids : path-like, optional
        Explicit eligible donor IDs, one per line. Binned contigs remain protected.
    input_format : {'generic', 'mmlong2'}
        Input discovery adapter. Explicit paths override mmlong2 discovery.
    input_root : path-like, optional
        mmlong2 sample directory when using that adapter.
    limits : ResourceLimits, optional
        Recruitment allocation limits.

    Returns
    -------
    Dataset
        Prepared artifact referencing the original assembly and embedding files.
    """
    limits = limits or ResourceLimits()
    if input_format == "mmlong2":
        if input_root is None:
            raise ValueError("mmlong2 input requires input_root")
        discovered = mmlong2_inputs(
            input_root, assembly=assembly, bins=bins, memberships=memberships, coverage=coverage
        )
        assembly, bins, memberships, coverage = (
            discovered[k] for k in ("assembly", "bins", "memberships", "coverage")
        )
    elif input_format != "generic":
        raise ValueError("Unknown input format")
    if assembly is None or coverage is None or (bins is None) == (memberships is None):
        raise ValueError("Supply assembly, coverage, and exactly one of bins or memberships")
    assembly, coverage = Path(assembly).resolve(), Path(coverage).resolve()
    paths = [assembly, coverage]
    bin_paths = [] if bins is None else bin_fasta_paths(bins)
    paths.extend(p.resolve() for p in bin_paths)
    paths.extend(
        Path(p).resolve()
        for p in (
            memberships,
            groups,
            source_ids,
            composition,
            embeddings,
            embedding_ids,
            embedding_space,
            donor_ids,
        )
        if p is not None
    )
    shared_index = Dataset(Path(assembly_index).resolve()) if assembly_index is not None else None
    if shared_index is not None:
        shared_index.verify(inputs=False)
        paths.extend([shared_index.path / "complete.json", shared_index.path / "dataset.json"])
    local_embedding = None
    if encoder_path is not None:
        from .representations.dna import local_contract

        if any(p is not None for p in (embeddings, embedding_ids, embedding_space)):
            raise ValueError("Choose precomputed embeddings or an explicit local encoder")
        space, model_files = local_contract(encoder_path, encoder_options)
        local_embedding = {
            "provider": "frozen_local",
            "space": space,
            "model_path": str(Path(encoder_path).resolve()),
            "model_files": model_files,
            "space_identity": fingerprint(space),
            "execution": {
                "device": (encoder_options or {}).get("device", "cpu"),
                "batch_size": (encoder_options or {}).get("batch_size", 8),
                "precision": (encoder_options or {}).get("precision", "float32"),
                "storage_dtype": (encoder_options or {}).get("storage_dtype", "float32"),
            },
        }
        paths.extend(Path(path) for path in model_files)
    elif encoder_options:
        raise ValueError("Encoder options require an explicit encoder_path")
    root = Path(output).resolve()
    if any(p.is_relative_to(root) for p in paths):
        raise ValueError("Prepared output must not contain its input files")
    identities = {str(p): sha(p) for p in paths}
    shared_arrays = []
    if shared_index is not None:
        source = shared_index.manifest
        if source["inputs_sha256"][source["assembly"]] != identities[str(assembly)]:
            raise ValueError("Shared assembly index requires byte-identical FASTA input")
        source_coverage = source.get("coverage_source")
        if (
            source_coverage is not None
            and source["inputs_sha256"][source_coverage] == identities[str(coverage)]
            and source.get("coverage_profile") == coverage_profile
        ):
            shared_arrays.append("coverage.npy")
        source_composition = source.get("composition_source")
        if (
            composition is not None
            and source_composition is not None
            and source["inputs_sha256"][source_composition] == identities[str(Path(composition).resolve())]
        ):
            shared_arrays.append("composition.npy")
    identity = fingerprint(
        {
            "inputs": identities,
            "coverage_profile": coverage_profile,
            "input_format": input_format,
            "schema": 1,
            "encoder": local_embedding,
        }
    )
    with FileLock(root.with_name(root.name + ".prepare.lock")):
        if (root / "complete.json").exists():
            dataset = Dataset(root)
            if dataset.verify(inputs=False)["identity"] != identity:
                raise ValueError("Prepared output already belongs to different inputs")
            return dataset
        if root.exists() and any(root.iterdir()):
            raise ValueError("Incomplete prepared directory exists; preserve it and choose a new output")
        root.mkdir(parents=True, exist_ok=True)
        if shared_index is not None:
            limits.require(shared_index.manifest["contigs"] * 768, "Shared assembly metadata")
            ids = np.load(shared_index.array_path("ids.npy"), allow_pickle=False).tolist()
        else:
            from .sequence_io import scan_records

            ids, lengths, sequence_hashes = [], [], []
            with sqlite3.connect(root / "sequences.sqlite") as db:
                db.execute(
                    "CREATE TABLE sequence(row INTEGER PRIMARY KEY,id TEXT UNIQUE,start INTEGER,end INTEGER,"
                    "bp INTEGER,sha TEXT,line_bases INTEGER,line_bytes INTEGER)"
                )
                for record in scan_records(assembly, limits):
                    limits.require((len(ids) + 1) * 768, "Dataset metadata")
                    digest = hashlib.sha256(record.sequence).hexdigest()
                    try:
                        db.execute(
                            "INSERT INTO sequence VALUES (?,?,?,?,?,?,?,?)",
                            (
                                len(ids),
                                record.name,
                                record.start,
                                record.end,
                                len(record.sequence),
                                digest,
                                record.line_bases,
                                record.line_bytes,
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate assembly ID: {record.name}") from error
                    ids.append(record.name)
                    lengths.append(len(record.sequence))
                    sequence_hashes.append(digest)
                db.commit()
            np.save(root / "ids.npy", np.array(ids))
            np.save(root / "lengths.npy", np.array(lengths, np.int64))
            np.save(root / "sequence_hashes.npy", np.array(sequence_hashes, dtype="S64"))
        if not ids:
            raise ValueError("Empty assembly")
        positions = {name: i for i, name in enumerate(ids)}
        bin_members: dict[str, list[str]] = {}
        assigned: set[str] = set()

        def assign(name: str, bin_id: str, *, table: bool = False) -> None:
            if table and bin_id in {"", "unbinned", "unassigned", "-", "NA", "0"}:
                return
            if name not in positions or name in assigned:
                raise ValueError(f"Unknown or multiply assigned bin contig: {name}")
            assigned.add(name)
            bin_members.setdefault(bin_id, []).append(name)

        if memberships is not None:
            for name, bin_id in _membership_rows(Path(memberships)):
                assign(name, bin_id, table=True)
        else:
            if shared_index is not None:
                shared_hashes = np.load(shared_index.array_path("sequence_hashes.npy"), mmap_mode="r")
                sequence_hashes = [value.decode() for value in shared_hashes]
                shared_hashes._mmap.close()
            for path in bin_paths:
                if path.stem in bin_members:
                    raise ValueError(f"Duplicate bin ID: {path.stem}")
                for name, sequence, _, _ in fasta_records(path, limits):
                    if (
                        name not in positions
                        or hashlib.sha256(sequence).hexdigest() != sequence_hashes[positions[name]]
                    ):
                        raise ValueError(f"Bin sequence differs from the assembly: {name}")
                    assign(name, path.stem)
                if path.stem not in bin_members:
                    raise ValueError(f"Empty bin FASTA: {path}; no original bin may be silently omitted")
        if not bin_members:
            raise ValueError("No assigned bins")
        group_map = {name: name for name in bin_members}
        if groups is not None:
            with Path(groups).open(encoding="utf-8") as stream:
                pairs = [line.rstrip("\n").split("\t") for line in stream if line.strip()]
            if pairs and pairs[0] in (["bin_id", "group_id"], ["bin", "group"]):
                pairs = pairs[1:]
            group_map = dict(pairs)
            if (
                len(group_map) != len(pairs)
                or set(group_map) != set(bin_members)
                or not all(group_map.values())
            ):
                raise ValueError("Related groups must map every bin exactly once")
        pool = [name for name in ids if name not in assigned]
        source_map = {name: name for name in bin_members}
        if source_ids is not None:
            with Path(source_ids).open(encoding="utf-8") as stream:
                pairs = [line.rstrip("\n").split("\t") for line in stream if line.strip()]
            if pairs and pairs[0] == ["bin_id", "source_id"]:
                pairs = pairs[1:]
            source_map = dict(pairs)
            if (
                len(source_map) != len(pairs)
                or set(source_map) != set(bin_members)
                or not all(source_map.values())
                or len(set(source_map.values())) != len(source_map)
            ):
                raise ValueError("Source IDs must map every bin exactly once to a unique nonempty ID")
        if donor_ids is not None:
            requested = Path(donor_ids).read_text(encoding="utf-8").splitlines()
            if len(requested) != len(set(requested)) or not set(requested) <= set(pool):
                raise ValueError("Explicit donors must be unique, unbinned assembly contigs")
            pool = requested
        representation = local_embedding
        if any(p is not None for p in (embeddings, embedding_ids, embedding_space)):
            if any(p is None for p in (embeddings, embedding_ids, embedding_space)):
                raise ValueError(
                    "Embeddings require the array, row IDs and embedding-space manifest together"
                )
            matrix = np.load(embeddings, mmap_mode="r", allow_pickle=False)
            try:
                rows = Path(embedding_ids).read_text(encoding="utf-8").splitlines()
                limits.require(len(rows) * 512, "Embedding row index")
                if (
                    matrix.ndim != 2
                    or matrix.dtype not in (np.dtype("float16"), np.dtype("float32"))
                    or len(rows) != len(matrix)
                    or len(rows) != len(set(rows))
                ):
                    raise ValueError("Embedding array must be float16/float32 with one unique ID per row")
                by_id = {name: i for i, name in enumerate(rows)}
                if not set(ids) <= set(by_id):
                    raise ValueError("Embedding population omits assembly contigs")
                space = read(embedding_space)
                required = {
                    "encoder",
                    "encoder_sha256",
                    "tokenizer_sha256",
                    "pooling",
                    "window_bp",
                    "windows",
                    "max_tokens",
                    "normalization",
                    "dimension",
                }
                if (
                    not required <= set(space)
                    or space["dimension"] != matrix.shape[1]
                    or space["normalization"] != "l2"
                ):
                    raise ValueError("Supply the complete encoder-space contract with L2 normalization")
                width = min(limits.chunk_size, max(1, limits.memory_bytes // max(1, matrix.shape[1] * 32)))
                tolerance = 5e-4 if matrix.dtype == np.float16 else 1e-4
                maximum_norm_error = 0.0
                for a in range(0, len(rows), width):
                    # Validate reduced-precision storage in float32 blocks, retaining
                    # the original matrix and its values without renormalization.
                    block = np.asarray(matrix[a : a + width], np.float32)
                    norms = np.linalg.norm(block, axis=1)
                    if not np.isfinite(block).all() or not np.allclose(norms, 1, atol=tolerance, rtol=0):
                        raise ValueError("Embedding rows must be finite and already L2 normalized")
                    if len(norms):
                        maximum_norm_error = max(maximum_norm_error, float(np.abs(norms - 1).max()))
                np.save(root / "embedding_rows.npy", np.array([by_id[name] for name in ids], np.int64))
                representation = {
                    "path": str(Path(embeddings).resolve()),
                    "space": space,
                    "space_identity": fingerprint(space),
                    "array_sha256": identities[str(Path(embeddings).resolve())],
                    "storage_dtype": str(matrix.dtype),
                    "normalization_validation_atol": tolerance,
                    "maximum_norm_error": maximum_norm_error,
                }
            finally:
                matrix._mmap.close()
        if "coverage.npy" not in shared_arrays:
            np.save(root / "coverage.npy", _coverage(coverage, ids, coverage_profile))
        if composition is not None and "composition.npy" not in shared_arrays:
            np.save(root / "composition.npy", _composition(Path(composition), ids, limits))
        coverage_array = np.load(
            shared_index.array_path("coverage.npy")
            if "coverage.npy" in shared_arrays
            else root / "coverage.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        coverage_variance = bool(np.isfinite(coverage_array[:, 1]).all())
        coverage_array._mmap.close()
        np.save(root / "pool.npy", np.array([positions[name] for name in pool], np.int64))
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "assembly": str(assembly),
            "assembly_index": {
                "path": str(shared_index.path),
                "receipt_sha256": sha(shared_index.path / "complete.json"),
            }
            if shared_index is not None
            else None,
            "inputs_sha256": identities,
            "input_format": input_format,
            "bins": {name: sorted(members) for name, members in sorted(bin_members.items())},
            "groups": group_map,
            "groups_declared": groups is not None,
            "source_ids": source_map,
            "bin_fastas": {p.stem: str(p.resolve()) for p in bin_paths},
            "embedding": representation,
            "contigs": len(ids),
            "candidate_contigs": len(pool),
            "coverage_variance": coverage_variance,
            "coverage_source": str(coverage),
            "coverage_profile": coverage_profile,
            "shared_arrays": shared_arrays,
            "composition_source": str(Path(composition).resolve()) if composition is not None else None,
            "training_policy": "original membership only; caller supplies unrepaired bins",
        }
        save(root / "dataset.json", manifest)
        complete(root, {"kind": "dataset", "identity": identity})
        return Dataset(root)
