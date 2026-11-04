import json
from types import SimpleNamespace

import numpy as np
import pytest

from bincomplete import ResourceLimits, prepare_dataset
from bincomplete.artifacts import fingerprint, save, sha
from bincomplete.cache import DerivedCache
from bincomplete.data import fasta_records, mmlong2_inputs
from bincomplete.retrieval.exact import ExactDistance


def test_duplicate_ids_and_memberships_rejected(tmp_path):
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">a\nACGT\n>a\nTTTT\n")
    memberships = tmp_path / "members.tsv"
    memberships.write_text("a\tfirst\na\tsecond\n")
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text("contig_id\tmean\tvariance\na\t1\t0\n")
    with pytest.raises(ValueError, match="Duplicate assembly ID"):
        prepare_dataset(tmp_path / "duplicate", assembly=fasta, memberships=memberships, coverage=coverage)
    fasta.write_text(">a\nACGT\n>b\nTTTT\n")
    with pytest.raises(ValueError, match="multiply assigned"):
        prepare_dataset(tmp_path / "membership", assembly=fasta, memberships=memberships, coverage=coverage)


def test_bin_fasta_sequence_identity_enforced(tmp_path):
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">a\nACGT\n")
    bins = tmp_path / "bins"
    bins.mkdir()
    (bins / "one.fa").write_text(">a\nAAAA\n")
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text("contig_id\tmean\na\t1\n")
    with pytest.raises(ValueError, match="differs from the assembly"):
        prepare_dataset(tmp_path / "prepared", assembly=fasta, bins=bins, coverage=coverage)


def test_explicit_donors_cannot_borrow_other_bins(dataset, tmp_path):
    root = dataset.path.parent
    donors = tmp_path / "donors.txt"
    donors.write_text("c0\n")
    with pytest.raises(ValueError, match="unbinned"):
        prepare_dataset(
            tmp_path / "bad",
            assembly=root / "assembly.fa",
            memberships=root / "members.tsv",
            coverage=root / "coverage.tsv",
            donor_ids=donors,
        )


def test_mmlong_adapter_discovers_final_files_and_requires_profile(tmp_path):
    root = tmp_path / "sample"
    results = root / "results"
    (results / "bins").mkdir(parents=True)
    (results / "sample_assembly.fasta").write_text(">c0\nACGTACGT\n>free\nAAAA\n")
    (results / "bins/bin1.fa").write_text(">c0\nACGTACGT\n")
    mapping = root / "tmp/binning/mapping"
    mapping.mkdir(parents=True)
    coverage = mapping / "1-depth.tsv"
    coverage.write_text(
        "contigName\tcontigLen\ttotalAvgDepth\tx.bam\tx.bam-var\ty.bam\ty.bam-var\n"
        "c0\t8\t10\t7\t2\t3\t1\nfree\t4\t8\t6\t1\t2\t1\n"
    )
    assert mmlong2_inputs(root)["assembly"].name == "sample_assembly.fasta"
    with pytest.raises(ValueError, match="one JGI coverage profile"):
        prepare_dataset(tmp_path / "ambiguous", input_format="mmlong2", input_root=root)
    prepared = prepare_dataset(
        tmp_path / "prepared", input_format="mmlong2", input_root=root, coverage_profile="x.bam"
    )
    np.testing.assert_array_equal(np.load(prepared.path / "coverage.npy"), [[7, 2], [6, 1]])
    assert prepared.manifest["candidate_contigs"] == 1
    assert not list(prepared.path.glob("*.fa"))


def test_sequence_and_fit_budgets_fail_explicitly(tmp_path, dataset, settings):
    from bincomplete import fit

    fasta = tmp_path / "large.fa"
    fasta.write_text(">x\n" + "A" * 5000)
    tiny = ResourceLimits(memory_gib=0.000001)
    with pytest.raises(MemoryError, match="Sequence buffer"):
        list(fasta_records(fasta, tiny))
    settings["training"] = {"max_training_rows": 1}
    with pytest.raises(ValueError, match="row budget"):
        fit(
            dataset,
            tmp_path / "model",
            method="kmers",
            settings=settings,
            limits=ResourceLimits(threads=1),
            cache=tmp_path / "cache",
        )


def test_derived_cache_quota_evicts_only_unleased_derived_entries(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    sentinel = root / "user-input.txt"
    sentinel.write_text("preserve")
    limits = ResourceLimits(cache_gib=24000 / 1024**3)
    first = DerivedCache(root, limits)

    def writer(path):
        np.save(path, np.ones(2000, np.float32))

    first.array({"test": 1}, writer, expected_bytes=8000)
    second = DerivedCache(root, limits)
    second.array({"test": 2}, writer, expected_bytes=8000)
    with pytest.raises(MemoryError, match="pinned"):
        second.array({"test": 3}, writer, expected_bytes=8000)
    first.close()
    second.array({"test": 3}, writer, expected_bytes=8000)
    second.close()
    assert sentinel.read_text() == "preserve"
    assert sum(path.stat().st_size for path in (root / "arrays").rglob("*") if path.is_file()) <= 24000


def test_derived_cache_corruption_does_not_silently_recompute(tmp_path):
    limits = ResourceLimits()
    cache = DerivedCache(tmp_path / "cache", limits)
    cache.array({"test": 1}, lambda path: np.save(path, np.ones(2)), expected_bytes=16)
    cache.close()
    next((tmp_path / "cache/arrays").glob("*/values.npy")).write_bytes(b"corrupt")
    other = DerivedCache(tmp_path / "cache", limits)
    try:
        with pytest.raises(ValueError, match="integrity"):
            other.array({"test": 1}, lambda path: pytest.fail("unexpected recompute"), expected_bytes=16)
    finally:
        other.close()


@pytest.mark.parametrize("entry", ["foreign_name", "incomplete", "malformed", "wrong_identity", "extra_file"])
def test_cache_eviction_preserves_unrecognized_entries(tmp_path, entry):
    root = tmp_path / "cache"
    identity = {"test": "unrecognized"}
    folder = root / "arrays" / ("user-data" if entry == "foreign_name" else fingerprint(identity))
    folder.mkdir(parents=True)
    values = folder / "values.npy"
    np.save(values, np.ones(2000, np.float32))
    marker = folder / "complete.json"
    if entry != "incomplete":
        save(marker, {"identity": identity, "sha256": sha(values)})
    if entry == "malformed":
        marker.write_text("invalid JSON")
    elif entry == "wrong_identity":
        save(marker, {"identity": {"different": True}, "sha256": sha(values)})
    elif entry == "extra_file":
        (folder / "user-input.txt").write_text("preserve")
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    cache = DerivedCache(root, ResourceLimits(cache_gib=16000 / 1024**3))
    try:
        with pytest.raises(MemoryError, match="not owned"):
            cache.array(
                {"new": True}, lambda path: np.save(path, np.ones(2000, np.float32)), expected_bytes=8000
            )
    finally:
        cache.close()
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_million_row_retrieval_never_materializes_the_embedding_population(tmp_path):
    class VirtualVectors:
        shape = (1000000, 8)
        ndim = 2
        largest_request = 0

        def __getitem__(self, rows):
            self.largest_request = max(self.largest_request, len(rows))
            return np.ones((len(rows), 8), np.float32)

        def __array__(self, *args):
            raise AssertionError("Entire embedding population was materialized")

    class VirtualIDs:
        def __getitem__(self, rows):
            return np.array([f"c{int(row):07d}" for row in rows])

    count = 1000000
    coverage = np.full(count, 10.0)
    store = SimpleNamespace(ids=VirtualIDs(), lengths=np.full(count, 2000), array=lambda name: coverage)
    vectors = VirtualVectors()
    retriever = ExactDistance(top_k=10, min_length=0, chunk_size=257)
    result = retriever.retrieve(np.array([0]), np.arange(1, count), vectors, store)
    np.testing.assert_array_equal(result.indices, np.arange(1, 11))
    assert vectors.largest_request <= 257
    (tmp_path / "scale.json").write_text(
        json.dumps({"rows": count, "largest_vector_block": vectors.largest_request})
    )
