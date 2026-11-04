"""Shared indexes remain immutable while donor/membership contexts stay separate."""

import json

import numpy as np
import pytest

from bincomplete import ResourceLimits, prepare_dataset


def test_shared_preparation_reuses_metadata_and_keeps_pool_separate(dataset, tmp_path, monkeypatch):
    from bincomplete import sequence_io

    parent = dataset.path.parent
    members = tmp_path / "other_members.tsv"
    members.write_text("contig_id\tbin_id\nc8\tx\nc9\tx\n")

    def rescan(*args):
        raise AssertionError("Shared assembly must not be rescanned")

    monkeypatch.setattr(sequence_io, "scan_records", rescan)
    data = prepare_dataset(
        tmp_path / "shared",
        assembly=parent / "assembly.fa",
        memberships=members,
        coverage=parent / "coverage.tsv",
        assembly_index=dataset.path,
    )
    data.verify()
    for filename in ["ids.npy", "lengths.npy", "sequence_hashes.npy", "sequences.sqlite", "coverage.npy"]:
        assert not (data.path / filename).exists()
        assert data.array_path(filename) == dataset.array_path(filename)
    assert data.manifest["bins"] == {"x": ["c8", "c9"]}
    assert set(np.load(data.array_path("pool.npy"))) != set(np.load(dataset.array_path("pool.npy")))
    assert next(data.sequences([0], ResourceLimits()))


def test_shared_index_rejects_different_assembly_and_detects_metadata_tampering(dataset, tmp_path):
    parent = dataset.path.parent
    other = tmp_path / "different.fa"
    other.write_bytes((parent / "assembly.fa").read_bytes().replace(b">c0", b">d0", 1))
    with pytest.raises(ValueError, match="byte-identical"):
        prepare_dataset(
            tmp_path / "bad",
            assembly=other,
            memberships=parent / "members.tsv",
            coverage=parent / "coverage.tsv",
            assembly_index=dataset.path,
        )
    shared = prepare_dataset(
        tmp_path / "shared",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        assembly_index=dataset.path,
    )
    with (dataset.path / "ids.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="Artifact output changed"):
        shared.verify(inputs=False)


def test_shared_composition_is_reused_but_changed_coverage_is_local(dataset, tmp_path):
    parent = dataset.path.parent
    composition = tmp_path / "composition.tsv"
    ids = np.load(dataset.array_path("ids.npy"))
    composition.write_text(
        "contig_id\t"
        + "\t".join(map(str, range(136)))
        + "\n"
        + "".join(str(name) + "\t" + "\t".join(["1"] * 136) + "\n" for name in ids)
    )
    first = prepare_dataset(
        tmp_path / "first",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        composition=composition,
        assembly_index=dataset.path,
    )
    changed = tmp_path / "coverage.tsv"
    changed.write_text("contig_id\tmean\tvariance\n" + "".join(f"{name}\t77\t3\n" for name in ids))
    second = prepare_dataset(
        tmp_path / "second",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=changed,
        composition=composition,
        assembly_index=first.path,
    )
    second.verify()
    assert second.array_path("composition.npy") == first.array_path("composition.npy")
    assert not (second.path / "composition.npy").exists()
    assert (second.path / "coverage.npy").exists()
    np.testing.assert_array_equal(np.load(second.array_path("coverage.npy"))[:, 0], 77)


def test_legacy_indexes_can_be_shared_and_read_windows(dataset, tmp_path):
    import sqlite3

    from bincomplete import ResourceLimits
    from bincomplete.artifacts import sha

    # Simulate the previous schema and refresh only this synthetic artifact receipt.
    db_path = dataset.path / "sequences.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("ALTER TABLE sequence DROP COLUMN line_bases")
        db.execute("ALTER TABLE sequence DROP COLUMN line_bytes")
        db.commit()
    receipt = json.loads((dataset.path / "complete.json").read_text())
    receipt["outputs_sha256"]["sequences.sqlite"] = sha(db_path)
    (dataset.path / "complete.json").write_text(json.dumps(receipt))
    parent = dataset.path.parent
    shared = prepare_dataset(
        tmp_path / "shared",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        assembly_index=dataset.path,
    )
    limits = ResourceLimits()
    full = next(shared.sequences([0], limits))
    assert next(shared.sequence_windows([0], limits, window_bp=10, count=1)) == [full[:10]]
