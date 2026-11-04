"""Export every original TSV bin as a control without selecting any additions."""

import csv
import json

from bincomplete import fit, propose
from bincomplete.artifacts import sha
from bincomplete.cli import main
from bincomplete.data import fasta_records
from bincomplete.reconcile import SelectionArtifact


def test_empty_approval_exports_exact_original_controls(dataset, tmp_path, limits):
    prepared = dataset.manifest
    assert prepared["bin_fastas"] == {}
    originals = prepared["bins"]
    assembly = {name: sequence for name, sequence, _, _ in fasta_records(prepared["assembly"], limits)}
    original_members = {name for members in originals.values() for name in members}
    donors = set(assembly) - original_members
    assert donors
    model = fit(
        dataset,
        tmp_path / "model",
        method="cosine",
        limits=limits,
        cache=tmp_path / "cache",
    )
    assert model.manifest["optimizer_calls"] == 0
    proposals = propose(
        model,
        dataset,
        tmp_path / "proposals",
        recipients=["b0"],
        limits=limits,
        cache=tmp_path / "cache",
    )
    proposal_rows = [
        json.loads(line) for line in (proposals.path / "memberships.jsonl").read_text().splitlines()
    ]
    assert proposal_rows
    assert {row["recipient"] for row in proposal_rows} == {"b0"}
    assert any(set(row["added"]) & donors for row in proposal_rows)
    proposal_files = {path: sha(path) for path in proposals.path.rglob("*") if path.is_file()}
    approvals = tmp_path / "control_approvals_empty.txt"
    approvals.write_text("", encoding="utf-8")
    selection_path = tmp_path / "baseline_control_selection"
    resource_args = ["--threads", "1", "--memory-gib", "0.5", "--cache-gib", "0.01", "--chunk-size", "3"]
    assert (
        main(
            [
                "reconcile",
                "--proposals",
                str(proposals.path),
                "--approved-ids",
                str(approvals),
                "--output",
                str(selection_path),
                *resource_args,
            ]
        )
        == 0
    )
    selection = SelectionArtifact(selection_path).manifest
    assert selection["selected"] == []
    assert selection["compatible_recipients"] == 0
    output = tmp_path / "baseline_original_controls"
    assert (
        main(
            [
                "export",
                "--selection",
                str(selection_path),
                "--complete-directory",
                "--output",
                str(output),
                *resource_args,
            ]
        )
        == 0
    )
    manifest = json.loads((output / "export.json").read_text(encoding="utf-8"))
    assert manifest["changed"] == []
    assert {entry["recipient"] for entry in manifest["unchanged"]} == set(originals)
    with (output / "bins.tsv").open(encoding="utf-8", newline="") as stream:
        controls = list(csv.DictReader(stream, delimiter="\t"))
    assert len(controls) == len(originals)
    assert {row["recipient"] for row in controls} == set(originals)
    exported_paths = set()
    for row in controls:
        assert row["proposal_id"] == ""
        assert row["changed"] == "false"
        assert row["quality_verdict"] == "UNCHANGED_CONTROL"
        path = output / row["fasta"]
        exported_paths.add(path)
        assert sha(path) == row["sha256"]
        records = [(name, sequence) for name, sequence, _, _ in fasta_records(path, limits)]
        members = originals[row["recipient"]]
        assert len(records) == len(members)
        assert dict(records) == {name: assembly[name] for name in members}
        assert not {name for name, _ in records} & donors
    assert exported_paths == set((output / "bins").glob("*.fa"))
    assert all(sha(path) == digest for path, digest in proposal_files.items())
