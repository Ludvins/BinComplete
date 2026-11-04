"""External-user cases, real dependencies and tabular review boundaries."""

import csv
import json
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from bincomplete import candidate_table, export, fit, prepare_dataset, propose, reconcile
from bincomplete.artifacts import read, save, sha
from bincomplete.cli import main
from bincomplete.data import fasta_records
from bincomplete.demo import demo
from bincomplete.preflight import preflight


def test_distribution_examples_and_environment_are_project_neutral():
    from importlib.resources import files

    root = Path(__file__).resolve().parents[1]
    for folder in (root / "configs", root / "docs"):
        assert not (folder / "archive").exists()
    environment = json.loads(
        files("bincomplete").joinpath("profiles/standard_v1/tested_environment.json").read_text()
    )
    assert set(environment) == {"schema_version", "recipe", "python", "packages", "scope"}
    for path in (root / "configs").rglob("*.json"):
        values = json.loads(path.read_text())
        assert "result" not in values
        assert "results" not in values


@pytest.mark.parametrize("label", ["0", "NA", "unassigned", "unbinned"])
def test_fasta_bin_names_are_literal_and_all_original_members_stay_protected(
    dataset, tmp_path, limits, label
):
    parent = dataset.path.parent
    sequences = {
        name: sequence.decode() for name, sequence, _, _ in fasta_records(parent / "assembly.fa", limits)
    }
    directory = tmp_path / "bins"
    directory.mkdir()
    for index, name in enumerate([label, "first", "second", "third"]):
        (directory / f"{name}.fa").write_text(
            "".join(f">c{i}\n{sequences[f'c{i}']}\n" for i in (2 * index, 2 * index + 1))
        )
    inputs = {
        "assembly": str(parent / "assembly.fa"),
        "bins": str(directory),
        "coverage": str(parent / "coverage.tsv"),
        "embeddings": str(parent / "embedding.npy"),
        "embedding_ids": str(parent / "embedding_ids.txt"),
        "embedding_space": str(parent / "space.json"),
    }
    prepared = prepare_dataset(tmp_path / "prepared", **inputs)
    assert len(prepared.manifest["bins"]) == 4 and label in prepared.manifest["bins"]
    assert prepared.manifest["candidate_contigs"] == 6
    save(
        tmp_path / "experiment.json",
        {
            "schema_version": 1,
            "output": "unused",
            "training": "sample",
            "evaluation": "sample",
            "methods": [{"method": "cosine"}],
            "datasets": {"sample": inputs},
        },
    )
    report = preflight(tmp_path / "experiment.json")
    assert report["ready"] and report["datasets"]["sample"]["bins"] == 4
    assert report["datasets"]["sample"]["protected_contigs"] == 8
    model = fit(prepared, tmp_path / "model", method="cosine", limits=limits, cache=tmp_path / "cache")
    props = propose(model, prepared, tmp_path / "proposals", limits=limits, cache=tmp_path / "cache")
    assert label in props.manifest["coverage"]
    for line in (props.path / "memberships.jsonl").read_text().splitlines():
        row = json.loads(line)
        core = set(prepared.manifest["bins"][row["recipient"]])
        assert core <= set(row["members"])
        assert all(name not in {f"c{i}" for i in range(8)} for name in set(row["members"]) - core)


def test_membership_table_zero_retains_documented_unassigned_semantics(dataset, tmp_path):
    parent = dataset.path.parent
    members = tmp_path / "members.tsv"
    members.write_text("contig_id\tbin_id\nc0\t0\nc1\tfirst\nc2\tfirst\n")
    result = prepare_dataset(
        tmp_path / "prepared",
        assembly=parent / "assembly.fa",
        memberships=members,
        coverage=parent / "coverage.tsv",
    )
    assert result.manifest["bins"] == {"first": ["c1", "c2"]}
    assert result.manifest["candidate_contigs"] == 12


def test_empty_bin_fasta_is_rejected(dataset, tmp_path):
    parent = dataset.path.parent
    directory = tmp_path / "bins"
    directory.mkdir()
    (directory / "empty.fa").write_text("")
    with pytest.raises(ValueError, match="Empty bin FASTA"):
        prepare_dataset(
            tmp_path / "prepared",
            assembly=parent / "assembly.fa",
            bins=directory,
            coverage=parent / "coverage.tsv",
        )


def test_local_encoder_preflight_requires_einops(dataset, tmp_path, monkeypatch):
    import bincomplete.preflight as module

    parent = dataset.path.parent
    model = tmp_path / "encoder"
    model.mkdir()
    save(model / "config.json", {"hidden_size": 8})
    save(model / "tokenizer.json", {"fixture": True})
    (model / "pytorch_model.bin").write_bytes(b"Not loaded by preflight")
    save(
        tmp_path / "config.json",
        {
            "schema_version": 1,
            "output": "unused",
            "training": "sample",
            "evaluation": "sample",
            "methods": [{"method": "cosine"}],
            "datasets": {
                "sample": {
                    "assembly": str(parent / "assembly.fa"),
                    "memberships": str(parent / "members.tsv"),
                    "coverage": str(parent / "coverage.tsv"),
                    "encoder_path": str(model),
                }
            },
        },
    )
    real_version = module.version

    def missing(name):
        if name == "einops":
            raise PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(module, "version", missing)
    report = preflight(tmp_path / "config.json")
    assert not report["ready"] and "Missing dependency einops" in " ".join(report["errors"])
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert "einops>=0.8.2" in project["project"]["optional-dependencies"]["embeddings"]


def test_candidate_and_export_tables_use_exact_sequence_keys_and_no_quality_claims(
    dataset, tmp_path, limits, capsys
):
    model = fit(dataset, tmp_path / "model", method="cosine", limits=limits, cache=tmp_path / "cache")
    props = propose(model, dataset, tmp_path / "props", limits=limits, cache=tmp_path / "cache")
    output = candidate_table([props], tmp_path / "candidates.tsv", limits=limits)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == props.manifest["proposal_count"]
    assert all(row["quality_status"] == "UNASSESSED" and len(row["sequence_set_id"]) == 64 for row in rows)
    before = sha(output)
    assert main(["candidates", "--proposals", str(props.path), "--output", str(output)]) == 0
    assert sha(output) == before
    exported = export(props, tmp_path / "exported", proposal_ids=[rows[0]["proposal_id"]], limits=limits)
    with exported.with_name("bins.tsv").open(newline="", encoding="utf-8") as stream:
        mapped = list(csv.DictReader(stream, delimiter="\t"))
    assert len(mapped) == 1 and mapped[0]["proposal_id"] == rows[0]["proposal_id"]
    assert mapped[0]["quality_verdict"] == "UNASSESSED"
    assert sha(exported.parent / mapped[0]["fasta"]) == mapped[0]["sha256"]
    selection = reconcile([props], [rows[0]["proposal_id"]], tmp_path / "selection", limits=limits)
    final = export(selection, tmp_path / "final", complete_directory=True, limits=limits)
    with final.with_name("bins.tsv").open(newline="", encoding="utf-8") as stream:
        final_rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(final_rows) == len(dataset.manifest["bins"])
    for row in final_rows:
        assert row["fasta"].startswith("bins/")
        assert sha(final.parent / row["fasta"]) == row["sha256"]
    output.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="has changed"):
        candidate_table([props], output)


def test_public_demo_runs_all_six_and_resumes_without_refitting(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    result = demo(tmp_path / "demo")
    assert read(result)["status"] == "PASSED" and read(result)["methods"] == 6
    assert read(result)["known_hq_gains"] is None
    status = result.parent / "results/status.json"
    before = sha(status)
    from bincomplete.scorers.vector import FeatureClassifier

    monkeypatch.setattr(FeatureClassifier, "fit", lambda *args: pytest.fail("Unexpected refit"))
    assert main(["demo", "--output", str(result.parent)]) == 0
    assert sha(status) == before
