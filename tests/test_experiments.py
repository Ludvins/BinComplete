"""Experiment orchestration uses fresh synthetic inputs and preserves failures."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from bincomplete import run_experiment
from bincomplete.artifacts import read, save, sha
from bincomplete.cli import main
from bincomplete.experiments.config import load_plan
from bincomplete.resources import FileLock


def configuration(dataset, tmp_path, settings, methods=("kmers", "cosine"), *, raw=False):
    parent = dataset.path.parent
    inputs = {"prepared": str(dataset.path)}
    if raw:
        inputs = {
            key: str(parent / value)
            for key, value in {
                "assembly": "assembly.fa",
                "memberships": "members.tsv",
                "coverage": "coverage.tsv",
                "embeddings": "embedding.npy",
                "embedding_ids": "embedding_ids.txt",
                "embedding_space": "space.json",
            }.items()
        }
    entries = []
    for method in methods:
        values = deepcopy(settings)
        if method in {"cosine", "single"}:
            values.pop("classifier")
        if method == "single":
            values.pop("masker")
            values["scorer"] = {
                "params": {
                    "max_epochs": 1,
                    "validation_fraction": 0,
                    "width": 16,
                    "heads": 2,
                    "layers": 1,
                    "feedforward": 24,
                    "batch_size": 2,
                    "repeats_per_source": 1,
                }
            }
        entries.append({"method": method, "settings": values})
    value = {
        "schema_version": 1,
        "output": str(tmp_path / "experiment"),
        "training": "sample",
        "evaluation": "sample",
        "datasets": {"sample": inputs},
        "methods": entries,
        "resources": {"threads": 1, "memory_gib": 0.5, "cache_gib": 0.01, "chunk_size": 3},
    }
    path = tmp_path / "comparison.json"
    save(path, value)
    return path, value


def test_all_methods_share_preparation_and_cache_and_resume(dataset, tmp_path, settings, monkeypatch):
    pytest.importorskip("torch")
    from bincomplete import context
    from bincomplete.experiments import runner
    from bincomplete.scorers.sets import SetScorer
    from bincomplete.scorers.vector import FeatureClassifier

    config, _ = configuration(
        dataset, tmp_path, settings, ("kmers", "dnabert_s", "single", "cosine", "hgb_full_pool"), raw=True
    )
    original_prepare, original_frequencies = runner.prepare_dataset, context.frequencies
    calls = {"prepare": 0, "composition": 0}

    def prepare(*args, **kwargs):
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def frequencies(*args, **kwargs):
        calls["composition"] += 1
        return original_frequencies(*args, **kwargs)

    monkeypatch.setattr(runner, "prepare_dataset", prepare)
    monkeypatch.setattr(context, "frequencies", frequencies)
    assert main(["experiment", "--config", str(config)]) == 0
    root = tmp_path / "experiment"
    summary = read(root / "summary.json")
    assert summary["status"] == "SUCCEEDED"
    assert len(summary["methods"]) == 5
    assert calls == {"prepare": 1, "composition": 14}
    assert not list(root.rglob("*.fa"))
    comparison = read(root / "comparison.json")
    assert len(comparison["pairwise"]) == 10
    assert comparison["common_recipients_all_methods"] == 4
    assert comparison["quality_status"] == "NOT_EVALUATED_EXTERNAL_TOOLS_REQUIRED"
    assert comparison["distinct_construction_union"] > 0
    before = sha(root / "status.json")
    monkeypatch.setattr(FeatureClassifier, "fit", lambda *a, **k: pytest.fail("unexpected refit"))
    monkeypatch.setattr(SetScorer, "fit", lambda *a, **k: pytest.fail("unexpected refit"))
    assert run_experiment(config).manifest == summary
    assert calls == {"prepare": 1, "composition": 14}
    assert sha(root / "status.json") == before


def test_failed_proposals_preserve_attempt_and_reuse_completed_fit(dataset, tmp_path, settings, monkeypatch):
    from bincomplete.experiments import runner
    from bincomplete.scorers.vector import FeatureClassifier

    config, value = configuration(dataset, tmp_path, settings)
    original = runner.propose
    fail = True

    def propose(model, population, destination, **kwargs):
        if fail and "kmers" in destination.parts:
            destination.mkdir(parents=True)
            (destination / "partial_evidence.txt").write_text("preserve this")
            raise RuntimeError("simulated proposal interruption")
        return original(model, population, destination, **kwargs)

    monkeypatch.setattr(runner, "propose", propose)
    assert main(["experiment", "--config", str(config)]) == 1
    root = Path(value["output"])
    failed = read(root / "summary.json")
    assert failed["methods"]["kmers"]["status"] == "FAILED"
    assert failed["methods"]["cosine"]["status"] == "SUCCEEDED"
    assert not (root / "complete.json").exists()
    fit_receipt = sha(root / "methods/kmers/fit/0001/complete.json")
    fail = False
    monkeypatch.setattr(FeatureClassifier, "fit", lambda *a, **k: pytest.fail("completed fit was repeated"))
    result = run_experiment(config).manifest
    assert result["status"] == "SUCCEEDED"
    assert sha(root / "methods/kmers/fit/0001/complete.json") == fit_receipt
    assert (root / "methods/kmers/propose/0001/partial_evidence.txt").read_text() == "preserve this"
    assert result["methods"]["kmers"]["proposals"].endswith("0002")
    assert result["methods"]["cosine"]["proposals"].endswith("0001")
    assert list((root / "errors").glob("*.log"))


def test_completed_child_corruption_is_rejected_without_replacement(dataset, tmp_path, settings):
    config, _ = configuration(dataset, tmp_path, settings, ("cosine",))
    artifact = run_experiment(config)
    before = sha(artifact.path / "status.json")
    scores = next(artifact.path.glob("methods/cosine/propose/0001/rankings/*/scores.npy"))
    with scores.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run_experiment(config)
    assert sha(artifact.path / "status.json") == before
    assert not (artifact.path / "methods/cosine/propose/0002").exists()


@pytest.mark.parametrize(
    "change", ["no_methods", "duplicate", "case_duplicate", "escape", "unknown", "bad_settings"]
)
def test_invalid_contract_rejected_before_outputs(dataset, tmp_path, settings, change):
    config, value = configuration(dataset, tmp_path, settings)
    if change == "no_methods":
        value["methods"] = []
    elif change == "duplicate":
        value["methods"].append(value["methods"][0])
    elif change == "case_duplicate":
        value["methods"].append(dict(value["methods"][0], id="KMERS"))
    elif change == "escape":
        value["methods"][0]["id"] = "../outside"
    elif change == "unknown":
        value["quality_filter"] = "HQ"
    else:
        value["methods"][0]["settings"]["classifier"]["params"]["invented"] = 1
    save(config, value)
    with pytest.raises(ValueError):
        run_experiment(config)
    assert not Path(value["output"]).exists()


def test_duplicate_raw_dataset_definitions_prepare_once(dataset, tmp_path, settings, monkeypatch):
    from bincomplete.experiments import runner

    config, value = configuration(dataset, tmp_path, settings, ("cosine",), raw=True)
    value["datasets"]["other"] = deepcopy(value["datasets"]["sample"])
    value["evaluation"] = "other"
    save(config, value)
    count = []
    original = runner.prepare_dataset
    monkeypatch.setattr(runner, "prepare_dataset", lambda *a, **k: (count.append(1), original(*a, **k))[1])
    result = run_experiment(config).manifest
    assert len(count) == 1
    assert result["datasets"]["sample"] == result["datasets"]["other"]


def test_configuration_change_and_concurrent_owner_are_rejected(dataset, tmp_path, settings):
    config, value = configuration(dataset, tmp_path, settings, ("cosine",))
    root = Path(value["output"])
    with FileLock(root.with_name(root.name + ".experiment.lock")):
        with pytest.raises(OSError):
            run_experiment(config)
    run_experiment(config)
    value["methods"][0]["settings"]["search"] = {"params": {"max_checks": 3}}
    save(config, value)
    with pytest.raises(ValueError, match="identity changed"):
        run_experiment(config)


def test_toml_settings_and_relative_paths(dataset, tmp_path):
    config = tmp_path / "recipe.toml"
    settings = tmp_path / "settings.json"
    save(settings, {"search": {"params": {"max_checks": 3}}})
    config.write_text(
        'schema_version = 1\noutput = "result"\ntraining = "sample"\nevaluation = "sample"\n'
        f"[datasets.sample]\nprepared = {json.dumps(str(dataset.path))}\n"
        '[[methods]]\nmethod = "cosine"\nsettings = "settings.json"\n',
        encoding="utf-8",
    )
    plan = load_plan(config)
    assert plan.document["output"] == str(tmp_path / "result")
    assert plan.document["methods"][0]["recipe"]["search"]["params"]["max_checks"] == 3
    assert str(settings) in plan.document["settings_files_sha256"]


def test_preparation_interruption_is_preserved_and_retried(dataset, tmp_path, settings, monkeypatch):
    from bincomplete.experiments import runner

    config, value = configuration(dataset, tmp_path, settings, ("cosine",), raw=True)
    original = runner.prepare_dataset

    def incomplete(destination, **kwargs):
        destination.mkdir(parents=True)
        (destination / "partial_index").write_text("interrupted")
        raise RuntimeError("simulated preparation failure")

    monkeypatch.setattr(runner, "prepare_dataset", incomplete)
    with pytest.raises(RuntimeError, match="preparation failure"):
        run_experiment(config)
    monkeypatch.setattr(runner, "prepare_dataset", original)
    summary = run_experiment(config).manifest
    root = Path(value["output"])
    assert summary["status"] == "SUCCEEDED"
    assert summary["datasets"]["sample"]["path"].endswith("0002")
    assert (root / "datasets/sample/0001/partial_index").read_text() == "interrupted"


def test_stop_policy_leaves_following_methods_pending(dataset, tmp_path, settings, monkeypatch):
    from bincomplete.experiments import runner

    config, value = configuration(dataset, tmp_path, settings)
    value["on_error"] = "stop"
    save(config, value)
    monkeypatch.setattr(runner, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fit failed")))
    summary = run_experiment(config).manifest
    assert summary["status"] == "FAILED"
    assert summary["methods"]["cosine"]["status"] == "PENDING"
    assert not (Path(value["output"]) / "methods/cosine").exists()


def test_comparison_counts_common_targets_and_assignment_conflicts(tmp_path):
    from bincomplete.artifacts import complete
    from bincomplete.engine import ProposalArtifact
    from bincomplete.experiments.comparison import compare_proposals
    from bincomplete.resources import ResourceLimits

    shared = {"recipient": "b0", "members": ["a", "x"], "added": ["x"]}
    definitions = {
        "left": [shared, {"recipient": "b1", "members": ["b", "x"], "added": ["x"]}],
        "right": [shared, {"recipient": "b2", "members": ["c", "z"], "added": ["z"]}],
    }
    artifacts = {}
    for name, rows in definitions.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "memberships.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        save(
            folder / "proposals.json",
            {
                "dataset_identity": "synthetic",
                "method": "cosine",
                "coverage": {row["recipient"]: {"scored": 2} for row in rows},
                "proposal_count": len(rows),
                "overlap": {},
            },
        )
        complete(folder, {"identity": name})
        artifacts[name] = ProposalArtifact(folder)
    report = compare_proposals(artifacts, ResourceLimits())
    assert report["distinct_construction_union"] == 3
    assert report["pairwise"][0]["common_recipients"] == 1
    assert report["pairwise"][0]["identical_constructions"] == 1
    assert report["pairwise"][0]["left_only_constructions_on_common_targets"] == 0
    assert report["pairwise"][0]["right_only_constructions_on_common_targets"] == 0
    assert report["donor_contigs_proposed_for_multiple_recipients"] == 1


def test_distinct_raw_contexts_share_assembly_index_and_resume(dataset, tmp_path, settings):
    from bincomplete import Dataset

    config, value = configuration(dataset, tmp_path, settings, ("kmers", "cosine"), raw=True)
    donors = tmp_path / "donors.txt"
    donors.write_text("\n".join(f"c{i}" for i in range(8, 13)) + "\n")
    value["datasets"]["evaluation"] = deepcopy(value["datasets"]["sample"])
    value["datasets"]["evaluation"]["donor_ids"] = str(donors)
    value["evaluation"] = "evaluation"
    save(config, value)
    summary = run_experiment(config).manifest
    first = Dataset(Path(summary["datasets"]["sample"]["path"]))
    second = Dataset(Path(summary["datasets"]["evaluation"]["path"]))
    assert first.path != second.path
    assert second.array_path("sequences.sqlite") == first.array_path("sequences.sqlite")
    assert second.array_path("coverage.npy") == first.array_path("coverage.npy")
    assert second.manifest["candidate_contigs"] == 5
    assert first.manifest["candidate_contigs"] == 6
    assert run_experiment(config).manifest == summary
