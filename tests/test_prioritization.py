"""Assessment ordering preserves candidates and separates priority from quality."""

import csv
from copy import deepcopy

import numpy as np
import pytest

from bincomplete import candidate_table, fit, propose
from bincomplete.artifacts import read, sha
from bincomplete.cli import main
from bincomplete.prioritization import proposal_priorities


def proposal(key, added, *, recipient="bin", method="method"):
    return {"proposal_id": key, "recipient": recipient, "method": method, "added": added}


def test_balancing_uses_saved_id_alignment_and_keeps_the_complete_pool():
    # Input donor and proposal order is deliberately unrelated to score rank.
    ids = ["c", "a", "d", "b"]
    scores = [0.6, 0.95, 0.2, 0.85]
    proposals = [
        proposal("single_b", ["b"]),
        proposal("triple", ["a", "b", "c"]),
        proposal("single_a", ["a"]),
        proposal("pair", ["c", "a"]),
        proposal("single_d", ["d"]),
    ]
    before = deepcopy(proposals)
    output = proposal_priorities(proposals, ids, scores, priority_limit=2)
    assert list(output) == ["single_a", "pair", "triple", "single_b", "single_d"]
    assert set(output) == {p["proposal_id"] for p in proposals} and proposals == before
    assert output["pair"]["mean_added_score"] == pytest.approx(0.775)
    assert output["pair"]["min_added_score"] == 0.6
    assert output["single_b"]["within_size_rank"] == 2
    assert [p["assessment_tier"] for p in output.values()] == [1, 1, 2, 2, 2]
    assert [p["priority_rank"] for p in output.values()] == list(range(1, 6))


def test_ties_and_negative_similarity_are_deterministic():
    rows = [proposal("z", ["b"]), proposal("a", ["a"]), proposal("both", ["a", "b"])]
    ranked = proposal_priorities(rows, ["a", "b"], [-0.2, -0.2])
    assert list(ranked) == ["a", "both", "z"]
    assert ranked == proposal_priorities(rows[::-1], ["b", "a"], [-0.2, -0.2])


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_priority_limit_is_only_a_positive_tier_boundary(limit):
    with pytest.raises(ValueError, match="positive integer"):
        proposal_priorities([], [], [], priority_limit=limit)


@pytest.mark.parametrize(
    "ids,scores", [(["a"], [float("nan")]), (["a", "a"], [1, 2]), (["a"], []), ([["a"]], [1])]
)
def test_invalid_score_alignment_is_rejected(ids, scores):
    with pytest.raises(ValueError):
        proposal_priorities([], ids, scores)


def test_missing_scores_and_mixed_groups_are_rejected():
    with pytest.raises(ValueError, match="no saved score"):
        proposal_priorities([proposal("x", ["missing"])], ["a"], [0.4])
    with pytest.raises(ValueError, match="one recipient"):
        proposal_priorities([proposal("x", ["a"]), proposal("y", ["a"], recipient="other")], ["a"], [0.4])


def test_candidate_inventory_adds_priority_without_editing_proposals(dataset, tmp_path, limits):
    model = fit(dataset, tmp_path / "model", method="cosine", limits=limits, cache=tmp_path / "cache")
    proposals = propose(model, dataset, tmp_path / "props", limits=limits, cache=tmp_path / "cache")
    hashes = {
        name: sha(proposals.path / name) for name in read(proposals.path / "complete.json")["outputs_sha256"]
    }
    output = candidate_table([proposals], tmp_path / "candidates.tsv", limits=limits, priority_limit=1)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    original = [
        __import__("json").loads(line)
        for line in (proposals.path / "memberships.jsonl").read_text().splitlines()
    ]
    assert {r["proposal_id"] for r in rows} == {p["proposal_id"] for p in original}
    assert len(rows) == len(original) and any(r["assessment_tier"] == "2" for r in rows)
    assert all(
        r["quality_status"] == "UNASSESSED" and r["priority_strategy"] == "size_balanced_mean" for r in rows
    )
    keys = [
        (int(r["assessment_tier"]), int(r["priority_rank"]), r["recipient"], r["method"], r["proposal_id"])
        for r in rows
    ]
    assert keys == sorted(keys)
    receipt = read(output.with_name(output.name + ".json"))
    assert receipt["all_proposals_retained"] and receipt["rows"] == len(original)
    assert all(sha(proposals.path / name) == value for name, value in hashes.items())
    assert (
        main(
            [
                "candidates",
                "--proposals",
                str(proposals.path),
                "--output",
                str(tmp_path / "cli.tsv"),
                "--priority-limit",
                "1",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="other artifacts"):
        candidate_table([proposals], output, priority_limit=2)
    assert np.isfinite([float(r["mean_added_score"]) for r in rows]).all()


def test_successful_experiment_exports_the_full_ordered_inventory(dataset, tmp_path):
    import json

    from bincomplete import run_experiment
    from bincomplete.artifacts import save

    config = tmp_path / "experiment.json"
    save(
        config,
        {
            "schema_version": 1,
            "output": str(tmp_path / "results"),
            "training": "sample",
            "evaluation": "sample",
            "datasets": {"sample": {"prepared": str(dataset.path)}},
            "methods": [{"method": "cosine"}],
            "resources": {"threads": 1, "memory_gib": 0.5, "cache_gib": 0.01, "chunk_size": 3},
        },
    )
    artifact = run_experiment(config)
    summary = artifact.manifest
    assert summary["assessment_priority"]["all_proposals_retained"]
    output = artifact.path / "candidates.tsv"
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    proposal_root = artifact.path / "methods/cosine/propose/0001"
    original = [json.loads(line) for line in (proposal_root / "memberships.jsonl").read_text().splitlines()]
    assert {r["proposal_id"] for r in rows} == {p["proposal_id"] for p in original}
    marker = read(artifact.path / "complete.json")
    assert marker["outputs_sha256"]["candidates.tsv"] == sha(output)
    before = sha(output)
    assert run_experiment(config).manifest["assessment_priority"] == summary["assessment_priority"]
    assert sha(output) == before
