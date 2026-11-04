import itertools
import json

import pytest

from bincomplete import ProposalArtifact, reconcile
from bincomplete.artifacts import complete, save


def artifact(tmp_path, rows):
    root = tmp_path / "proposals"
    root.mkdir()
    (root / "memberships.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    save(root / "proposals.json", {"dataset": "external_dataset", "dataset_identity": "population"})
    complete(root, {"identity": "test", "kind": "proposals"})
    return ProposalArtifact(root)


def test_maximum_beats_greedy_and_has_full_membership_witness(tmp_path):
    rows = [
        {"proposal_id": "a", "recipient": "A", "members": ["coreA", "shared"]},
        {"proposal_id": "b", "recipient": "B", "members": ["coreB", "shared"]},
        {"proposal_id": "c", "recipient": "A", "members": ["coreA", "other"]},
    ]
    source = artifact(tmp_path, rows)
    result = reconcile([source], ["a", "b", "c"], tmp_path / "selection").manifest
    assert result["approved_recipient_union"] == 2
    assert result["compatible_recipients"] == 2
    assert [row["proposal_id"] for row in result["selected"]] == ["b", "c"]
    assert result["optimal"] and result["canonical_tie_break"]


def test_union_is_distinct_from_compatible_and_canonical(tmp_path):
    rows = [
        {"proposal_id": "a", "recipient": "A", "members": ["a", "shared"]},
        {"proposal_id": "b", "recipient": "B", "members": ["b", "shared"]},
    ]
    source = artifact(tmp_path, rows)
    result = reconcile([source], ["b", "a"], tmp_path / "selection").manifest
    assert result["approved_recipient_union"] == 2
    assert result["compatible_recipients"] == 1
    assert result["selected"][0]["proposal_id"] == "a"
    with pytest.raises(ValueError, match="Approved IDs"):
        reconcile([source], ["unknown"], tmp_path / "bad")


def test_solver_matches_exhaustive_small_cases(tmp_path):
    rows = [
        {"proposal_id": str(i), "recipient": str(i // 2), "members": [f"core{i // 2}", f"donor{i % 3}"]}
        for i in range(8)
    ]
    source = artifact(tmp_path, rows)
    feasible = []
    for included in itertools.product((True, False), repeat=len(rows)):
        chosen = [row for row, use in zip(rows, included) if use]
        members = [member for row in chosen for member in row["members"]]
        if len(members) == len(set(members)):
            feasible.append((sum(included), included))
    best_count, best_bits = max(feasible)
    result = reconcile([source], [row["proposal_id"] for row in rows], tmp_path / "result").manifest
    assert result["compatible_recipients"] == best_count
    assert [row["proposal_id"] for row in result["selected"]] == [
        row["proposal_id"] for row, use in zip(rows, best_bits) if use
    ]


def test_protected_and_zero_approval(tmp_path):
    source = artifact(tmp_path, [{"proposal_id": "a", "recipient": "A", "members": ["core"]}])
    assert (
        reconcile([source], ["a"], tmp_path / "protected", protected_bins=["A"]).manifest[
            "compatible_recipients"
        ]
        == 0
    )
    assert reconcile([source], [], tmp_path / "none").manifest["optimal"]
