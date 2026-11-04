"""Tabular proposal inventories for external review, without quality verdicts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import fingerprint, read, save, sha
from .data import Dataset
from .engine import ProposalArtifact
from .prioritization import DEFAULT_PRIORITY_LIMIT, PRIORITY_STRATEGY, PRIORITY_VERSION, proposal_priorities
from .resources import FileLock, ResourceLimits


def candidate_table(
    artifacts: list[ProposalArtifact],
    output: str | Path,
    *,
    limits: ResourceLimits | None = None,
    priority_limit: int = DEFAULT_PRIORITY_LIMIT,
) -> Path:
    """Write one TSV row per saved proposal with exact membership/sequence keys.

    All proposals remain UNASSESSED. This operation neither materializes FASTAs
    nor runs quality tools. Identical constructions across methods share keys,
    while proposal IDs retain their own provenance and external approval identity.
    Scores and addition-size-balanced priority ranks are included by default.
    ``priority_limit`` marks the first assessment tier; it never removes rows.
    """
    if not artifacts:
        raise ValueError("Supply at least one proposal artifact")
    if type(priority_limit) is not int or priority_limit < 1:
        raise ValueError("priority_limit must be a positive integer; all proposals are retained")
    limits = limits or ResourceLimits()
    manifests = [(artifact, artifact.manifest) for artifact in artifacts]
    if len({str(artifact.path.resolve()) for artifact in artifacts}) != len(artifacts):
        raise ValueError("Duplicate proposal artifact")
    priority = {
        "strategy": PRIORITY_STRATEGY,
        "version": PRIORITY_VERSION,
        "first_tier_limit": priority_limit,
    }
    identity = fingerprint(
        {"inventory_version": 2, "artifacts": [m for _, m in manifests], "priority": priority}
    )
    target = Path(output).resolve()
    proof = target.with_name(target.name + ".json")
    temporary = target.with_name(target.name + ".partial")
    with FileLock(target.with_name(target.name + ".lock")):
        if target.exists() or proof.exists():
            if not (target.exists() and proof.exists()):
                raise ValueError("Incomplete candidate table exists; preserve it and choose a new output")
            receipt = read(proof)
            if receipt["identity"] != identity or sha(target) != receipt["sha256"]:
                raise ValueError("Candidate table belongs to other artifacts or has changed")
            return target
        if temporary.exists():
            raise ValueError("Preserved partial candidate table exists; choose a new output")
        populations: dict[str, dict[str, str]] = {}
        population_ids: dict[str, np.ndarray] = {}
        total_rows = 0
        ordered_rows = []
        reserved_bytes = 0
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                [
                    "proposal_id",
                    "method",
                    "recipient",
                    "construction_id",
                    "sequence_set_id",
                    "member_count",
                    "added_bp",
                    "members_json",
                    "artifact",
                    "quality_status",
                    "added_contig_count",
                    "mean_added_score",
                    "min_added_score",
                    "within_size_rank",
                    "priority_rank",
                    "priority_strategy",
                    "assessment_tier",
                ]
            )
            for artifact, manifest in manifests:
                name = manifest["dataset"]
                if name not in populations:
                    dataset = Dataset(Path(name))
                    dataset.verify()
                    ids = np.load(dataset.array_path("ids.npy"), mmap_mode="r", allow_pickle=False)
                    hashes = np.load(
                        dataset.array_path("sequence_hashes.npy"), mmap_mode="r", allow_pickle=False
                    )
                    try:
                        count = sum(len(p) for p in populations.values()) + len(ids)
                        limits.require(count * 768, "Candidate inventory sequence index")
                        populations[name] = {
                            str(key): bytes(value).decode("ascii") for key, value in zip(ids, hashes)
                        }
                        population_ids[name] = np.array(ids, copy=True)
                    finally:
                        ids._mmap.close()
                        hashes._mmap.close()
                grouped = defaultdict(list)
                with (artifact.path / "memberships.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line)
                        grouped[row["recipient"]].append(row)
                for recipient, proposals in sorted(grouped.items()):
                    ranking = artifact.path / manifest["coverage"][recipient]["ranking"]
                    donor_rows = np.load(ranking / "rows.npy", mmap_mode="r", allow_pickle=False)
                    scores = np.load(ranking / "scores.npy", mmap_mode="r", allow_pickle=False)
                    try:
                        ids = population_ids[name]
                        if (
                            donor_rows.ndim != 1
                            or not np.issubdtype(donor_rows.dtype, np.integer)
                            or len(set(map(int, donor_rows))) != len(donor_rows)
                            or (
                                len(donor_rows)
                                and (int(donor_rows.min()) < 0 or int(donor_rows.max()) >= len(ids))
                            )
                        ):
                            raise ValueError("Invalid saved donor row indices")
                        limits.require(
                            reserved_bytes + len(ids) * 768 + len(donor_rows) * 512, "Priority score index"
                        )
                        ranked = proposal_priorities(
                            proposals, ids[donor_rows], scores, priority_limit=priority_limit
                        )
                    finally:
                        donor_rows._mmap.close()
                        scores._mmap.close()
                    for row in proposals:
                        members = sorted(row["members"])
                        construction = fingerprint({"recipient": row["recipient"], "members": members})
                        sequences = fingerprint({key: populations[name][key] for key in members})
                        fields = [
                            row["proposal_id"],
                            manifest["method"],
                            row["recipient"],
                            construction,
                            sequences,
                            len(members),
                            row.get("added_bp", ""),
                            json.dumps(members),
                            str(artifact.path.resolve()),
                            "UNASSESSED",
                        ]
                        record = ranked[row["proposal_id"]]
                        fields.extend(
                            record[key]
                            for key in (
                                "added_contig_count",
                                "mean_added_score",
                                "min_added_score",
                                "within_size_rank",
                                "priority_rank",
                                "priority_strategy",
                                "assessment_tier",
                            )
                        )
                        reserved_bytes += 2048 + len(fields[7]) * 4
                        limits.require(
                            reserved_bytes + sum(len(p) for p in populations.values()) * 768,
                            "Ordered candidate inventory",
                        )
                        ordered_rows.append(
                            (
                                (
                                    record["assessment_tier"],
                                    record["priority_rank"],
                                    recipient,
                                    manifest["method"],
                                    row["proposal_id"],
                                ),
                                fields,
                            )
                        )
                        total_rows += 1
            for _, fields in sorted(ordered_rows):
                writer.writerow(fields)
        temporary.replace(target)
        save(
            proof,
            {
                "schema_version": 2,
                "identity": identity,
                "sha256": sha(target),
                "rows": total_rows,
                "quality_status": "UNASSESSED_EXTERNAL_TOOLS_REQUIRED",
                "priority": priority,
                "rank_scope": "One method artifact and recipient; raw scores are not comparable across groups",
                "row_order": ["assessment_tier", "priority_rank", "recipient", "method", "proposal_id"],
                "all_proposals_retained": True,
                "deferred_proposals": sum(fields[-1] == 2 for _, fields in ordered_rows),
            },
        )
    return target
