"""Bounded structural comparisons of completed membership proposals."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint
from ..engine import ProposalArtifact
from ..resources import ResourceLimits


def compare_proposals(artifacts: dict[str, ProposalArtifact], limits: ResourceLimits) -> dict[str, Any]:
    """Compare target coverage and identical constructions without quality claims.

    Parameters
    ----------
    artifacts : dict of str to ProposalArtifact
        Completed method artifacts for one prepared evaluation population.
    limits : ResourceLimits
        Upper bound for the in-memory membership-identity index.

    Returns
    -------
    dict
        Method coverage, pairwise membership overlap and prospective donor
        conflicts. Counts describe proposals, not HQ gains or compatible repairs.
    """
    by_method, by_recipient, coverage, details = {}, {}, {}, {}
    donor_owners: dict[str, set[str]] = {}
    used = 0
    population = None
    ordered = dict(sorted(artifacts.items()))
    for identifier, artifact in ordered.items():
        manifest = artifact.manifest
        if population is not None and population != manifest["dataset_identity"]:
            raise ValueError("Experiment comparison requires a common evaluation population")
        population = manifest["dataset_identity"]
        keys: set[str] = set()
        recipients: dict[str, set[str]] = {}
        coverage[identifier] = set(manifest["coverage"])
        with (artifact.path / "memberships.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                recipient = row["recipient"]
                members = row["members"]
                limits.require(used + len(members) * 256 + len(line) * 4, "Experiment comparison index")
                key = fingerprint({"recipient": recipient, "members": sorted(members)})
                if key not in keys:
                    used += 1024 + len(recipient) * 4
                    keys.add(key)
                    recipients.setdefault(recipient, set()).add(key)
                for donor in row["added"]:
                    if donor not in donor_owners:
                        used += 512 + len(donor) * 4
                        donor_owners[donor] = set()
                    if recipient not in donor_owners[donor]:
                        used += 256 + len(recipient) * 4
                        donor_owners[donor].add(recipient)
                limits.require(used, "Experiment comparison index")
        by_method[identifier], by_recipient[identifier] = keys, recipients
        details[identifier] = {
            "method": manifest["method"],
            "evaluated_recipients": len(coverage[identifier]),
            "scored_candidates": sum(row["scored"] for row in manifest["coverage"].values()),
            "proposals": manifest["proposal_count"],
            "distinct_constructions": len(keys),
            "overlap": manifest["overlap"],
            "artifact": str(artifact.path),
        }
    common = set.intersection(*coverage.values()) if coverage else set()
    pairs = []
    for left, right in combinations(ordered, 2):
        targets = coverage[left] & coverage[right]
        limits.require(used * 2, "Experiment pairwise comparison buffers")
        first = {key for name in targets for key in by_recipient[left].get(name, set())}
        second = {key for name in targets for key in by_recipient[right].get(name, set())}
        shared_recipients = sum(
            bool(by_recipient[left].get(name, set()) & by_recipient[right].get(name, set()))
            for name in targets
        )
        pairs.append(
            {
                "methods": [left, right],
                "common_recipients": len(targets),
                "identical_constructions": len(first & second),
                "recipients_with_identical_constructions": shared_recipients,
                "left_only_constructions_on_common_targets": len(first - second),
                "right_only_constructions_on_common_targets": len(second - first),
            }
        )
    owners: dict[str, list[str]] = {}
    limits.require(used * 2, "Experiment overlap ownership index")
    for identifier, keys in by_method.items():
        for key in keys:
            owners.setdefault(key, []).append(identifier)
    intersections = Counter(tuple(sorted(names)) for names in owners.values())
    for identifier in details:
        details[identifier]["constructions_unique_to_method"] = intersections[(identifier,)]
    return {
        "schema_version": 1,
        "methods": details,
        "common_recipients_all_methods": len(common),
        "distinct_recipient_union": len(set().union(*coverage.values())) if coverage else 0,
        "distinct_construction_union": len(owners),
        "pairwise": pairs,
        "exact_method_intersections": [
            {"methods": list(names), "constructions": count}
            for names, count in sorted(intersections.items())
            if count
        ],
        "donor_contigs_proposed_for_multiple_recipients": sum(
            len(names) > 1 for names in donor_owners.values()
        ),
        "quality_status": "NOT_EVALUATED_EXTERNAL_TOOLS_REQUIRED",
        "compatibility_status": "No compatible gain selection inferred from raw proposals",
        "comparison_scope": "Identical recipient and full membership; target overlap alone does not imply equal budgets or donor rules",
    }


def write_report(path: Path, summary: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Write a compact human-readable experiment report without HQ assertions."""
    lines = [
        "# Recruitment experiment",
        "",
        f"Status: {summary['status']}",
        "",
        "Methods ran sequentially with shared inputs and a bounded derived cache.",
        "Quality assessment remains external. Counts below describe candidate constructions.",
        "",
        "| Method ID | Method | Recipients | Candidates scored | Constructions | Unique constructions |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for identifier, row in comparison["methods"].items():
        lines.append(
            f"| {identifier} | {row['method']} | {row['evaluated_recipients']} | {row['scored_candidates']} | {row['distinct_constructions']} | {row['constructions_unique_to_method']} |"
        )
    lines.extend(
        [
            "",
            f"Distinct construction union: {comparison['distinct_construction_union']}.",
            f"Common recipients across completed methods: {comparison['common_recipients_all_methods']}.",
            "",
        ]
    )
    if summary.get("assessment_priority"):
        lines.extend(
            [
                "Assessment order: [candidates.tsv](candidates.tsv), with mean/minimum added-contig scores and addition-size-balanced ranks.",
                "Every proposal is retained. Tier 1 contains the first 40 positions per method artifact/recipient; tier 2 contains all remaining positions.",
                "This is an evaluation queue, not an HQ prediction. Quality assessment remains external.",
                "",
            ]
        )
    for identifier, row in summary["methods"].items():
        if row["status"] != "SUCCEEDED":
            lines.append(f"- {identifier}: {row['status']}. {row.get('error', '')}")
    lines.extend(
        [
            "",
            "## Next steps",
            "",
            "These proposals have not been assessed for HQ. Inspect memberships.jsonl to choose candidate IDs.",
            "Export candidates, assess them with external quality tools, then record approved proposal IDs.",
            "Approval must use quality evidence for the exact full membership and sequences.",
            "",
        ]
    )
    for identifier, row in summary["methods"].items():
        if row["status"] == "SUCCEEDED":
            lines.extend(
                [
                    f"Proposal artifact for {identifier}: `{row['proposals']}`",
                    "",
                    "```shell",
                    f'bincomplete export --proposals "{row["proposals"]}" --proposal-ids candidates.txt --output candidates_{identifier}',
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "Reconcile approved IDs across the desired artifacts by repeating --proposals:",
            "",
            "```shell",
            "bincomplete reconcile --proposals PATH --approved-ids approved.txt --output selected",
            "bincomplete export --selection selected --complete-directory --output refined_bins",
            "```",
            "",
            "See the evaluation guide for external assessment and multiple-method examples.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
