"""Score-based assessment priority for existing proposals, without filtering."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

PRIORITY_STRATEGY = "size_balanced_mean"
PRIORITY_VERSION = 1
DEFAULT_PRIORITY_LIMIT = 40


def proposal_priorities(
    proposals: list[dict[str, Any]],
    donor_ids: Any,
    scores: Any,
    *,
    priority_limit: int = DEFAULT_PRIORITY_LIMIT,
) -> dict[str, dict[str, Any]]:
    """Score and order one artifact/recipient group without removing proposals.

    Parameters
    ----------
    proposals : list of dict
        Existing proposals for one recipient in one method artifact. Every
        proposal must add at least one distinct scored contig.
    donor_ids, scores : array-like
        Aligned donor IDs and saved membership scores. These arrays need not
        already be sorted. Raw scores are compared within this group only.
    priority_limit : int, default=40
        Number of proposals assigned to the first assessment tier. All remaining
        proposals receive tier 2 and remain in the returned mapping.

    Returns
    -------
    dict
        Proposal IDs mapped to mean/minimum added-contig score, addition count,
        within-size rank, overall priority rank, strategy and assessment tier.
        Neither scores nor ranks establish biological quality.
    """
    if type(priority_limit) is not int or priority_limit < 1:
        raise ValueError("priority_limit must be a positive integer; it never filters proposals")
    names, values = np.asarray(donor_ids), np.asarray(scores)
    if names.ndim != 1 or values.ndim != 1 or len(names) != len(values):
        raise ValueError("Donor IDs and scores must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite membership score")
    lookup = {str(name): float(value) for name, value in zip(names, values)}
    if len(lookup) != len(names):
        raise ValueError("Duplicate scored donor ID")
    if len({(p["recipient"], p["method"]) for p in proposals}) > 1:
        raise ValueError("Priority ranks require one recipient and method artifact")
    groups: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen: set[str] = set()
    for proposal in proposals:
        key = proposal["proposal_id"]
        added = proposal["added"]
        if key in seen or not added or len(added) != len(set(added)):
            raise ValueError("Duplicate proposal ID or empty/duplicate added contigs")
        seen.add(key)
        try:
            donor_scores = [lookup[name] for name in added]
        except KeyError as error:
            raise ValueError(f"Added contig has no saved score: {error.args[0]}") from error
        count = len(added)
        groups[count].append(
            (
                key,
                {
                    "added_contig_count": count,
                    "mean_added_score": math.fsum(donor_scores) / count,
                    "min_added_score": min(donor_scores),
                    "priority_strategy": PRIORITY_STRATEGY,
                },
            )
        )
    for group in groups.values():
        group.sort(key=lambda item: (-item[1]["mean_added_score"], item[0]))
        for rank, (_, record) in enumerate(group, 1):
            record["within_size_rank"] = rank
    output: dict[str, dict[str, Any]] = {}
    for position in range(max(map(len, groups.values()), default=0)):
        for count in sorted(groups):
            group = groups[count]
            if position < len(group):
                key, record = group[position]
                rank = len(output) + 1
                output[key] = record | {
                    "priority_rank": rank,
                    "assessment_tier": 1 if rank <= priority_limit else 2,
                }
    return output
