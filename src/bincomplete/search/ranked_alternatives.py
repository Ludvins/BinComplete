"""Finite addition-only memberships: singleton trials before combinations.

There is no classifier-score acceptance threshold. Scores only order donors.
The proposal cap counts distinct memberships; quality evaluation stays external.
"""

import numpy as np

from ..core import register
from .limits import proposal_limit

TEMPLATES = (
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1, 2),
    (0, 1, 2, 3, 4),
    (0, 3),
    (0, 4),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (1, 2, 3),
    tuple(range(10)),
    tuple(range(20)),
)


@register("search", "ranked_alternatives")
class RankedAlternatives:
    donor_ownership = "independent"
    needs_pools = False

    def __init__(
        self, max_checks=None, max_added_fraction=1.0, fallback_singletons=50, *, max_proposals=None
    ):
        max_checks = proposal_limit(max_checks, max_proposals, default=20)
        if type(max_checks) is not int or not 1 <= max_checks <= 20:
            raise ValueError("max_checks must be an integer in 1..20")
        if not 0 < max_added_fraction <= 1 or type(fallback_singletons) is not int or fallback_singletons < 5:
            raise ValueError("Invalid length limit or fallback singleton range")
        self.max_checks = max_checks
        self.max_proposals = max_checks
        self.max_added_fraction = max_added_fraction
        self.fallback_singletons = fallback_singletons

    def propose(self, core, ranked, store):
        core = np.asarray(core, int)
        ranked = np.asarray(ranked, int)
        if len(core) != len(set(core)) or len(ranked) != len(set(ranked)):
            raise ValueError("Duplicate core or candidate indices")
        protected = store.protected()
        core_set = set(core)
        if any(c in core_set or str(store.ids[c]) in protected for c in ranked):
            raise ValueError("Protected donor in ranking")
        original_bp = int(store.lengths[core].sum())
        templates = [(f"template_{i + 1}", ranks) for i, ranks in enumerate(TEMPLATES)]
        templates += [(f"fallback_rank_{i + 1}", (i,)) for i in range(5, self.fallback_singletons)]
        seen = set()
        emitted = 0
        for name, ranks in templates:
            # An incomplete prefix must not silently become a smaller recipe.
            if max(ranks) >= len(ranked):
                continue
            added = ranked[list(ranks)].tolist()
            added_bp = int(store.lengths[added].sum())
            members = tuple(sorted(core_set | set(added)))
            if members in seen or added_bp > self.max_added_fraction * original_bp:
                continue
            seen.add(members)
            emitted += 1
            yield dict(
                indices=list(members),
                added=added,
                removed=[],
                step=emitted,
                recipe=name,
                donor_ranks=[r + 1 for r in ranks],
                requested=len(ranks),
                added_bp=added_bp,
                max_added_fraction=self.max_added_fraction,
            )
            if emitted == self.max_checks:
                break
