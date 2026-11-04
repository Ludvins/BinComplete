"""Uniform addition-only search with singleton and prefix-boundary alternatives.

No recipient, contig, quality score or reference-witness input is accepted.
"""

from itertools import product

import numpy as np

from ..core import register
from .limits import proposal_limit


@register("search", "expanded_ranked_alternatives")
class ExpandedRankedAlternatives:
    donor_ownership = "independent"
    needs_pools = False

    def __init__(
        self,
        max_checks=None,
        singleton_depth=50,
        boundary_width=2,
        max_added_fraction=1.0,
        *,
        max_proposals=None,
    ):
        max_checks = proposal_limit(max_checks, max_proposals, default=128)
        if type(max_checks) is not int or max_checks < 20 or max_checks > 4096:
            raise ValueError("max_checks must be an integer in 20..4096")
        if type(singleton_depth) is not int or singleton_depth < 5:
            raise ValueError("singleton_depth must be an integer >=5")
        if type(boundary_width) is not int or not 0 <= boundary_width <= 16:
            raise ValueError("boundary_width must be an integer in 0..16")
        if not 0 < max_added_fraction <= 1:
            raise ValueError("Invalid length protection")
        self.max_checks = max_checks
        self.max_proposals = max_checks
        self.singleton_depth = singleton_depth
        self.fallback_singletons = singleton_depth
        self.boundary_width = boundary_width
        self.max_added_fraction = max_added_fraction

    def propose(self, core, ranked, store):
        from bincomplete.search.ranked_alternatives import RankedAlternatives

        core, ranked = np.asarray(core, int), np.asarray(ranked, int)
        if len(core) != len(set(core)) or len(ranked) != len(set(ranked)):
            raise ValueError("Duplicate core or donor indices")
        protected, core_set = store.protected(), set(core)
        if any(i in core_set or str(store.ids[i]) in protected for i in ranked):
            raise ValueError("Protected donor in ranking")
        original_bp = int(store.lengths[core].sum())
        seen, emitted = set(), 0
        inherited = RankedAlternatives(
            max_checks=20,
            max_added_fraction=self.max_added_fraction,
            fallback_singletons=self.singleton_depth,
        )
        for proposal in inherited.propose(core, ranked, store):
            members = tuple(sorted(proposal["indices"]))
            assert members not in seen
            seen.add(members)
            emitted += 1
            yield proposal
        templates = [(f"expanded_singleton_{i + 1}", (i,)) for i in range(self.singleton_depth)]
        # Local rank uncertainty around the two inherited multi-donor prefixes.
        for size in (10, 20):
            for incoming, removed in product(range(size, size + self.boundary_width), reversed(range(size))):
                positions = tuple(i for i in range(size) if i != removed) + (incoming,)
                templates.append((f"prefix_{size}_drop_{removed + 1}_add_{incoming + 1}", positions))
        for name, positions in templates:
            if emitted >= self.max_checks:
                return
            if max(positions) >= len(ranked):
                continue
            added = list(map(int, ranked[list(positions)]))
            members = tuple(sorted(core_set | set(added)))
            added_bp = int(store.lengths[added].sum())
            if members in seen or added_bp > self.max_added_fraction * original_bp:
                continue
            seen.add(members)
            emitted += 1
            yield {
                "indices": list(members),
                "added": added,
                "removed": [],
                "step": emitted,
                "recipe": name,
                "donor_ranks": [i + 1 for i in positions],
                "requested": len(positions),
                "added_bp": added_bp,
                "max_added_fraction": self.max_added_fraction,
            }
