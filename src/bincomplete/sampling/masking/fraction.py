import numpy as np

from ...core import register
from ...masking import Mask


@register("masker", "leave_one_out_and_fraction")
class FractionMasker:
    def __init__(self, extra_masks=20, fractions=(0.05, 0.1, 0.15, 0.2), decoy_every=5, deduplicate=True):
        if (
            type(extra_masks) is not int
            or extra_masks < 0
            or not fractions
            or any(not 0 < f < 1 for f in fractions)
        ):
            raise ValueError("Invalid mask fractions/count")
        if type(decoy_every) is not int or decoy_every < 0 or not isinstance(deduplicate, bool):
            raise ValueError("Invalid decoy/deduplication settings")
        self.extra_masks, self.fractions, self.decoy_every = extra_masks, tuple(fractions), decoy_every
        self.deduplicate = deduplicate

    def generate(self, members, lengths, rng):
        members = np.asarray(members, int)
        if len(members) < 2:
            return
        seen = set()
        for hidden in members:
            mask = Mask(members[members != hidden], np.array([hidden]))
            seen.add((tuple(sorted(mask.core)), (int(hidden),), False))
            yield mask
        total = lengths[members].sum()
        for episode in range(self.extra_masks):
            shuffled = rng.permutation(members)
            count = (
                int(
                    np.argmin(
                        np.abs(
                            np.cumsum(lengths[shuffled[:-1]]) / total
                            - self.fractions[episode % len(self.fractions)]
                        )
                    )
                )
                + 1
            )
            hidden, core = np.sort(shuffled[:count]), np.sort(shuffled[count:])
            absent = bool(self.decoy_every and (episode + 1) % self.decoy_every == 0)
            key = tuple(core), tuple(hidden), absent
            if self.deduplicate and key in seen:
                continue
            seen.add(key)
            yield Mask(core, hidden, absent)


@register("masker", "leave_one_out")
class LeaveOneOut(FractionMasker):
    def __init__(self):
        super().__init__(extra_masks=0)
