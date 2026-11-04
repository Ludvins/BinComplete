"""A source-local mask family with an explicit cap on hidden contig count."""

import numpy as np

from ...core import register
from ...masking import Mask


@register("masker", "leave_one_out_and_count")
class CountMasker:
    def __init__(self, max_hidden=4, extra_masks=20, decoy_every=0):
        if any(type(x) is not int or x < 0 for x in [max_hidden, extra_masks, decoy_every]) or max_hidden < 1:
            raise ValueError("Invalid count mask settings")
        self.max_hidden, self.extra_masks, self.decoy_every = max_hidden, extra_masks, decoy_every

    def generate(self, members, lengths, rng):
        members = np.asarray(members, int)
        if len(members) < 2:
            return
        for c in members:
            yield Mask(members[members != c], np.array([c]))
        for episode in range(self.extra_masks):
            count = 1 + episode % min(self.max_hidden, len(members) - 1)
            hidden = np.sort(rng.choice(members, count, replace=False))
            absent = bool(self.decoy_every and (episode + 1) % self.decoy_every == 0)
            yield Mask(members[~np.isin(members, hidden)], hidden, absent)
