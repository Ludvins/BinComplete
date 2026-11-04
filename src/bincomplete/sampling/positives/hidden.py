import numpy as np

from ...core import register


@register("positive_sampler", "hidden_members")
class HiddenMembers:
    def __init__(self):
        pass

    def sample(self, episode, context, rng):
        return np.array([], int) if episode.absent else episode.hidden.copy()
