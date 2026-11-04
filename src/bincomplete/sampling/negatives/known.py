import numpy as np

from ...core import register


@register("negative_sampler", "hard_and_random")
class HardAndRandom:
    def __init__(self, hard=64, random=32):
        if any(type(n) is not int or n < 0 for n in [hard, random]) or hard + random == 0:
            raise ValueError("Negative counts must be nonnegative and not both zero")
        self.hard, self.random = hard, random

    def sample(self, episode, context, rng):
        hard = context.retrieved[np.isin(context.retrieved, context.negatives)][: self.hard]
        pool = context.eligible_negatives[~np.isin(context.eligible_negatives, hard)]
        random = rng.choice(pool, min(self.random, len(pool)), replace=False)
        return np.unique(np.r_[hard, random])


@register("negative_sampler", "embedding_hard_random")
class EmbeddingHardRandom:
    needs_retrieval = False
    needs_eligibility = False

    def __init__(self, hard=64, random=32):
        if any(type(x) is not int or x < 0 for x in [hard, random]) or hard + random < 1:
            raise ValueError("Invalid negative counts")
        self.hard, self.random = hard, random

    def sample(self, episode, context, rng):
        spec = context.context.recipe["negative_sampler"]
        name = spec["representations"]["embedding"]
        z = context.context.representation(name)
        centre = np.mean(z[episode.core], axis=0)
        centre /= max(np.linalg.norm(centre), 1e-12)
        pool = context.negatives
        hard, best_scores = np.array([], int), np.array([], np.float32)
        width = context.context.recipe["resources"]["chunk_size"]
        for start in range(0, len(pool), width):
            block = pool[start : start + width]
            scores = np.asarray(z[block], np.float32) @ centre
            joined, costs = np.r_[hard, block], np.r_[best_scores, scores]
            order = np.lexsort((context.store.ids[joined], -costs))[: self.hard]
            hard, best_scores = joined[order], costs[order]
        other = pool[~np.isin(pool, hard)]
        random = rng.choice(other, min(self.random, len(other)), replace=False)
        return np.unique(np.r_[hard, random])
