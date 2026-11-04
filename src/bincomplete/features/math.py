import numpy as np

METRICS = {"cosine", "cosine_similarity", "euclidean", "hellinger", "absolute"}


def distance(a, b, metric):
    if metric == "euclidean":
        return np.linalg.norm(a - b, axis=-1)
    if metric in ["cosine", "cosine_similarity"]:
        similarity = (a * b).sum(-1) / np.maximum(
            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12
        )
        return similarity if metric == "cosine_similarity" else 1 - similarity
    if metric == "hellinger":
        if (a < 0).any() or (b < 0).any():
            raise ValueError("Hellinger requires a nonnegative representation")
        return np.sqrt(np.maximum(((np.sqrt(a) - np.sqrt(b)) ** 2).sum(-1) / 2, 0))
    if metric == "absolute":
        if a.shape[-1] != 1 or b.shape[-1] != 1:
            raise ValueError("Absolute distance requires a scalar representation")
        return np.abs(a - b)[..., 0]
    raise ValueError(f"Unknown metric {metric}")


def summary(values, lengths, kind="centroid", weights="length"):
    if not len(values):
        raise ValueError("Empty core")
    if kind == "median":
        if weights != "uniform":
            raise ValueError("Median currently supports uniform weights only")
        return np.median(values, axis=0)
    if kind != "centroid" or weights not in ["uniform", "length"]:
        raise ValueError("Unknown summary or weighting")
    w = np.asarray(lengths, float) if weights == "length" else np.ones(len(values))
    return np.average(values, axis=0, weights=w)
