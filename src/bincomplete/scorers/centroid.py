"""Analytic query predictor using the same contract as learned predictors."""

import numpy as np

from ..core import QuerySet, register


@register("scorer", "centroid_queries")
class CentroidQueries:
    uses_features = False

    def __init__(self, weights="length"):
        if weights not in ["length", "uniform"]:
            raise ValueError("Invalid centroid weights")
        self.weights = weights

    def fit(self, context, example_path, stats, folder, sources, groups):
        return self

    def predict_queries(self, core, context):
        name = context.recipe["scorer"]["representations"]["embedding"]
        weights = context.store.lengths[core] if self.weights == "length" else None
        value = np.average(context.representation(name)[core], axis=0, weights=weights)
        return QuerySet(value[None, :], context.representation_id(name))

    def score(self, core, candidates, context):
        name = context.recipe["scorer"]["representations"]["embedding"]
        q = self.predict_queries(core, context).values[0]
        q = q / max(np.linalg.norm(q), 1e-12)
        x = np.asarray(context.representation(name)[candidates], np.float32)
        return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)) @ q
