"""Read-only historical serialization shim; fitting uses the public engine."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelBundle:
    classifier: object
    schema: tuple
    feature_signature: str
    training_signature: str
    training_sources: tuple
    scorer: object = None
    scoring_signature: str = None

    def predict_scores(self, batch, feature_signature):
        if batch.schema != self.schema or feature_signature != self.feature_signature:
            raise ValueError("Model feature schema or representation identity differs; refit the model")
        if self.classifier is None:
            raise ValueError("Whole-core scorer requires score(core, candidates, context)")
        if not len(batch.values):
            return np.array([], float)
        return self._check(self.classifier.predict_scores(batch.values), len(batch.values))

    @staticmethod
    def _check(values, count):
        result = np.asarray(values, float)
        if result.shape != (count,) or not np.isfinite(result).all():
            raise ValueError("Candidate scorer violates the scoring contract")
        return result

    def predict_queries(self, core, context):
        if self.scoring_signature != context.scoring_signature:
            raise ValueError("Model/query representation identity differs")
        predictor = getattr(self.scorer, "predict_queries", None)
        if not callable(predictor):
            raise ValueError("This scorer does not predict retrieval queries")
        return predictor(core, context)

    def score(self, core, candidates, context):
        if self.scoring_signature != context.scoring_signature:
            raise ValueError("Model scoring schema or representation identity differs; refit the model")
        if not len(candidates):
            return np.array([], float)
        from ..config.references import references

        for name in references(context.recipe["scorer"]).values():
            context.require_representation_rows(name, np.r_[core, candidates])
        return self._check(self.scorer.score(core, candidates, context), len(candidates))
