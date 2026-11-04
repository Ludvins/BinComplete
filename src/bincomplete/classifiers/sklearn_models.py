import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..core import register


class SklearnClassifier:
    score_kind = "membership_score"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        if len(np.unique(y)) != 2:
            raise ValueError("Training requires both known positive and negative labels")
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    @property
    def parameters(self):
        return self.estimator.get_params(deep=False)

    def predict_scores(self, X):
        positive = list(self.estimator.classes_).index(1)
        result = self.estimator.predict_proba(X)[:, positive]
        if not np.isfinite(result).all():
            raise ValueError("Classifier produced nonfinite scores")
        return result


def estimator(kind, params, defaults):
    unknown = set(params) - set(kind().get_params())
    if unknown:
        raise ValueError(f"Unknown {kind.__name__} hyperparameters: {sorted(unknown)}")
    if params.get("n_jobs", 1) not in [None, 1]:
        raise ValueError("Classifier workers must use n_jobs=1; pipeline owns the shared thread budget")
    instance = kind(**(defaults | params))
    # Validate hyperparameter types/ranges before loading representations or fitting.
    instance._validate_params()
    return SklearnClassifier(instance)


@register("classifier", "hist_gradient_boosting")
def hist_gradient_boosting(**params):
    return estimator(
        HistGradientBoostingClassifier,
        params,
        dict(
            max_iter=150,
            max_leaf_nodes=7,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=73019,
        ),
    )
