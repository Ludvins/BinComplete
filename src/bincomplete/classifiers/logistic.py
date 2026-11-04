"""Source-weighted logistic membership classification."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..core import register


@register("classifier", "scaled_logistic")
class ScaledLogistic:
    score_kind = "membership_score"

    def __init__(self, C=1.0, max_iter=1000, tol=1e-6, random_state=73019):
        self.scaler = StandardScaler()
        self.estimator = LogisticRegression(
            C=C, max_iter=max_iter, tol=tol, random_state=random_state, solver="lbfgs"
        )
        self.estimator._validate_params()

    @property
    def parameters(self):
        return self.estimator.get_params(deep=False) | {
            "feature_scaling": "source_weighted_standard_scaler_train_only"
        }

    def fit(self, X, y, sample_weight=None):
        if len(np.unique(y)) != 2:
            raise ValueError("Membership training requires both labels")
        self.scaler.fit(X, sample_weight=sample_weight)
        normalized = self.scaler.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.estimator.fit(normalized, y, sample_weight=sample_weight)
        return self

    def predict_scores(self, X):
        positive = list(self.estimator.classes_).index(1)
        result = self.estimator.predict_proba(self.scaler.transform(X))[:, positive]
        if not np.isfinite(result).all():
            raise ValueError("Logistic scores are not finite")
        return result
