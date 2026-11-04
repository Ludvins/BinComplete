from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class CandidateSet:
    indices: np.ndarray
    scores: np.ndarray
    by_query: tuple = ()


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray
    schema: tuple[str, ...]


@dataclass(frozen=True)
class Episode:
    source: str
    group: str
    core: np.ndarray
    hidden: np.ndarray
    absent: bool = False


@dataclass(frozen=True)
class QuerySet:
    """Predictions expressed in one explicitly identified frozen vector space."""

    values: np.ndarray
    representation_id: str

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Query vectors must be a nonempty finite matrix")
        if not isinstance(self.representation_id, str) or not self.representation_id:
            raise ValueError("Query vectors require a representation identity")


class QueryPredictor(Protocol):
    def predict_queries(self, core: np.ndarray, context: Any) -> QuerySet: ...


class QueryRetriever(Protocol):
    def retrieve_queries(
        self, queries: QuerySet, core, candidates, values, store, representation_id: str
    ) -> CandidateSet: ...


class Representation(Protocol):
    def identity(self, store: Any) -> dict: ...
    def transform(self, store: Any) -> np.ndarray: ...


class Feature(Protocol):
    def column_names(self) -> tuple[str, ...]: ...
    def transform(
        self,
        core: np.ndarray,
        candidates: np.ndarray,
        values: np.ndarray,
        lengths: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray: ...


class Classifier(Protocol):
    score_kind: str

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None): ...
    def predict_scores(self, X: np.ndarray) -> np.ndarray: ...


class CandidateScorer(Protocol):
    uses_features: bool

    def fit(self, context: Any, example_path: Any, stats: dict, folder: Any, sources: list, groups: dict): ...
    def score(self, core: np.ndarray, candidates: np.ndarray, context: Any) -> np.ndarray: ...


class Masker(Protocol):
    def generate(self, members, lengths, rng) -> Iterator[Any]: ...


class Sampler(Protocol):
    def sample(self, episode: Episode, context: Any, rng) -> np.ndarray: ...


class Retriever(Protocol):
    def eligible(self, core, candidates, store) -> np.ndarray: ...
    def retrieve(self, core, candidates, values, store) -> CandidateSet: ...


class Search(Protocol):
    def propose(self, core, ranked, store) -> Iterator[dict]: ...
