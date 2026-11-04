"""Exact blockwise retrieval for arbitrary learned or analytic query embeddings."""

import numpy as np

from ..core import CandidateSet, register
from .exact import ExactDistance


@register("retriever", "embedding_queries")
class EmbeddingQueries(ExactDistance):
    def __init__(
        self, top_k=1000, min_length=2000, max_length_fraction=1.0, coverage_fold=4.0, chunk_size=4096
    ):
        super().__init__(
            top_k=top_k,
            min_length=min_length,
            max_length_fraction=max_length_fraction,
            coverage_fold=coverage_fold,
            chunk_size=chunk_size,
        )

    def retrieve(self, core, candidates, values, store):
        raise ValueError("embedding_queries requires explicit predicted queries")

    def retrieve_queries(self, queries, core, candidates, values, store, representation_id):
        if queries.representation_id != representation_id:
            raise ValueError("Queries and donors use different representation identities")
        q = np.asarray(queries.values, np.float32)
        if values.ndim != 2 or q.shape[1] != values.shape[1]:
            raise ValueError("Query/donor embedding dimensions differ")
        qnorm = np.linalg.norm(q, axis=1, keepdims=True)
        if (qnorm < 1e-12).any():
            raise ValueError("Zero query vector")
        q = q / qnorm
        candidates = np.asarray(candidates, int)
        if len(set(candidates)) != len(candidates):
            raise ValueError("Duplicate retrieval candidates")
        best = [np.array([], int) for _ in range(len(q) + 1)]
        scores = [np.array([], float) for _ in best]
        for a in range(0, len(candidates), self.chunk_size):
            cc = self.eligible(core, candidates[a : a + self.chunk_size], store)
            if not len(cc):
                continue
            x = np.asarray(values[cc], np.float32)
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            if (norms < 1e-12).any():
                raise ValueError("Zero donor embedding")
            similarities = (x / norms) @ q.T
            for j in range(len(best)):
                s = similarities.max(1) if j == len(q) else similarities[:, j]
                ix, v = np.r_[best[j], cc], np.r_[scores[j], s]
                order = np.lexsort((store.ids[ix], -v))[: self.top_k]
                best[j], scores[j] = ix[order], v[order]
        return CandidateSet(best[-1], scores[-1], tuple(best[:-1]))
