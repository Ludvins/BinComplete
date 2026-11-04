import numpy as np

from ..core import CandidateSet, register
from ..features.math import METRICS, distance, summary


@register("retriever", "exact_distance")
class ExactDistance:
    def __init__(
        self,
        top_k=1000,
        metric="cosine",
        summary="centroid",
        weights="length",
        min_length=2000,
        max_length_fraction=1.0,
        coverage_fold=4.0,
        chunk_size=4096,
    ):
        if type(top_k) is not int or top_k < 1 or metric not in METRICS - {"cosine_similarity"}:
            raise ValueError("Invalid retriever top_k or distance")
        if (
            summary not in ["centroid", "median"]
            or weights not in ["length", "uniform"]
            or (summary == "median" and weights != "uniform")
        ):
            raise ValueError("Invalid retrieval summary")
        if (
            min_length < 0
            or max_length_fraction <= 0
            or (coverage_fold is not None and coverage_fold < 1)
            or type(chunk_size) is not int
            or chunk_size < 1
        ):
            raise ValueError("Invalid retrieval eligibility or chunk size")
        self.top_k, self.metric = top_k, metric
        self.summary, self.weights = summary, weights
        self.min_length, self.max_length_fraction = min_length, max_length_fraction
        self.coverage_fold, self.chunk_size = coverage_fold, chunk_size

    def eligible(self, core, candidates, store):
        core, candidates = np.asarray(core, int), np.asarray(candidates, int)
        good = (store.lengths[candidates] >= self.min_length) & (
            store.lengths[candidates] <= self.max_length_fraction * store.lengths[core].sum()
        )
        good &= ~np.isin(candidates, core)
        if self.coverage_fold is not None:
            coverage = store.array("coverage")
            logs = np.log(coverage[core] + 0.1)
            centre = float(np.average(logs, weights=store.lengths[core]))
            good &= np.abs(np.log(coverage[candidates] + 0.1) - centre) <= np.log(self.coverage_fold)
        return candidates[good]

    def retrieve(self, core, candidates, values, store):
        core, candidates = np.asarray(core, int), np.asarray(candidates, int)
        if not len(core) or len(set(candidates)) != len(candidates):
            raise ValueError("Invalid retrieval core or duplicate candidates")
        centre = summary(values[core], store.lengths[core], self.summary, self.weights)
        best = np.array([], dtype=np.int64)
        costs = np.array([], float)
        for a in range(0, len(candidates), self.chunk_size):
            cc = self.eligible(core, candidates[a : a + self.chunk_size], store)
            if not len(cc):
                continue
            dd = distance(values[cc], centre, self.metric)
            joined, joined_cost = np.r_[best, cc], np.r_[costs, dd]
            if len(joined) > self.top_k:
                threshold = np.partition(joined_cost, self.top_k - 1)[self.top_k - 1]
                keep = np.flatnonzero(joined_cost <= threshold)
                joined, joined_cost = joined[keep], joined_cost[keep]
            order = np.lexsort((store.ids[joined], joined_cost))[: self.top_k]
            best, costs = joined[order], joined_cost[order]
        return CandidateSet(best, -costs)
