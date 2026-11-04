"""Composition, coverage and embedding membership features with bounded pair matrices."""

import numpy as np

from ..core import register
from .schema import BASE_NAMES, DNA_NAMES


class Selectable:
    requires_representation = True
    names = ()

    def __init__(self, statistics=None):
        self.statistics = tuple(self.names if statistics is None else statistics)
        if (
            not self.statistics
            or len(set(self.statistics)) != len(self.statistics)
            or not set(self.statistics) <= set(self.names)
        ):
            raise ValueError(f"Unknown/duplicate statistics; choose from {self.names}")

    def column_names(self):
        return self.statistics

    def select(self, columns):
        return np.column_stack([columns[name] for name in self.statistics])


@register("feature", "composition_distances")
class Composition(Selectable):
    names = tuple(BASE_NAMES[:4])

    def transform(self, core, candidates, values, lengths, chunk_size):
        k = np.asarray(values[core], float)
        if (k < 0).any():
            raise ValueError("Composition requires nonnegative frequencies")
        w = lengths[core].astype(float)
        w /= w.sum()
        centre = w @ k
        root = np.sqrt(centre)
        roots = np.sqrt(k)
        dispersion = np.sqrt(np.maximum(((roots - root) ** 2).sum(1) / 2, 0)) @ w
        result = np.empty((len(candidates), len(self.statistics)), np.float32)
        for a in range(0, len(candidates), min(chunk_size, 128)):
            cc = candidates[a : a + min(chunk_size, 128)]
            x = np.asarray(values[cc], float)
            if (x < 0).any():
                raise ValueError("Composition requires nonnegative frequencies")
            xr = np.sqrt(x)
            nearest = np.full(len(cc), np.inf)
            for b in range(0, len(core), 64):
                d = np.sqrt(np.maximum(((xr[:, None] - roots[None, b : b + 64]) ** 2).sum(2) / 2, 0))
                nearest = np.minimum(nearest, d.min(1))
            columns = dict(
                zip(
                    self.names,
                    [
                        np.sqrt(np.maximum(((xr - root) ** 2).sum(1) / 2, 0)),
                        nearest,
                        1
                        - (x @ centre)
                        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(centre), 1e-12),
                        np.full(len(cc), dispersion),
                    ],
                )
            )
            result[a : a + len(cc)] = self.select(columns)
        return result


@register("feature", "coverage_agreement")
class Coverage(Selectable):
    names = tuple(BASE_NAMES[4:])

    def __init__(self, statistics=None, dispersion_floor=0.35):
        super().__init__(statistics)
        if not np.isfinite(dispersion_floor) or dispersion_floor <= 0:
            raise ValueError("Coverage dispersion_floor must be positive")
        self.dispersion_floor = dispersion_floor

    def transform(self, core, candidates, values, lengths, chunk_size):
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                "coverage_agreement requires raw coverage with include_variance=true and log=false"
            )
        v = np.asarray(values[core], float)
        if (v < 0).any():
            raise ValueError("Coverage and variance must be nonnegative")
        w = lengths[core].astype(float)
        w /= w.sum()
        logs = np.log(v[:, 0] + 0.1)
        centre = float(w @ logs)
        spread = float(np.sqrt(w @ ((logs - centre) ** 2)))
        core_cv = float(w @ np.log1p(np.sqrt(v[:, 1]) / (v[:, 0] + 0.1)))
        result = np.empty((len(candidates), len(self.statistics)), np.float32)
        for a in range(0, len(candidates), chunk_size):
            cc = candidates[a : a + chunk_size]
            x = np.asarray(values[cc], float)
            if (x < 0).any():
                raise ValueError("Coverage and variance must be nonnegative")
            lc = np.log(x[:, 0] + 0.1)
            delta = lc - centre
            nearest = np.full(len(cc), np.inf)
            for b in range(0, len(logs), 128):
                nearest = np.minimum(nearest, abs(lc[:, None] - logs[None, b : b + 128]).min(1))
            columns = dict(
                zip(
                    self.names,
                    [
                        delta,
                        abs(delta),
                        nearest,
                        np.full(len(cc), spread),
                        abs(delta) / max(spread, self.dispersion_floor),
                        abs(np.log1p(np.sqrt(x[:, 1]) / (x[:, 0] + 0.1)) - core_cv),
                    ],
                )
            )
            result[a : a + len(cc)] = self.select(columns)
        return result


@register("feature", "embedding_similarities")
class EmbeddingSimilarity(Selectable):
    names = tuple(DNA_NAMES)

    def transform(self, core, candidates, values, lengths, chunk_size):
        # Historical arrays are already L2 normalized. Do not renormalize silently.
        retained = np.asarray(values[core], np.float32)
        w = lengths[core].astype(float)
        w /= w.sum()
        centre = np.asarray(w @ retained, np.float32)
        centre /= max(float(np.linalg.norm(centre)), 1e-12)
        result = np.empty((len(candidates), len(self.statistics)), np.float32)
        for a in range(0, len(candidates), min(chunk_size, 128)):
            cc = candidates[a : a + min(chunk_size, 128)]
            proposed = np.asarray(values[cc], np.float32)
            lo = np.full(len(cc), np.inf)
            hi = -lo.copy()
            total = np.zeros(len(cc))
            squares = total.copy()
            weighted = total.copy()
            for b in range(0, len(core), 128):
                sim = proposed @ retained[b : b + 128].T
                lo = np.minimum(lo, sim.min(1))
                hi = np.maximum(hi, sim.max(1))
                total += sim.astype(float).sum(1)
                squares += (sim.astype(float) ** 2).sum(1)
                weighted += sim @ w[b : b + 128]
            mean = total / len(core)
            columns = dict(
                zip(
                    self.names,
                    [
                        proposed @ centre,
                        hi,
                        lo,
                        mean,
                        np.sqrt(np.maximum(squares / len(core) - mean**2, 0)),
                        weighted,
                    ],
                )
            )
            result[a : a + len(cc)] = self.select(columns)
        return result
