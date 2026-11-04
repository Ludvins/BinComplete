"""Lazy data bindings for numerical components; no scientific evaluator is imported."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from . import features, sampling, scorers  # noqa: F401
from .artifacts import fingerprint, save, sha
from .cache import DerivedCache
from .classifiers import logistic, sklearn_models  # noqa: F401
from .core import build
from .data import Dataset
from .features.composer import FeatureComposer
from .representations.kmers import frequencies
from .resources import ResourceLimits
from .retrieval import exact, queries  # noqa: F401
from .search import expanded, ranked_alternatives  # noqa: F401


class RowView:
    """Align an existing matrix through row indirection without a matrix copy."""

    def __init__(self, matrix: np.ndarray, rows: np.ndarray):
        self.matrix, self.rows = matrix, rows
        self.shape = (len(rows), matrix.shape[1])
        self.ndim, self.dtype = 2, matrix.dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            row, column = key
            return self.matrix[self.rows[row], column]
        return self.matrix[self.rows[key]]


class Store:
    """Expose prepared metadata and bounded sequences to numerical components."""

    def __init__(self, dataset: Dataset, cache: DerivedCache, limits: ResourceLimits):
        self.dataset, self.cache, self.limits = dataset, cache, limits
        self.manifest = dataset.verify()
        self.inputs = {"contigs": dataset.path / "dataset.json"}
        self.maps: dict[str, np.memmap] = {}
        for name in ("ids", "lengths", "coverage", "sequence_hashes", "pool"):
            self.maps[name] = np.load(dataset.array_path(name + ".npy"), mmap_mode="r", allow_pickle=False)
        if self.manifest.get("composition_source") is not None:
            self.maps["composition"] = np.load(
                dataset.array_path("composition.npy"), mmap_mode="r", allow_pickle=False
            )
        self.ids, self.lengths = self.maps["ids"], self.maps["lengths"]
        limits.require(len(self.ids) * 768, "Metadata and row lookup")
        self.lookup = {str(name): i for i, name in enumerate(self.ids)}
        self.row_identity = self.manifest["identity"]

    def indices(self, ids: list[str]) -> np.ndarray:
        return np.array([self.lookup[name] for name in ids], np.int64)

    def input_hash(self, name: str) -> str:
        if name == "training_bins":
            return fingerprint(self.manifest["bins"])
        return self.manifest["identity"]

    def array_identity(self, name: str) -> dict[str, str]:
        return {"dataset": self.manifest["identity"], "array": name}

    def array(self, name: str) -> np.ndarray:
        return self.maps["coverage"][:, 0] if name == "coverage" else self.maps[name]

    def protected(self) -> set[str]:
        return {name for members in self.manifest["bins"].values() for name in members}

    def sources(self) -> tuple[list[dict[str, Any]], dict[str, str]]:
        return (
            [
                {
                    "genome_id": self.manifest.get("source_ids", {}).get(name, name),
                    "bin_name": name,
                    "members": members,
                }
                for name, members in self.manifest["bins"].items()
                if len(members) >= 2
            ],
            {
                self.manifest.get("source_ids", {}).get(name, name): group
                for name, group in self.manifest["groups"].items()
            },
        )

    def sequences(self, rows: Any):
        return self.dataset.sequences(rows, self.limits)

    def sequence_windows(self, rows: Any, *, window_bp: int, count: int):
        return self.dataset.sequence_windows(rows, self.limits, window_bp=window_bp, count=count)

    def close(self) -> None:
        for array in self.maps.values():
            array._mmap.close()
        self.maps.clear()
        self.lookup.clear()


class Context:
    """Bind one dataset and one resolved method for the lifetime of an operation."""

    def __init__(
        self,
        dataset: Dataset,
        spec: dict[str, Any],
        output: Path,
        cache_root: str | Path | None,
        limits: ResourceLimits,
    ):
        self.recipe, self.output, self.limits = spec, output, limits
        self.cache = DerivedCache(cache_root, limits)
        self._representations: dict[str, Any] = {}
        self._external_maps: list[np.memmap] = []
        self._threads = threadpool_limits(limits=limits.threads)
        try:
            self.store = Store(dataset, self.cache, limits)
            self.reserved_bytes = len(self.store.ids) * 768 + min(64 * 1024**2, limits.memory_bytes // 16)
            self._initialize(spec, limits)
        except BaseException:
            self.close()
            raise

    def _initialize(self, spec: dict[str, Any], limits: ResourceLimits) -> None:
        self.code = {
            p.relative_to(Path(__file__).parent).as_posix(): sha(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        }
        self.runtime: dict[str, Any] = {}
        self.components = {
            name: build(name, spec[name])
            for name in ("retriever", "masker", "positive_sampler", "negative_sampler", "scorer", "search")
        }
        if spec["classifier"] is not None:
            self.components["classifier"] = build("classifier", spec["classifier"])
        self.features = FeatureComposer(
            spec["features"],
            self.representation,
            limits.chunk_size,
            self.representation_id,
            allow_empty=not spec["features"],
        )
        self.scoring_signature = fingerprint(self.compatibility())

    def status(self, phase: str, **values: Any) -> None:
        save(self.output / "progress.json", {"phase": phase} | values)

    def compatibility(self) -> dict[str, Any]:
        method = self.recipe["method"]
        embedding = self.store.manifest["embedding"]
        if method != "kmers" and embedding is None:
            raise ValueError(
                f"{method} requires frozen DNABERT-S embeddings with an embedding-space manifest"
            )
        if (
            method in {"kmers", "dnabert_s", "hgb_full_pool", "logistic_full_pool"}
            and not self.store.manifest["coverage_variance"]
        ):
            raise ValueError(
                "Feature-classifier profiles require measured coverage variance; it cannot be fabricated"
            )
        space = embedding["space"] if embedding is not None and method != "kmers" else None
        if space is not None and space["encoder"] != "dnabert_s":
            raise ValueError("This profile requires the declared DNABERT-S vector space")
        numerical_code = {
            name: digest
            for name, digest in self.code.items()
            if name.startswith(
                (
                    "features/",
                    "representations/",
                    "scorers/",
                    "classifiers/",
                    "retrieval/",
                    "sampling/",
                    "masking.py",
                )
            )
        }
        return {
            "profile_version": self.recipe["profile_version"],
            "method": method,
            "features": self.recipe["features"],
            "schema": list(self.features.schema) if hasattr(self, "features") else [],
            "composition": {
                "k": 4,
                "canonical": True,
                "dtype": "float32",
                "formula": "supplied_table_normalized"
                if self.store.manifest.get("composition_source") is not None
                else "valid_window_frequency",
            },
            "coverage": "raw_mean_variance_log_offset_0.1",
            "embedding_space": space,
            "numerical_code": numerical_code,
            "preprocessing": self.recipe.get("preprocessing", {}),
        }

    def representation_id(self, name: str) -> str:
        embedding = self.store.manifest["embedding"]
        return fingerprint(
            {
                "rows": self.store.row_identity,
                "name": name,
                "embedding": embedding if name == "embedding" else None,
            }
        )

    def representation(self, name: str):
        if name in self._representations:
            return self._representations[name]
        if name == "raw_coverage":
            result = self.store.maps["coverage"][:, :1].astype(
                self.recipe.get("preprocessing", {}).get("coverage_dtype", "float64"), copy=False
            )
        elif name == "coverage_features":
            result = self.store.maps["coverage"].astype(
                self.recipe.get("preprocessing", {}).get("coverage_dtype", "float64"), copy=False
            )
        elif name == "embedding":
            spec = self.store.manifest["embedding"]
            if spec is None:
                raise ValueError("No frozen embeddings supplied")
            if spec.get("provider") == "frozen_local":
                from .representations.dna import frozen_array

                result = frozen_array(self, spec)
            else:
                matrix = np.load(spec["path"], mmap_mode="r", allow_pickle=False)
                rows = np.load(
                    self.store.dataset.path / "embedding_rows.npy", mmap_mode="r", allow_pickle=False
                )
                self._external_maps.extend([matrix, rows])
                result = RowView(matrix, rows)
        elif name == "kmers":
            supplied = self.store.manifest.get("composition_source")
            identity = {
                "kind": "canonical4_v1",
                "assembly": self.store.manifest["inputs_sha256"][self.store.manifest["assembly"]],
                "implementation": sha(Path(__file__).parent / "representations" / "kmers.py"),
                "preprocessing": self.recipe.get("preprocessing", {}).get("composition", {}),
                "supplied_table_sha256": self.store.manifest["inputs_sha256"][supplied]
                if supplied is not None
                else None,
            }

            def writer(path: Path) -> None:
                array = np.lib.format.open_memmap(
                    path, mode="w+", dtype="float32", shape=(len(self.store.ids), 136)
                )
                try:
                    if supplied is not None:
                        source = self.store.maps["composition"]
                        for start in range(0, len(source), self.limits.chunk_size):
                            block = np.asarray(source[start : start + self.limits.chunk_size], np.float32)
                            total = (
                                np.cumsum(block, axis=1, dtype=np.float32)[:, -1:]
                                if identity["preprocessing"].get("renormalize_float32", False)
                                else block.sum(axis=1, keepdims=True, dtype=np.float64)
                            )
                            array[start : start + len(block)] = block / total
                    else:
                        for row, sequence in enumerate(self.store.sequences(range(len(self.store.ids)))):
                            array[row] = frequencies(sequence, 4, True, **identity["preprocessing"])
                    array.flush()
                finally:
                    array._mmap.close()

            result = self.cache.array(identity, writer, expected_bytes=len(self.store.ids) * 136 * 4)
        else:
            raise ValueError(f"Unknown representation: {name}")
        self._representations[name] = result
        return result

    def require_representation_rows(self, name: str, rows: np.ndarray) -> None:
        if len(rows) and (rows.min() < 0 or rows.max() >= len(self.store.ids)):
            raise ValueError("Representation row outside the prepared population")

    def retrieve(self, core: np.ndarray, candidates: np.ndarray):
        return self.components["retriever"].retrieve(
            core, candidates, self.representation("kmers"), self.store
        )

    def close(self) -> None:
        self._representations.clear()
        for array in self._external_maps:
            array._mmap.close()
        self._external_maps.clear()
        self.cache.close()
        if hasattr(self, "store"):
            self.store.close()
        self._threads.restore_original_limits()
