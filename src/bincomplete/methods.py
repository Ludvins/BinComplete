"""Explicit, quality-blind numerical method profiles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .resources import ResourceLimits

METHODS = ("kmers", "dnabert_s", "single", "cosine", "hgb_full_pool", "logistic_full_pool")

METHOD_ALIASES = {
    "hgb_composition_shortlist": "kmers",
    "hgb_embedding_shortlist": "dnabert_s",
    "attention_query": "single",
}
METHOD_LABELS = {
    "kmers": "HGB with composition and coverage features, composition shortlist",
    "dnabert_s": "HGB with composition, coverage and DNABERT-S features, composition shortlist",
    "single": "Single-query transformer attention, learned-query shortlist",
    "cosine": "Length-weighted DNABERT-S cosine similarity, full candidate pool",
    "hgb_full_pool": "HGB with composition, coverage and DNABERT-S features, full candidate pool",
    "logistic_full_pool": "Scaled logistic regression with all membership features, full candidate pool",
}
METHOD_CHOICES = tuple(METHOD_ALIASES) + METHODS


def normalize_method(method: object) -> str:
    """Resolve a public method name to its stable stored numerical profile ID."""
    if not isinstance(method, str):
        raise ValueError(f"Unknown method {method}; choose from {METHOD_CHOICES}")
    name = METHOD_ALIASES.get(method, method)
    if name not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHOD_CHOICES}")
    return name


def method_catalog() -> list[dict[str, str]]:
    """Describe the six algorithms without changing their saved profile IDs."""
    preferred = {value: key for key, value in METHOD_ALIASES.items()}
    return [
        {"name": preferred.get(method, method), "profile_id": method, "description": METHOD_LABELS[method]}
        for method in METHODS
    ]


def normalize_search_params(params: dict[str, Any]) -> dict[str, Any]:
    """Accept an explicit proposal cap while preserving the stored recipe key."""
    result = deepcopy(params)
    if "max_proposals" in result:
        value = result.pop("max_proposals")
        if "max_checks" in result and result["max_checks"] != value:
            raise ValueError("max_proposals and max_checks must agree when supplied together")
        result["max_checks"] = value
    return result


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve supported input aliases before combining profile overrides."""
    result = deepcopy(settings)
    search = result.get("search")
    if isinstance(search, dict) and isinstance(search.get("params"), dict):
        search["params"] = normalize_search_params(search["params"])
    return result


def recipe(method: str, limits: ResourceLimits, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one explicit profile while keeping selection and scoring independent.

    Parameters
    ----------
    method : str
        One of the six names in ``METHODS``. No implicit all-method mode exists.
    limits : ResourceLimits
        Allocation and thread limits for this operation.
    settings : dict, optional
        Explicit parameter overrides for named components, training or seed.

    Returns
    -------
    dict
        Fully resolved method configuration saved with every model.
    """
    method = normalize_method(method)
    requested_method = method
    if method == "logistic_full_pool":
        method = "hgb_full_pool"
    feature_specs = {
        "composition": {"name": "composition_distances", "representation": "kmers"},
        "coverage": {"name": "coverage_agreement", "representation": "coverage_features"},
    }
    if method in {"dnabert_s", "hgb_full_pool"}:
        feature_specs["embedding"] = {"name": "embedding_similarities", "representation": "embedding"}
    spec: dict[str, Any] = {
        "method": method,
        "seed": 73019,
        "preprocessing": {},
        "profile_version": 1,
        "features": feature_specs if method not in {"single", "cosine"} else {},
        "resources": asdict(limits),
        "retriever": {
            "name": "exact_distance",
            "representation": "kmers",
            "params": {
                "top_k": 1000,
                "metric": "hellinger",
                "weights": "length",
                "min_length": 2000,
                "max_length_fraction": 1.0,
                "coverage_fold": 4.0,
                "chunk_size": limits.chunk_size,
            },
        },
        "masker": {
            "name": "leave_one_out_and_fraction",
            "params": {
                "extra_masks": 20,
                "fractions": [0.05, 0.1, 0.15, 0.2],
                "decoy_every": 5,
                "deduplicate": False,
            },
        },
        "positive_sampler": {"name": "hidden_members"},
        "negative_sampler": {"name": "hard_and_random", "params": {"hard": 64, "random": 32}},
        "classifier": {"name": "hist_gradient_boosting", "params": {}},
        "scorer": {"name": "feature_classifier", "params": {}},
        "training": {
            "candidate_policy": "retrieved",
            "weighting": "source_balanced",
            "max_training_rows": 2000000,
        },
        "validation": {"candidate_policy": "retrieved", "masker": None},
        "search": {"name": "ranked_alternatives", "params": {"max_checks": 20, "max_added_fraction": 1.0}},
        "pool_policy": "full" if method in {"cosine", "hgb_full_pool"} else "shortlist",
    }
    if method in {"single", "hgb_full_pool"}:
        spec["training"]["candidate_policy"] = "all_known"
        spec["negative_sampler"] = {
            "name": "embedding_hard_random",
            "representations": {"embedding": "embedding"},
            "params": {"hard": 64, "random": 32},
        }
        spec["masker"] = {"name": "leave_one_out_and_count", "params": {"extra_masks": 20, "max_hidden": 4}}
    if method == "single":
        spec["profile_version"] = 2
        spec["validation"] = {
            "candidate_policy": "all_known",
            "masker": {
                "name": "leave_one_out_and_count",
                "params": {"max_hidden": 4, "extra_masks": 20, "decoy_every": 0},
            },
        }
        spec["masker"] = {"name": "leave_one_out", "params": {}}
        spec["classifier"] = None
        spec["scorer"] = {
            "name": "set_model",
            "representations": {"embedding": "embedding", "coverage": "raw_coverage"},
            "params": {
                "objective": "completion",
                "width": 128,
                "queries": 1,
                "heads": 4,
                "layers": 2,
                "feedforward": 256,
                "max_epochs": 40,
                "batch_size": 16,
                "repeats_per_source": 8,
                "learning_rate": 0.001,
                "temperature": 0.1,
                "patience": 6,
                "validation_fraction": 0.2,
                "device": "cpu",
                "seed": 73019,
            },
        }
    if method == "cosine":
        spec["classifier"] = None
        spec["scorer"] = {
            "name": "centroid_queries",
            "representations": {"embedding": "embedding"},
            "params": {"weights": "length"},
        }
    if requested_method == "logistic_full_pool":
        spec["method"] = requested_method
        spec["classifier"] = {"name": "scaled_logistic", "params": {}}
    allowed = {
        "seed",
        "classifier",
        "scorer",
        "masker",
        "negative_sampler",
        "retriever",
        "search",
        "training",
        "validation",
        "preprocessing",
    }
    for key, value in normalize_settings(settings or {}).items():
        if key not in allowed:
            raise ValueError(f"Unsupported profile override: {key}")
        if key == "seed":
            if type(value) is not int:
                raise ValueError("seed must be an integer")
            spec[key] = value
        elif key == "preprocessing":
            if not isinstance(value, dict) or set(value) - {"composition", "coverage_dtype"}:
                raise ValueError("Invalid preprocessing settings")
            composition = value.get("composition", {})
            if not isinstance(composition, dict) or set(composition) - {
                "alphabet_order",
                "pseudocount",
                "renormalize_float32",
            }:
                raise ValueError("Invalid composition preprocessing")
            if composition.get("alphabet_order", "ACGT") not in {"ACGT", "ATGC"}:
                raise ValueError("Composition alphabet_order must be ACGT or ATGC")
            pseudocount = composition.get("pseudocount", 0.0)
            import math

            if type(pseudocount) not in (int, float) or not math.isfinite(pseudocount) or pseudocount < 0:
                raise ValueError("Composition pseudocount must be finite and nonnegative")
            if type(composition.get("renormalize_float32", False)) is not bool:
                raise ValueError("Composition renormalize_float32 must be boolean")
            if value.get("coverage_dtype", "float64") not in {"float32", "float64"}:
                raise ValueError("Coverage dtype must be float32 or float64")
            spec[key] = deepcopy(value)
        elif key == "search":
            if (
                not isinstance(value, dict)
                or set(value) - {"name", "params"}
                or not isinstance(value.get("name", spec[key]["name"]), str)
                or not value.get("name", spec[key]["name"])
                or not isinstance(value.get("params", {}), dict)
            ):
                raise ValueError("Search overrides require a component name and params object")
            if value.get("name", spec[key]["name"]) != spec[key]["name"]:
                spec[key] = {"name": value["name"], "params": deepcopy(value.get("params", {}))}
            else:
                spec[key].setdefault("params", {}).update(deepcopy(value.get("params", {})))
        elif key == "validation":
            if not isinstance(value, dict) or set(value) - {"candidate_policy", "masker"}:
                raise ValueError("Invalid validation settings")
            spec[key].update(deepcopy(value))
        elif key == "training":
            if set(value) - set(spec[key]):
                raise ValueError("Unknown training setting")
            spec[key].update(deepcopy(value))
        else:
            if not isinstance(value, dict) or set(value) - {"params"} or spec[key] is None:
                raise ValueError("Component overrides accept an explicit params object only")
            spec[key].setdefault("params", {}).update(deepcopy(value.get("params", {})))
    if spec["training"]["weighting"] != "source_balanced" or spec["training"]["candidate_policy"] not in {
        "retrieved",
        "all_known",
    }:
        raise ValueError("Unsupported training policy")
    if type(spec["training"]["max_training_rows"]) is not int or spec["training"]["max_training_rows"] < 1:
        raise ValueError("max_training_rows must be positive")
    validation = spec["validation"]
    if not isinstance(validation["candidate_policy"], str) or validation["candidate_policy"] not in {
        "retrieved",
        "all_known",
    }:
        raise ValueError("Unsupported validation candidate policy")
    masker = validation["masker"]
    if masker is not None and (
        not isinstance(masker, dict)
        or set(masker) - {"name", "params"}
        or not isinstance(masker.get("name"), str)
        or not masker["name"]
        or not isinstance(masker.get("params", {}), dict)
    ):
        raise ValueError("Invalid validation masker")
    return spec
