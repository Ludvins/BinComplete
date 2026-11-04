"""Strict experiment configuration and input-relative path resolution."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..artifacts import read, sha
from ..core import build
from ..data import mmlong2_inputs
from ..defaults import expand_experiment, merge
from ..methods import normalize_method, normalize_settings, recipe
from ..resources import ResourceLimits

PATH_INPUTS = {
    "assembly",
    "assembly_index",
    "bins",
    "memberships",
    "coverage",
    "composition",
    "groups",
    "source_ids",
    "embeddings",
    "embedding_ids",
    "embedding_space",
    "encoder_path",
    "donor_ids",
    "input_root",
}
DATA_KEYS = PATH_INPUTS | {"input_format", "coverage_profile", "prepared", "encoder_options"}
RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class ExperimentPlan:
    """Validated configuration, resolved recipes and bounded resource settings.

    Parameters
    ----------
    document : dict
        JSON-compatible input contract, with absolute paths and pinned settings.
    limits : ResourceLimits
        One shared resource policy used sequentially by every operation.
    """

    document: dict[str, Any]
    limits: ResourceLimits


def _document(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    elif path.suffix.lower() == ".json":
        value = read(path)
    else:
        raise ValueError("Configuration and settings must use .toml or .json")
    if not isinstance(value, dict):
        raise ValueError(f"Expected a configuration object: {path}")
    return value


def _keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Invalid {label}; allowed keys: {sorted(allowed)}")


def _name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value)
        or value.lower() in RESERVED
    ):
        raise ValueError(f"Invalid {label}: use a short alphanumeric name with underscores or hyphens")
    return value


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Paths must be nonempty strings")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_plan(config: str | Path, output: str | Path | None = None) -> ExperimentPlan:
    """Resolve a strict TOML/JSON experiment before creating any outputs.

    Parameters
    ----------
    config : path-like
        Experiment configuration. Relative input/settings paths use its directory.
    output : path-like, optional
        Output override; relative overrides use the current working directory.

    Returns
    -------
    ExperimentPlan
        Explicit datasets, selected method recipes, recipients and resource policy.
    """
    config = Path(config).resolve()
    raw = _document(config)
    inherited: dict[str, dict] | None = None
    defaults_provenance = None
    if type(raw.get("schema_version")) is int and raw["schema_version"] == 2:
        raw, inherited, defaults_provenance = expand_experiment(raw, config.parent)
    _keys(
        raw,
        {
            "schema_version",
            "output",
            "cache",
            "resources",
            "training",
            "evaluation",
            "datasets",
            "methods",
            "recipients",
            "on_error",
        },
        "experiment",
    )
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Experiment schema_version must be 1")
    base = config.parent
    root = Path(output).resolve() if output is not None else _path(raw.get("output"), base)
    resource_values = raw.get("resources", {})
    _keys(resource_values, set(asdict(ResourceLimits())), "resources")
    limits = ResourceLimits(**resource_values)
    cache = _path(raw["cache"], base) if "cache" in raw else root / "cache"
    if cache == root or root.is_relative_to(cache):
        raise ValueError("Experiment output must not be inside the rebuildable cache")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Declare at least one named dataset")
    if len({name.casefold() for name in datasets}) != len(datasets):
        raise ValueError("Dataset names must also be unique on case-insensitive filesystems")
    training, evaluation = raw.get("training"), raw.get("evaluation")
    if (
        not isinstance(training, str)
        or not isinstance(evaluation, str)
        or not {training, evaluation} <= set(datasets)
    ):
        raise ValueError("training and evaluation must name declared datasets")
    if set(datasets) != {training, evaluation}:
        raise ValueError("Remove unused datasets or select them as training/evaluation")
    resolved = {}
    for name, spec in datasets.items():
        _name(name, "dataset name")
        _keys(spec, DATA_KEYS, "dataset inputs")
        if "prepared" in spec:
            if set(spec) != {"prepared"}:
                raise ValueError("A prepared dataset reference cannot also declare raw inputs")
            prepared = _path(spec["prepared"], base)
            if root.is_relative_to(prepared) or prepared.is_relative_to(root):
                raise ValueError("Keep referenced prepared inputs separate from experiment outputs")
            resolved[name] = {"prepared": str(prepared)}
            continue
        values = {
            key: str(_path(value, base)) if key in PATH_INPUTS else value for key, value in spec.items()
        }
        kind = values.get("input_format", "generic")
        if kind == "mmlong2":
            if "input_root" not in values:
                raise ValueError("mmlong2 datasets require input_root")
            discovered = mmlong2_inputs(
                values["input_root"],
                **{key: values.get(key) for key in ("assembly", "bins", "memberships", "coverage")},
            )
            values.update({key: str(value) for key, value in discovered.items() if value is not None})
        elif kind != "generic":
            raise ValueError("Dataset input_format must be generic or mmlong2")
        if (
            "assembly" not in values
            or "coverage" not in values
            or ("bins" in values) == ("memberships" in values)
        ):
            raise ValueError("Datasets need assembly, coverage and exactly one of bins or memberships")
        for key in PATH_INPUTS & set(values):
            path = Path(values[key])
            if not path.exists():
                raise ValueError(f"Missing dataset input: {path}")
            if path.is_relative_to(root):
                raise ValueError("Experiment output must not contain its raw inputs")
        resolved[name] = values
    selected = raw.get("methods")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Select methods explicitly; there is no implicit run-all default")
    methods, seen, setting_files = [], set(), {}
    for entry in selected:
        _keys(entry, {"method", "id", "settings"}, "method entry")
        method = normalize_method(entry.get("method"))
        identifier = _name(entry.get("id", method), "method ID")
        if identifier.casefold() in seen:
            raise ValueError("Method IDs must be unique; give variants different IDs")
        seen.add(identifier.casefold())
        settings = entry.get("settings", {})
        if isinstance(settings, str):
            path = _path(settings, base)
            settings = _document(path)
            setting_files[str(path)] = sha(path)
        if not isinstance(settings, dict):
            raise ValueError("Method settings must be an object or a TOML/JSON file path")
        settings = normalize_settings(settings)
        if inherited is not None:
            settings = merge(inherited[method], settings)
        spec = recipe(method, limits, settings)
        for category in (
            "retriever",
            "masker",
            "positive_sampler",
            "negative_sampler",
            "scorer",
            "search",
            "classifier",
        ):
            if spec[category] is not None:
                build(category, spec[category])
        methods.append({"id": identifier, "method": method, "settings": settings, "recipe": spec})
    recipients = raw.get("recipients")
    if isinstance(recipients, str):
        path = _path(recipients, base)
        recipients = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        setting_files[str(path)] = sha(path)
    if recipients is not None and (
        not isinstance(recipients, list)
        or not recipients
        or any(not isinstance(name, str) or not name for name in recipients)
        or len(set(recipients)) != len(recipients)
    ):
        raise ValueError("recipients must be a nonempty unique bin-ID list or a text-file path")
    policy = raw.get("on_error", "continue")
    if policy not in {"continue", "stop"}:
        raise ValueError("on_error must be continue or stop")
    document = {
        "schema_version": 1,
        "source_config": str(config),
        "source_sha256": sha(config),
        "settings_files_sha256": setting_files,
        "output": str(root),
        "cache": str(cache),
        "resources": asdict(limits),
        "training": training,
        "evaluation": evaluation,
        "datasets": resolved,
        "methods": methods,
        "recipients": recipients,
        "on_error": policy,
    }
    if defaults_provenance is not None:
        document["defaults"] = defaults_provenance
    return ExperimentPlan(document, limits)
