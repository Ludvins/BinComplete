"""Global installation settings and explicit resolution for compact experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from .cache import default_cache
from .methods import normalize_method, normalize_search_params, normalize_settings, recipe
from .resources import ResourceLimits
from .standard import STANDARD_METHODS, standard_settings

DNABERT_REPO = "zhihan1996/DNABERT-S"
DNABERT_REVISION = "1cdf84d992ace6f3e75c7356774b4da088c8dc7c"


def config_path() -> Path:
    """Locate the user configuration, allowing an explicit environment override."""
    if os.environ.get("BINCOMPLETE_CONFIG"):
        return Path(os.environ["BINCOMPLETE_CONFIG"]).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        return base / "BinComplete/config.toml"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "bincomplete/config.toml"


def model_root() -> Path:
    """Keep downloaded model assets separate from the evictable derived cache."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "bincomplete/models"


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge settings, replacing component parameters when its name changes."""
    result = deepcopy(base)
    for key, value in overrides.items():
        old = result.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            changed = "name" in value and value.get("name") != old.get("name")
            result[key] = deepcopy(value) if changed else merge(old, value)
        else:
            result[key] = deepcopy(value)
    return result


def builtin_defaults() -> dict[str, Any]:
    """Return the fixed preset and portable installation defaults."""
    return {
        "schema_version": 1,
        "preset": "standard_v1",
        "device": "cpu",
        "encoder_path": str(model_root() / "dnabert_s" / DNABERT_REVISION),
        "cache": str(default_cache()),
        "resources": {"threads": 2, "memory_gib": 6, "cache_gib": 10, "chunk_size": 4096},
        "search": standard_settings("cosine")["search"]["params"],
        "retriever": {"top_k": 1000, "min_length": 2000, "coverage_fold": 4.0, "max_length_fraction": 1.0},
        "encoder": {
            "pooling": "non_special_bp_weighted_windows",
            "window_bp": 6000,
            "windows": 4,
            "max_tokens": 2048,
            "reverse_complement": True,
            "storage_dtype": "float16",
            "precision": "auto",
            "batch_size": 8,
        },
        "enabled_methods": list(STANDARD_METHODS),
        "methods": {},
    }


def load_defaults(path: str | Path | None = None, *, required: bool = False) -> tuple[dict, dict]:
    """Read and validate global defaults without loading models or using the network."""
    source = Path(path).expanduser().resolve() if path is not None else config_path().resolve()
    raw: dict[str, Any] = {}
    digest = None
    if source.exists():
        content = source.read_bytes()
        text = content.decode("utf-8-sig")
        raw = json.loads(text) if source.suffix.lower() == ".json" else tomllib.loads(text)
        digest = hashlib.sha256(content).hexdigest()
    elif required or path is not None:
        raise ValueError(f"Missing global configuration: {source}; run bincomplete config init")
    allowed = set(builtin_defaults()) | {"cache"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError(f"Unknown global configuration keys; allowed: {sorted(allowed)}")
    if type(raw.get("schema_version", 1)) is not int or raw.get("schema_version", 1) != 1:
        raise ValueError("Global schema_version must be 1")
    if raw.get("preset", "standard_v1") != "standard_v1":
        raise ValueError("Unknown preset; choose standard_v1")
    if raw.get("device", "cpu") not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("Global device must be cpu, cuda or cuda:0")
    for key in ("resources", "search", "retriever", "encoder", "methods"):
        if not isinstance(raw.get(key, {}), dict):
            raise ValueError(f"Global {key} must be a table")
    raw["search"] = normalize_search_params(raw.get("search", {}))
    method_overrides = {}
    for name, value in raw.get("methods", {}).items():
        profile_id = normalize_method(name)
        if profile_id in method_overrides:
            raise ValueError("Global method aliases must not duplicate a stored profile ID")
        if not isinstance(value, dict):
            raise ValueError("Global method overrides must be tables")
        method_overrides[profile_id] = normalize_settings(value)
    raw["methods"] = method_overrides
    for key in ("resources", "search", "retriever", "encoder"):
        if set(raw.get(key, {})) - set(builtin_defaults()[key]):
            raise ValueError(f"Unknown global {key} setting")
    if set(raw.get("methods", {})) - set(STANDARD_METHODS):
        raise ValueError("Unknown method in global configuration")
    result = merge(builtin_defaults(), raw)
    enabled = result["enabled_methods"]
    if isinstance(enabled, list) and all(isinstance(value, str) for value in enabled):
        enabled = [normalize_method(value) for value in enabled]
        result["enabled_methods"] = enabled
    if (
        not isinstance(enabled, list)
        or not enabled
        or any(not isinstance(x, str) for x in enabled)
        or len(set(enabled)) != len(enabled)
        or not set(enabled) <= set(STANDARD_METHODS)
    ):
        raise ValueError("enabled_methods must be a nonempty unique list of supported methods")
    ResourceLimits(**result["resources"])
    for key in ("encoder_path", "cache"):
        if key in result:
            value = result[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Global {key} must be a nonempty path")
            destination = Path(value).expanduser()
            result[key] = str(
                (source.parent / destination).resolve()
                if not destination.is_absolute()
                else destination.resolve()
            )
    for value in result["methods"].values():
        if not isinstance(value, dict):
            raise ValueError("Global method overrides must be tables")
    provenance = {
        "preset": "standard_v1",
        "path": str(source) if source.exists() else None,
        "sha256": digest,
        "resolved": deepcopy(result),
    }
    return result, provenance


def method_defaults(defaults: dict[str, Any], device: str | None = None) -> dict[str, dict]:
    """Resolve preset, shared search/retrieval settings and method overrides."""
    selected_device = device if device is not None else defaults["device"]
    result = {}
    for method in STANDARD_METHODS:
        settings = standard_settings(method, device=selected_device)
        settings["search"]["params"].update(defaults["search"])
        settings["retriever"]["params"].update(defaults["retriever"])
        result[method] = merge(settings, defaults["methods"].get(method, {}))
    return result


def expand_experiment(raw: dict[str, Any], base: Path) -> tuple[dict, dict, dict]:
    """Expand only schema 2; schema 1 experiments never consult ambient defaults."""
    result = deepcopy(raw)
    path = result.pop("global_config", None)
    if path is not None:
        if not isinstance(path, str) or not path:
            raise ValueError("global_config must be a path")
        candidate = Path(path).expanduser()
        path = candidate if candidate.is_absolute() else base / candidate
    defaults, provenance = load_defaults(path)
    preset = result.pop("preset", defaults["preset"])
    if preset != "standard_v1":
        raise ValueError("Unknown experiment preset")
    device = result.pop("device", defaults["device"])
    settings = method_defaults(defaults, device)
    provenance["effective_device"] = device
    result["schema_version"] = 1
    result.setdefault("output", "results")
    result.setdefault("training", "sample")
    result.setdefault("evaluation", "sample")
    result.setdefault("on_error", "stop")
    result["resources"] = merge(defaults["resources"], result.get("resources", {}))
    if "cache" in defaults:
        result.setdefault("cache", defaults["cache"])
    selected = result.get("methods", defaults["enabled_methods"])
    if not isinstance(selected, list):
        raise ValueError("methods must be a list")
    result["methods"] = [{"method": value} if isinstance(value, str) else value for value in selected]
    needs_embeddings = False
    for entry in result["methods"]:
        if not isinstance(entry, dict):
            raise ValueError("Select supported methods with a method name or method/settings object")
        entry["method"] = normalize_method(entry.get("method"))
        overrides = entry.get("settings", {})
        if isinstance(overrides, str):
            settings_path = Path(overrides).expanduser()
            if not settings_path.is_absolute():
                settings_path = base / settings_path
            content = settings_path.read_text(encoding="utf-8-sig")
            overrides = (
                json.loads(content) if settings_path.suffix.lower() == ".json" else tomllib.loads(content)
            )
        if not isinstance(overrides, dict):
            raise ValueError("Method settings must be an object or a TOML/JSON path")
        overrides = normalize_settings(overrides)
        spec = recipe(
            entry["method"],
            ResourceLimits(**result["resources"]),
            merge(settings[entry["method"]], overrides),
        )
        needs_embeddings |= uses_embedding(spec)
    datasets = result.get("datasets")
    if not isinstance(datasets, dict) or any(not isinstance(x, dict) for x in datasets.values()):
        raise ValueError("Declare named dataset input tables")
    for spec in datasets.values():
        if "prepared" in spec or "embeddings" in spec or not needs_embeddings:
            continue
        spec.setdefault("encoder_path", defaults["encoder_path"])
        options = dict(defaults["encoder"], device=device)
        if options["precision"] == "auto":
            options["precision"] = "float32" if device == "cpu" else "bfloat16"
        spec["encoder_options"] = merge(options, spec.get("encoder_options", {}))
    return result, settings, provenance


def uses_embedding(value: Any) -> bool:
    """Inspect resolved component representation dependencies, including overrides."""
    if isinstance(value, dict):
        return any(uses_embedding(item) for item in value.values())
    if isinstance(value, list):
        return any(uses_embedding(item) for item in value)
    return isinstance(value, str) and value == "embedding"


def _toml(document: dict[str, Any], prefix: str = "") -> str:
    lines = []
    if prefix:
        lines.append(f"[{prefix}]")
    for key, value in document.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {json.dumps(value)}")
    for key, value in document.items():
        if isinstance(value, dict):
            lines.extend(["", _toml(value, f"{prefix}.{key}" if prefix else key)])
    return "\n".join(lines)


def write_defaults(
    path: str | Path | None = None, *, encoder_path: str | Path | None = None, device: str = "cpu"
) -> Path:
    """Create one editable global TOML/JSON file without overwriting a configuration."""
    # Validate device before creating directories or files.
    standard_settings("cosine", device=device)
    target = Path(path).expanduser().resolve() if path is not None else config_path().resolve()
    document = builtin_defaults()
    document["device"] = device
    document["search"]["max_proposals"] = document["search"].pop("max_checks")
    if encoder_path is not None:
        document["encoder_path"] = str(Path(encoder_path).expanduser().resolve())
    # Expose common training controls; the preset supplies all other parameters.
    for method in ("kmers", "dnabert_s", "hgb_full_pool", "logistic_full_pool"):
        document["methods"][method] = {"classifier": standard_settings(method)["classifier"]}
    params = standard_settings("single", device=device)["scorer"]["params"]
    document["methods"]["single"] = {
        "scorer": {
            "params": {
                key: params[key]
                for key in (
                    "width",
                    "layers",
                    "heads",
                    "feedforward",
                    "max_epochs",
                    "batch_size",
                    "learning_rate",
                    "patience",
                )
            }
        }
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, indent=2) if target.suffix.lower() == ".json" else _toml(document)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")
    return target
