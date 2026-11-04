"""Small environment receipt without credentials or unrelated installed packages."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import Any

PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "joblib",
    "threadpoolctl",
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "einops",
)


def environment_report(device: str = "cpu") -> dict[str, Any]:
    """Record versions and actual execution hardware without running any model."""
    from . import __version__

    packages: dict[str, str | None] = {}
    issues = []
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
            if name in {"numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"}:
                issues.append(f"Missing core dependency: {name}")
    execution = None
    if packages["torch"] is not None:
        from .numerical import torch_execution

        try:
            execution = torch_execution(device)
        except (ValueError, RuntimeError, AssertionError) as error:
            issues.append(str(error))
    elif device != "cpu":
        issues.append("CUDA execution requires the sets/embeddings optional dependencies")
    tested = json.loads(
        files("bincomplete").joinpath("profiles/standard_v1/tested_environment.json").read_text()
    )
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "packages": packages,
        "execution": execution,
        "differences_from_tested_packages": {
            name: {"current": value, "tested": tested["packages"].get(name)}
            for name, value in packages.items()
            if value != tested["packages"].get(name)
        },
        "issues": issues,
        "status": "CHECK_DEVICE_AND_DEPENDENCIES" if issues else "ENVIRONMENT_RECORDED",
        "scope": "Diagnostic receipt, not proof of identical results or validation on another dataset.",
    }
