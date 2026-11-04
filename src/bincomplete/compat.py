"""Opt-in command forwarding to a separately preserved historical installation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def forward_legacy(arguments: list[str]) -> int:
    """Forward a historical non-native command without importing legacy modules.

    Parameters
    ----------
    arguments : list of str
        Historical arguments including ``--config``. Scientific evaluation and
        orchestration commands are not available through this adapter.

    Returns
    -------
    int
        Exit status of the explicitly selected legacy command.
    """
    allowed = {
        "check",
        "inspect",
        "train",
        "fit",
        "propose",
        "propose-nearest",
        "benchmark",
        "validate",
        "export",
    }
    forbidden = {"score", "recover", "study", "--native"}
    if any(value in forbidden or value.startswith("--native=") for value in arguments):
        raise ValueError("Native evaluation and study orchestration remain external to this package")
    commands = [value for value in arguments if value in allowed]
    if len(commands) != 1:
        raise ValueError("Specify one supported historical non-native command")
    configured = os.environ.get("BINCOMPLETE_LEGACY_ROOT") or os.environ.get("BIN_RECRUITMENT_LEGACY_ROOT")
    if configured:
        root = Path(configured).resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "provenance" / "legacy_v030"
    entry = root / "run.py"
    if not entry.is_file():
        raise ValueError("Set BINCOMPLETE_LEGACY_ROOT to the preserved historical source tree")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", str(entry), *arguments], cwd=root, env=environment, check=False
    ).returncode
