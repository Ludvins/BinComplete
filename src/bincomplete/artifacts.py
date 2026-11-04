"""Content identities and atomic, immutable artifact receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 identity for JSON-compatible data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha(path: str | Path) -> str:
    """Hash a file using bounded reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: str | Path) -> Any:
    """Read a UTF-8 JSON document without altering timestamp strings."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON document in its existing directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(path: str | Path, value: Any) -> None:
    """Create a receipt or require an identical existing receipt."""
    path = Path(path)
    if path.exists():
        if fingerprint(read(path)) != fingerprint(value):
            raise ValueError(f"Immutable artifact identity changed: {path}")
    else:
        save(path, value)


def verify_outputs(folder: str | Path, receipt: dict[str, Any]) -> None:
    """Verify every relative output hash while rejecting path traversal."""
    root = Path(folder).resolve()
    for name, digest in receipt["outputs_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Artifact output changed or is missing: {name}")


def complete(folder: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Pin the completed files in an artifact directory."""
    root = Path(folder)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "complete.json")
    if any(".partial" in p.name for p in files):
        raise ValueError("Cannot complete an artifact containing partial files")
    receipt = metadata | {"outputs_sha256": {p.relative_to(root).as_posix(): sha(p) for p in files}}
    freeze(root / "complete.json", receipt)
    return receipt
