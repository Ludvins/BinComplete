"""Compare a built source distribution with its supported checkout contents.

This reads archive members without extracting or executing their contents.
Maintainer signoff receipts and repository-only automation are not distributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tomllib
from pathlib import Path, PurePosixPath

ROOT_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SUPPORT.md",
    "SECURITY.md",
    "MANIFEST.in",
    "pyproject.toml",
    "requirements.txt",
    "run.py",
    "zensical.toml",
    "requirements-docs.txt",
)


def supported_files(root: Path) -> dict[str, Path]:
    """List supported source, profiles, guides, examples and synthetic tests."""
    result = {name: root / name for name in ROOT_FILES}
    for folder, suffixes in (
        ("src", {".py", ".json"}),
        ("docs", {".md", ".png", ".svg", ".css", ".js"}),
        ("configs", {".json", ".toml", ".md"}),
        ("tools", {".py"}),
        ("tests", {".py"}),
    ):
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and path.suffix in suffixes
                and not any(part == "__pycache__" or part.endswith(".egg-info") for part in relative.parts)
            ):
                result[relative.as_posix()] = path
    return result


def check(root: Path, archive_path: Path) -> dict:
    """Require byte-identical supported files and exclude checkout-only receipts."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    prefix = f"{project['name']}-{project['version']}"
    expected = supported_files(root)
    errors = []
    archived = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != prefix:
                errors.append(f"Unexpected archive path: {member.name}")
                continue
            if member.isdir():
                continue
            if not member.isfile():
                errors.append(f"Archive member must be a regular file: {member.name}")
                continue
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if name in archived:
                errors.append(f"Duplicate archive member: {name}")
                continue
            stream = archive.extractfile(member)
            assert stream is not None
            with stream:
                archived[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    for name, source in sorted(expected.items()):
        if not source.is_file():
            errors.append(f"Missing supported checkout file: {name}")
        elif name not in archived:
            errors.append(f"Missing source-distribution file: {name}")
        elif hashlib.sha256(source.read_bytes()).hexdigest() != archived[name]:
            errors.append(f"Source-distribution content differs: {name}")
    for name in archived:
        parts = PurePosixPath(name).parts
        if name in {"AGENTS.md", "release_verification.json"} or "archive" in parts or "runs" in parts:
            errors.append(f"Checkout-only or project content must remain outside the distribution: {name}")
    return {
        "status": "FAILED" if errors else "SOURCE_DISTRIBUTION_READY",
        "version": project["version"],
        "supported_files_checked": len(expected),
        "archive_files_read": len(archived),
        "errors": errors,
        "scope": "Source archive content parity; no installation, scientific analysis or publishing performed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root.resolve(), args.sdist.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))
