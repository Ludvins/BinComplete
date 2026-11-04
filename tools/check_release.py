"""Check source readiness; --public checks the maintainer's checkout signoff.

This does not run metagenomic analyses or publish anything. A source-only pass
does not imply that clean-environment tests or hosted CI have completed.
The checkout signoff receipt is deliberately not included in source distributions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

REQUIRED_RELEASE_CHECKS = (
    "clean_windows",
    "clean_linux",
    "real_encoder_windows",
    "real_encoder_linux",
    "numerical_regression",
    "repository_hygiene",
    "package_contents",
    "source_distribution",
    "lint",
    "format",
    "types",
)


def _metadata_errors(root: Path, expected_version: str) -> list[str]:
    """Read version declarations without importing package code or dependencies."""
    errors = []
    namespace = root / "src/bincomplete/__init__.py"
    package_version = None
    if namespace.is_file():
        for node in ast.parse(namespace.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
            ):
                if isinstance(node.value, ast.Constant):
                    package_version = node.value.value
    if package_version != expected_version:
        errors.append("Package __version__ does not match project.version")
    citation = root / "CITATION.cff"
    if citation.is_file():
        match = re.search(r"^version:\s*([^\r\n]+)$", citation.read_text(encoding="utf-8"), re.MULTILINE)
        citation_version = match.group(1).strip().strip("\"'") if match else None
        if citation_version != expected_version:
            errors.append("CITATION.cff version does not match project.version")
    readme = root / "README.md"
    if readme.is_file() and re.search(
        r"\bversion\s+\d+\.\d+\.\d+(?:[a-z]+\d*)?\b", readme.read_text(encoding="utf-8"), re.IGNORECASE
    ):
        errors.append("README must refer to the CLI/package version instead of duplicating a release number")
    return errors


def check(root: Path, public: bool = False) -> dict:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    errors = []
    for name in [
        "LICENSE",
        "README.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SUPPORT.md",
        "SECURITY.md",
        "THIRD_PARTY.md",
        "CHANGELOG.md",
        "requirements.txt",
        "run.py",
        "docs/validation.md",
        "docs/security.md",
    ]:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    errors.extend(_metadata_errors(root, project["version"]))
    if not project.get("authors") or not project.get("maintainers"):
        errors.append("Author and maintainer metadata required")
    if project.get("license") != "MIT" or "LICENSE" not in project.get("license-files", []):
        errors.append("MIT licence expression and LICENSE file must be included in package metadata")
    if "einops>=0.8.2" not in project["optional-dependencies"]["embeddings"]:
        errors.append("Encoder runtime must declare einops")
    for folder in ("configs", "docs"):
        if (root / folder / "archive").exists():
            errors.append(f"Project archives must be kept outside the distribution: {folder}/archive")
    reference = root / "src/bincomplete/profiles/standard_v1/tested_environment.json"
    if reference.is_file():
        dependency_profile = json.loads(reference.read_text(encoding="utf-8"))
        if set(dependency_profile) != {"schema_version", "recipe", "python", "packages", "scope"}:
            errors.append(
                "Packaged dependency diagnostics must not contain project results or device receipts"
            )
    links = 0
    guides = list(root.glob("*.md")) + list((root / "docs").rglob("*.md"))
    guides += list((root / "configs").rglob("*.md"))
    for guide in guides:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", guide.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            links += 1
            if not (guide.parent / unquote(target)).exists():
                errors.append(f"Broken guide link in {guide.relative_to(root)}: {target}")
    evidence = root / "release_verification.json"
    evidence_checked = False
    if public:
        repository = project.get("urls", {}).get("Repository", "")
        if not repository.startswith("https://") or any(
            x in repository for x in ["example.", "OWNER", "REPO"]
        ):
            errors.append("Confirmed public Repository URL is required")
        contacts = project.get("maintainers", [])
        if not any(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", c.get("email", "")) for c in contacts):
            errors.append("Confirmed institutional maintainer email is required")
        if not evidence.is_file():
            errors.append("Checkout-only release_verification.json is required for public readiness")
        else:
            review = json.loads(evidence.read_text(encoding="utf-8"))
            evidence_checked = True
            if review.get("status") != "VERIFIED" or review.get("version") != project["version"]:
                errors.append("Release verification status/version does not match")
            for name, digest in review.get("source_sha256", {}).items():
                path = root / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    errors.append(f"Source changed since release verification: {name}")
            if not review.get("source_sha256"):
                errors.append("Release verification has no source manifest")
            ignored = {
                ".git",
                "__pycache__",
                ".ruff_cache",
                ".mypy_cache",
                ".pytest_cache",
                "build",
                "dist",
                "runs",
                "site",
                ".cache",
                ".venv",
                ".venv-docs",
            }
            current = {
                p.relative_to(root).as_posix()
                for p in root.rglob("*")
                if p.is_file()
                and not any(x in ignored or x.endswith(".egg-info") for x in p.relative_to(root).parts)
                and p.suffix not in {".pyc", ".partial", ".lock"}
                and p.name != "release_verification.json"
            }
            if current != set(review.get("source_sha256", {})):
                errors.append("Release source file set differs from its verification manifest")
            checks = review.get("checks", {})
            if not isinstance(checks, dict):
                errors.append("Release checks must map names to passing statuses")
                checks = {}
            for key in sorted(set(REQUIRED_RELEASE_CHECKS) | set(checks)):
                if checks.get(key) != "PASSED":
                    errors.append(f"Missing or non-passing release check: {key}")
    return {
        "status": "FAILED" if errors else "PUBLIC_READY" if public else "SOURCE_READY",
        "version": project["version"],
        "mode": "public" if public else "source",
        "guide_links_checked": links,
        "verification_receipt_checked": evidence_checked,
        "errors": errors,
        "scope": "Static source/identity/evidence check; no jobs launched and no publishing performed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    result = check(args.root.resolve(), args.public)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))
