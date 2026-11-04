"""Release gates, source-distribution parity and packaged profile authority."""

import hashlib
import io
import json
import runpy
import shutil
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = runpy.run_path(str(ROOT / "tools/check_release.py"))
DISTRIBUTION = runpy.run_path(str(ROOT / "tools/check_distribution.py"))


@pytest.fixture
def release_tree(tmp_path):
    root = tmp_path / "checkout"
    shutil.copytree(
        ROOT,
        root,
        ignore=shutil.ignore_patterns(
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
            "*.egg-info",
        ),
    )
    # Tests below isolate the gate from unrelated documentation presentation.
    (root / "README.md").write_text("# BinComplete\nResearch beta.\n", encoding="utf-8")
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    paths = [path for path in root.rglob("*") if path.is_file() and path.name != "release_verification.json"]
    receipt = {
        "status": "VERIFIED",
        "version": version,
        "checks": {name: "PASSED" for name in RELEASE["REQUIRED_RELEASE_CHECKS"]},
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    (root / "release_verification.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root


def test_public_release_requires_all_declared_checks(release_tree):
    assert RELEASE["check"](release_tree, public=True)["status"] == "PUBLIC_READY"
    path = release_tree / "release_verification.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for key in (*RELEASE["REQUIRED_RELEASE_CHECKS"], "additional_validation"):
        changed = receipt | {"checks": receipt["checks"] | {key: "FAILED"}}
        path.write_text(json.dumps(changed), encoding="utf-8")
        result = RELEASE["check"](release_tree, public=True)
        assert result["status"] == "FAILED"
        assert any(key in error and "release check" in error for error in result["errors"])


def test_public_release_rejects_missing_or_malformed_checks(release_tree):
    path = release_tree / "release_verification.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for value in ({}, [], None):
        path.write_text(json.dumps(receipt | {"checks": value}), encoding="utf-8")
        assert RELEASE["check"](release_tree, public=True)["status"] == "FAILED"


def test_source_check_needs_no_checkout_signoff(release_tree):
    (release_tree / "release_verification.json").unlink()
    result = RELEASE["check"](release_tree)
    assert result["status"] == "SOURCE_READY"
    assert not result["verification_receipt_checked"]
    assert RELEASE["check"](release_tree, public=True)["status"] == "FAILED"


def test_generated_documentation_does_not_change_source_signoff(release_tree):
    for folder in ("site", ".cache", ".venv", ".venv-docs"):
        generated = release_tree / folder / "generated.html"
        generated.parent.mkdir()
        generated.write_text("Generated content", encoding="utf-8")
    assert RELEASE["check"](release_tree, public=True)["status"] == "PUBLIC_READY"


@pytest.mark.parametrize("name", ["CITATION.cff", "src/bincomplete/__init__.py"])
def test_source_check_rejects_mismatched_versions(release_tree, name):
    path = release_tree / name
    text = "version: 0.0.0\n" if name.endswith(".cff") else '__version__ = "0.0.0"\n'
    path.write_text(text, encoding="utf-8")
    assert any("version" in error for error in RELEASE["check"](release_tree)["errors"])


def test_readme_does_not_duplicate_release_version(release_tree):
    (release_tree / "README.md").write_text("Research beta, version 0.0.0.\n", encoding="utf-8")
    assert any("README" in error for error in RELEASE["check"](release_tree)["errors"])


def _archive(root, output, *, omitted=(), changed=None, extra=None):
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    prefix = f"{project['name']}-{project['version']}"
    contents = {
        name: path.read_bytes()
        for name, path in DISTRIBUTION["supported_files"](root).items()
        if name not in omitted
    }
    contents.update(changed or {})
    contents.update(extra or {})
    with tarfile.open(output, "w:gz") as archive:
        for name, payload in contents.items():
            member = tarfile.TarInfo(f"{prefix}/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_distribution_requires_checkout_helper_and_byte_identical_content(release_tree, tmp_path):
    archive = tmp_path / "source.tar.gz"
    _archive(release_tree, archive)
    assert DISTRIBUTION["check"](release_tree, archive)["status"] == "SOURCE_DISTRIBUTION_READY"
    for updates, expected in (
        ({"omitted": ("run.py",)}, "Missing source-distribution file: run.py"),
        ({"omitted": ("zensical.toml",)}, "Missing source-distribution file: zensical.toml"),
        (
            {"omitted": ("requirements-docs.txt",)},
            "Missing source-distribution file: requirements-docs.txt",
        ),
        (
            {"omitted": ("docs/javascripts/mathjax.js",)},
            "Missing source-distribution file: docs/javascripts/mathjax.js",
        ),
        (
            {"omitted": ("docs/stylesheets/extra.css",)},
            "Missing source-distribution file: docs/stylesheets/extra.css",
        ),
        (
            {"omitted": ("docs/assets/logo.svg",)},
            "Missing source-distribution file: docs/assets/logo.svg",
        ),
        ({"changed": {"run.py": b"changed"}}, "Source-distribution content differs: run.py"),
        ({"extra": {"release_verification.json": b"{}"}}, "Checkout-only"),
        ({"extra": {"AGENTS.md": b"local instructions"}}, "Checkout-only"),
        ({"extra": {"../outside.py": b"invalid"}}, "Unexpected archive path"),
    ):
        _archive(release_tree, archive, **updates)
        result = DISTRIBUTION["check"](release_tree, archive)
        assert result["status"] == "FAILED"
        assert any(expected in error for error in result["errors"])


def test_supported_examples_match_packaged_standard_profiles():
    examples = {path.name: path for path in (ROOT / "configs/fresh").glob("*.json")}
    profiles = {
        path.name: path
        for path in (ROOT / "src/bincomplete/profiles/standard_v1").glob("*.json")
        if path.name != "tested_environment.json"
    }
    assert examples and set(examples) == set(profiles)
    for name, path in examples.items():
        assert path.read_bytes() == profiles[name].read_bytes(), f"Standard profile example drifted: {name}"
