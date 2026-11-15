"""Durable, verified CheckM2 and RNA checkpoints for an assessment batch."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifacts import fingerprint, freeze, read, sha, verify_outputs
from .assessment_inputs import Catalog
from .assessment_tools import Evaluator, Paused, numeric_pass, read_checkm2_attempt, read_rna
from .resources import ResourceLimits

CHECKM2_FILES = ("results/checkm2.log", "stdout.log", "stderr.log")
RNA_FILES = tuple(
    f"{domain}.{suffix}"
    for domain in ("bac", "arc")
    for suffix in ("rrna.gff", "trna.tsv", "barrnap.log", "trnascan.stdout.log", "trnascan.stderr.log")
)


def _attempt(folder: Path, stage: str) -> Path:
    return Path(tempfile.mkdtemp(dir=folder, prefix=f"{stage}-"))


def _checkpoint(
    receipt: Path,
    identity: str,
    files: tuple[str, ...] | Callable[[Path], tuple[str, ...]],
    evaluate: Callable[[Path], Any] | None = None,
) -> Path:
    if not receipt.exists():
        if evaluate is None:
            raise ValueError(f"Assessment checkpoint missing: {receipt}")
        attempt = _attempt(receipt.parent, receipt.stem)
        evaluate(attempt)
        temp = attempt / "tmp"
        if temp.is_dir() and not temp.is_symlink():
            shutil.rmtree(temp)
        expected = files(attempt) if callable(files) else files
        freeze(
            receipt,
            {
                "identity": identity,
                "attempt": attempt.name,
                "outputs_sha256": {file: sha(attempt / file) for file in expected},
            },
        )
    saved = read(receipt)
    attempt = (receipt.parent / saved["attempt"]).resolve()
    if (
        saved["identity"] != identity
        or attempt.parent != receipt.parent.resolve()
        or not attempt.name.startswith(f"{receipt.stem}-")
    ):
        raise ValueError(f"Assessment checkpoint identity differs: {receipt}")
    expected = files(attempt) if callable(files) else files
    if set(saved["outputs_sha256"]) != set(expected):
        raise ValueError(f"Assessment checkpoint files differ: {receipt}")
    verify_outputs(attempt, saved)
    return attempt


def _checkm2_files(attempt: Path) -> tuple[str, ...]:
    report = "results/quality_report.tsv"
    if (attempt / report).exists():
        return (*CHECKM2_FILES, report)
    proteins = sorted((attempt / "results/protein_files").glob("*.faa"))
    return (*CHECKM2_FILES, *(path.relative_to(attempt).as_posix() for path in proteins))


def _matches(entries: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if entries.keys() != manifest.keys():
        return False
    return all(
        manifest[name] == row | {"fasta": f"{name}.fa", "sha256": manifest[name]["sha256"]}
        for name, row in entries.items()
    )


def _load_results(
    folder: Path,
    identity: str,
    manifest: dict[str, Any],
    baseline: bool,
    evaluator: Evaluator | None = None,
    inputs: Path | None = None,
    paused: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    def verify_input(name: str) -> None:
        assert inputs is not None
        if sha(inputs / manifest[name]["fasta"]) != manifest[name]["sha256"]:
            raise ValueError(f"Assessment input FASTA changed during execution: {name}")

    def run_checkm2(attempt: Path) -> None:
        assert evaluator is not None and inputs is not None
        for name in manifest:
            verify_input(name)
        evaluator.checkm2(inputs, attempt, set(manifest))
        for name in manifest:
            verify_input(name)

    numeric = _checkpoint(
        folder / "checkm2.json",
        fingerprint({"batch": identity, "fasta_manifest": manifest}),
        _checkm2_files,
        run_checkm2 if evaluator else None,
    )
    results = read_checkm2_attempt(numeric, set(manifest))
    for name, result in results.items():
        if paused():
            raise Paused
        result["rna"] = None
        if baseline or (name.startswith("c") and numeric_pass(result)):
            fasta_hash = manifest[name]["sha256"]

            def run_rna(attempt: Path, name: str = name) -> None:
                assert evaluator is not None and inputs is not None
                verify_input(name)
                evaluator.rna(inputs / f"{name}.fa", attempt)
                verify_input(name)

            rna = _checkpoint(
                folder / f"{name}.rna.json",
                fingerprint({"batch": identity, "fasta_sha256": fasta_hash}),
                RNA_FILES,
                run_rna if evaluator else None,
            )
            result["rna"] = read_rna(rna)
    return results


def _finished(folder: Path, identity: str, results: dict[str, Any]) -> dict[str, Any]:
    files = ["plan.json", "manifest.json", "checkm2.json"]
    files += [f"{name}.rna.json" for name, row in results.items() if row["rna"] is not None]
    return {"identity": identity, "receipts_sha256": {file: sha(folder / file) for file in files}}


def read_batch(
    folder: Path,
    entries: dict[str, Any],
    evidence_identity: str,
    *,
    baseline: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = {"evidence_identity": evidence_identity, "baseline": baseline, "entries": entries}
    if read(folder / "plan.json") != plan:
        raise ValueError(f"Assessment batch plan changed: {folder}")
    identity = fingerprint(plan)
    manifest = read(folder / "manifest.json")
    if not _matches(entries, manifest):
        raise ValueError(f"Assessment FASTA manifest changed: {folder}")
    results = _load_results(folder, identity, manifest, baseline)
    if read(folder / "complete.json") != _finished(folder, identity, results):
        raise ValueError(f"Assessment batch receipt changed: {folder}")
    return manifest, results


def evaluate_batch(
    folder: Path,
    entries: dict[str, Any],
    evidence_identity: str,
    catalog: Catalog,
    evaluator: Evaluator,
    limits: ResourceLimits,
    paused: Callable[[], bool],
    *,
    baseline: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if paused():
        raise Paused
    if (folder / "complete.json").is_file():
        return read_batch(folder, entries, evidence_identity, baseline=baseline)
    plan = {"evidence_identity": evidence_identity, "baseline": baseline, "entries": entries}
    freeze(folder / "plan.json", plan)
    # The caller holds the assessment lock, inherited by its POSIX tool processes.
    owned = folder / "fasta-work"
    if owned.is_symlink() or (owned.exists() and not owned.is_dir()):
        raise ValueError(f"Unexpected assessment scratch path: {owned}")
    if os.name == "posix" and owned.is_dir():
        marker = owned / ".owner.json"
        if any(owned.iterdir()) and (
            not marker.is_file() or marker.is_symlink() or read(marker) != {"plan": fingerprint(plan)}
        ):
            raise ValueError(f"Cannot reclaim unrecognized assessment scratch: {owned}")
        shutil.rmtree(owned)
    owned.mkdir(exist_ok=True)
    freeze(owned / ".owner.json", {"plan": fingerprint(plan)})
    try:
        with tempfile.TemporaryDirectory(dir=owned) as temp:
            inputs, manifest = catalog.materialize(entries, Path(temp), limits)
            freeze(folder / "manifest.json", manifest)
            if paused():
                raise Paused
            results = _load_results(folder, fingerprint(plan), manifest, baseline, evaluator, inputs, paused)
    finally:
        if list(owned.iterdir()) == [owned / ".owner.json"]:
            (owned / ".owner.json").unlink()
            owned.rmdir()
    freeze(folder / "complete.json", _finished(folder, fingerprint(plan), results))
    return manifest, results
