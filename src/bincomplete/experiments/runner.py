"""Resumable experiment orchestration around the public preparation/model APIs."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from ..artifacts import fingerprint, freeze, read, save, sha, verify_outputs
from ..data import Dataset, prepare_dataset
from ..engine import ModelArtifact, ProposalArtifact, fit, propose
from ..environment import environment_report
from ..inventory import candidate_table
from ..resources import FileLock
from .comparison import compare_proposals, write_report
from .config import load_plan


@dataclass(frozen=True)
class ExperimentArtifact:
    """Reference to experiment status, per-method artifacts and comparisons.

    Parameters
    ----------
    path : Path
        Experiment directory. Failed runs retain status and attempts for resumption.
    """

    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        """Read the latest summary and verify successful terminal output hashes."""
        marker = self.path / "complete.json"
        if marker.exists():
            verify_outputs(self.path, read(marker))
        summary = read(self.path / "summary.json")
        if marker.exists():
            _verify_completed(self.path, summary)
        return summary


def _persist(root: Path, state: dict[str, Any]) -> None:
    save(root / "status.json", state)


def _stage(
    root: Path,
    state: dict[str, Any],
    record: dict[str, Any],
    relative: Path,
    operation: Callable[[Path], Any],
) -> Any:
    """Reuse a completed stage or allocate a new attempt without deleting failures."""
    attempts = record.setdefault("attempts", [])
    previous = attempts[-1] if attempts else None
    destination = None
    if previous is not None:
        candidate = (root / previous["path"]).resolve()
        if candidate.parent != (root / relative).resolve():
            raise ValueError("Stage attempt escaped its experiment directory")
        if (candidate / "complete.json").exists():
            destination = candidate
        elif previous["status"] == "SUCCEEDED":
            raise ValueError("A previously completed stage is missing its receipt")
        else:
            previous["status"] = "PRESERVED_INCOMPLETE"
    if destination is None:
        destination = (root / relative / f"{len(attempts) + 1:04d}").resolve()
        if not destination.is_relative_to(root) or destination.exists():
            raise ValueError("New stage attempt must be an unused experiment path")
        previous = {"path": destination.relative_to(root).as_posix(), "status": "RUNNING"}
        attempts.append(previous)
    record.update(status="RUNNING", path=destination.relative_to(root).as_posix())
    assert previous is not None
    _persist(root, state)
    try:
        result = operation(destination)
    except BaseException as error:
        status = "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED"
        previous.update(status=status, error=f"{type(error).__name__}: {error}")
        record.update(status=status, error=previous["error"])
        log = root / "errors" / ("_".join(relative.parts) + f"_{len(attempts):04d}.log")
        log.parent.mkdir(exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        _persist(root, state)
        raise
    previous["status"] = "SUCCEEDED"
    record.update(status="SUCCEEDED", receipt_sha256=sha(destination / "complete.json"))
    record.pop("error", None)
    _persist(root, state)
    return result


def _verify_completed(root: Path, summary: dict[str, Any]) -> None:
    """Verify external inputs and every completed child, including nested outputs."""
    verified = set()
    for record in summary["datasets"].values():
        path = Path(record["path"])
        if path in verified:
            continue
        manifest = Dataset(path).verify()
        if (
            manifest["identity"] != record["identity"]
            or sha(path / "complete.json") != record["receipt_sha256"]
        ):
            raise ValueError("Completed experiment dataset identity changed")
        verified.add(path)
    for record in summary["methods"].values():
        for field, kind in (("model", ModelArtifact), ("proposals", ProposalArtifact)):
            path = Path(record[field])
            kind(path).manifest
            if sha(path / "complete.json") != record[field + "_receipt_sha256"]:
                raise ValueError("Completed experiment method receipt changed")


def run_experiment(
    config: str | Path,
    *,
    output: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ExperimentArtifact:
    """Prepare shared inputs and run explicitly selected methods sequentially.

    Parameters
    ----------
    config : path-like
        Strict JSON or TOML experiment configuration. It may reference original
        generic/mmlong2 inputs or existing prepared datasets, with distinct roles
        for training and evaluation.
    output : path-like, optional
        New output directory override. Existing experiment contracts must match.
    progress : callable, optional
        Receives small stage/method status dictionaries suitable for a CLI.

    Returns
    -------
    ExperimentArtifact
        Status, frozen resolved configuration, child artifacts and structural
        comparisons. Failed attempts are retained; verified completed stages are
        reused when this command is invoked again with the same configuration.

    Notes
    -----
    No scientific evaluator is invoked. Resuming an incomplete stage creates a
    fresh attempt for that stage, rather than pretending it has a valid result.
    """
    from .. import __version__

    plan = load_plan(config, output)
    document, limits = plan.document, plan.limits
    root = Path(document["output"])
    code_root = Path(__file__).resolve().parents[1]
    runtime = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}
    if any(row["method"] == "single" for row in document["methods"]):
        runtime["torch"] = version("torch")
    contract = {
        "configuration": document,
        "package_version": __version__,
        "runtime": runtime,
        "environment": environment_report("cpu"),
        "implementation": {
            p.relative_to(code_root).as_posix(): sha(p) for p in sorted(code_root.rglob("*.py"))
        },
    }
    identity = fingerprint(contract)
    with FileLock(root.with_name(root.name + ".experiment.lock"), blocking=False):
        marker = root / "experiment.json"
        if root.exists() and any(root.iterdir()) and not marker.exists():
            raise ValueError("Experiment output is not empty and has no owned experiment contract")
        freeze(marker, {"identity": identity, **contract})
        if (root / "complete.json").exists():
            receipt = read(root / "complete.json")
            if receipt["identity"] != identity:
                raise ValueError("Completed experiment has a different identity")
            verify_outputs(root, receipt)
            summary = read(root / "summary.json")
            _verify_completed(root, summary)
            if progress:
                progress({"phase": "complete", "status": "SUCCEEDED", "reused": True})
            return ExperimentArtifact(root)
        state: dict[str, Any] = (
            read(root / "status.json")
            if (root / "status.json").exists()
            else {
                "identity": identity,
                "datasets": {},
                "methods": {
                    row["id"]: {"method": row["method"], "status": "PENDING"} for row in document["methods"]
                },
            }
        )
        if state["identity"] != identity:
            raise ValueError("Experiment status belongs to a different contract")
        state["status"] = "RUNNING"
        _persist(root, state)
        try:
            datasets: dict[str, Dataset] = {}
            inputs: dict[str, str] = {}
            prepared: dict[str, Dataset] = {}
            assembly_indexes: dict[str, Dataset] = {}
            for name, spec in document["datasets"].items():
                key = fingerprint(spec)
                record = state["datasets"].setdefault(name, {})
                if key in prepared:
                    dataset = prepared[key]
                    record.update(status="SUCCEEDED", shared_with=inputs[key])
                elif "prepared" in spec:
                    dataset = Dataset(Path(spec["prepared"]))
                    dataset.verify()
                    record.update(status="SUCCEEDED", reference=spec["prepared"])
                else:
                    if progress:
                        progress({"phase": "preparing", "dataset": name})
                    preparation = dict(spec)
                    assembly_key = str(Path(spec["assembly"]).resolve())
                    if "assembly_index" not in preparation and assembly_key in assembly_indexes:
                        preparation["assembly_index"] = assembly_indexes[assembly_key].path
                    dataset = _stage(
                        root,
                        state,
                        record,
                        Path("datasets") / name,
                        partial(prepare_dataset, **preparation, limits=limits),
                    )
                metadata = dataset.verify()
                record.update(
                    dataset=str(dataset.path),
                    identity=metadata["identity"],
                    receipt_sha256=sha(dataset.path / "complete.json"),
                )
                assembly_indexes.setdefault(str(Path(metadata["assembly"]).resolve()), dataset)
                datasets[name] = dataset
                prepared[key], inputs[key] = dataset, name
                _persist(root, state)
            training, evaluation = datasets[document["training"]], datasets[document["evaluation"]]
            recipients = document["recipients"]
            if recipients is not None and not set(recipients) <= set(evaluation.manifest["bins"]):
                raise ValueError("Requested recipients are absent from the evaluation bins")
            for spec in document["methods"]:
                identifier = spec["id"]
                record = state["methods"][identifier]
                record["status"] = "RUNNING"
                record.pop("error", None)
                _persist(root, state)
                try:
                    if progress:
                        progress({"phase": "fitting", "method_id": identifier, "method": spec["method"]})
                    model = _stage(
                        root,
                        state,
                        record.setdefault("fit", {}),
                        Path("methods") / identifier / "fit",
                        partial(
                            fit,
                            training,
                            method=spec["method"],
                            settings=spec["settings"],
                            limits=limits,
                            cache=document["cache"],
                        ),
                    )
                    if progress:
                        progress({"phase": "proposing", "method_id": identifier})
                    proposals = _stage(
                        root,
                        state,
                        record.setdefault("propose", {}),
                        Path("methods") / identifier / "propose",
                        partial(
                            propose,
                            model,
                            evaluation,
                            recipients=recipients,
                            limits=limits,
                            cache=document["cache"],
                        ),
                    )
                    record.update(status="SUCCEEDED", model=str(model.path), proposals=str(proposals.path))
                except Exception as error:
                    record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                    _persist(root, state)
                    if progress:
                        progress({"phase": "failed", "method_id": identifier, "error": record["error"]})
                    if document["on_error"] == "stop":
                        break
                _persist(root, state)
            status = (
                "SUCCEEDED"
                if all(row["status"] == "SUCCEEDED" for row in state["methods"].values())
                else "FAILED"
            )
            summary = {
                "schema_version": 1,
                "identity": identity,
                "status": status,
                "datasets": {
                    name: {
                        "path": str(dataset.path),
                        "identity": dataset.manifest["identity"],
                        "receipt_sha256": sha(dataset.path / "complete.json"),
                    }
                    for name, dataset in datasets.items()
                },
                "methods": {
                    identifier: {
                        "method": row["method"],
                        "status": row["status"],
                        **(
                            {
                                "model": row["model"],
                                "proposals": row["proposals"],
                                "model_receipt_sha256": sha(Path(row["model"]) / "complete.json"),
                                "proposals_receipt_sha256": sha(Path(row["proposals"]) / "complete.json"),
                            }
                            if row["status"] == "SUCCEEDED"
                            else {"error": row.get("error", "Not attempted")}
                        ),
                    }
                    for identifier, row in state["methods"].items()
                },
                "quality_status": "NOT_EVALUATED_EXTERNAL_TOOLS_REQUIRED",
                "execution": "sequential",
            }
            completed = {
                identifier: ProposalArtifact(Path(row["proposals"]))
                for identifier, row in state["methods"].items()
                if row["status"] == "SUCCEEDED"
            }
            if status == "SUCCEEDED":
                inventory = candidate_table(list(completed.values()), root / "candidates.tsv", limits=limits)
                summary["assessment_priority"] = {
                    "path": str(inventory),
                    "receipt_sha256": sha(inventory.with_name(inventory.name + ".json")),
                    "all_proposals_retained": True,
                    "strategy": "size_balanced_mean",
                    "first_tier_limit": 40,
                }
            comparison = compare_proposals(completed, limits)
            save(root / "comparison.json", comparison)
            save(root / "summary.json", summary)
            write_report(root / "report.md", summary, comparison)
            state["status"] = status
            _persist(root, state)
            if status == "SUCCEEDED":
                _verify_completed(root, summary)
                files = [
                    root / name
                    for name in (
                        "experiment.json",
                        "status.json",
                        "summary.json",
                        "comparison.json",
                        "report.md",
                        "candidates.tsv",
                        "candidates.tsv.json",
                    )
                ]
                files.extend(Path(row["path"]) / "complete.json" for row in summary["datasets"].values())
                files.extend(
                    Path(row[field]) / "complete.json"
                    for row in summary["methods"].values()
                    for field in ("model", "proposals")
                )
                freeze(
                    root / "complete.json",
                    {
                        "kind": "experiment",
                        "identity": identity,
                        "outputs_sha256": {
                            p.relative_to(root).as_posix(): sha(p) for p in files if p.is_relative_to(root)
                        },
                    },
                )
            if progress:
                progress({"phase": "complete", "status": status})
            return ExperimentArtifact(root)
        except BaseException as error:
            state.update(
                status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
                error=f"{type(error).__name__}: {error}",
            )
            for record in state["methods"].values():
                if record["status"] == "RUNNING":
                    record.update(status=state["status"], error=state["error"])
            _persist(root, state)
            raise
