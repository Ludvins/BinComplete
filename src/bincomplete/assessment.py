"""Resumable quality assessment of a completed BinComplete experiment."""

from __future__ import annotations

import signal
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import fingerprint, freeze, read, save, sha
from .assessment_evidence import evaluate_batch, read_batch
from .assessment_inputs import Catalog
from .assessment_selection import Assessment, write_baseline, write_reports, write_summary
from .assessment_tools import Evaluator, Paused
from .resources import FileLock, ResourceLimits


def _chunks(rows: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@contextmanager
def _pause_signal() -> Iterator[tuple[threading.Event, threading.Event]]:
    paused = threading.Event()
    terminating = threading.Event()
    signals = {}
    if threading.current_thread() is threading.main_thread():

        def on_signal(signum: int, frame: Any) -> None:
            if signum == signal.SIGTERM:
                terminating.set()
            paused.set()

        for name in ("SIGUSR1", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signals[sig] = signal.signal(sig, on_signal)
    try:
        yield paused, terminating
    finally:
        for sig, handler in signals.items():
            signal.signal(sig, handler)


def _make_import(source: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    with FileLock(source.with_name(source.name + ".assess.lock"), blocking=False):
        contract = read(source / "contract.json")
        if contract["mode"] != "greedy" or contract["evidence"] != evidence:
            raise ValueError(
                "Reuse requires a greedy assessment with the same experiment and evaluator identity"
            )
        baseline_marker = source / "baseline" / "complete.json"
        if not baseline_marker.is_file():
            raise ValueError("The source assessment baseline is incomplete")
        baselines = {f"baseline/{name}": digest for name, digest in read(baseline_marker).items()}
        candidates = sorted((source / "batches").glob("round-*/batch-*/complete.json"))
        return {
            "source": str(source),
            "contract_sha256": sha(source / "contract.json"),
            "baseline_sha256": sha(baseline_marker),
            "batches": baselines
            | {path.parent.relative_to(source).as_posix(): sha(path) for path in candidates},
        }


def _read_import(
    root: Path,
    source: Path | None,
    evidence: dict[str, Any],
    catalog: Catalog,
    identity: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any], str]]]:
    if source is None:
        return {}, []
    marker = root / "import.json"
    if not marker.exists():
        freeze(marker, _make_import(source, evidence))
    snapshot = read(marker)
    if (
        snapshot["source"] != str(source)
        or sha(source / "contract.json") != snapshot["contract_sha256"]
        or sha(source / "baseline" / "complete.json") != snapshot["baseline_sha256"]
    ):
        raise ValueError("Reused assessment source identity changed")
    baseline: dict[str, Any] = {}
    candidates = []
    for relative, digest in snapshot["batches"].items():
        folder = (source / relative).resolve()
        if not folder.is_relative_to(source) or sha(folder / "complete.json") != digest:
            raise ValueError(f"Reused assessment batch changed: {folder}")
        original_batch = relative.startswith("baseline/")
        entries = read(folder / "plan.json")["entries"]
        keys = [row["construction_id"] for name, row in entries.items() if name.startswith("c")]
        originals = [row["recipient"] for row in entries.values()] if original_batch else []
        expected = catalog.batch(keys, originals)
        manifest, results = read_batch(folder, expected, identity, baseline=original_batch)
        if original_batch:
            _add_baseline(baseline, manifest, results)
        else:
            candidates.append((manifest, results, str(folder)))
    if baseline.keys() != catalog.originals.keys():
        raise ValueError("Reused baseline does not cover the exact original panel")
    return baseline, candidates


def _add_baseline(baseline: dict[str, Any], manifest: dict[str, Any], results: dict[str, Any]) -> None:
    for name, entry in manifest.items():
        recipient = entry["recipient"]
        if recipient in baseline:
            raise ValueError(f"Duplicate baseline bin: {recipient}")
        baseline[recipient] = results[name] | {"sha256": entry["sha256"]}


def _execute(
    root: Path,
    catalog: Catalog,
    evaluator: Evaluator,
    limits: ResourceLimits,
    mode: str,
    max_rounds: int | None,
    source: Path | None,
    evidence: dict[str, Any],
    paused: threading.Event,
) -> Path:
    identity = fingerprint(evidence)
    summary_path = root / "summary.json"
    prior = read(summary_path) if summary_path.is_file() else {}
    previous_selection = prior.get("last_published_selection") or prior.get("selection")
    baseline, imports = _read_import(root, source, evidence, catalog, identity)
    rounds = 0
    batches = 0
    state = None
    active_batch: Path | None = None

    def tool_progress(tool: str, log: Path) -> None:
        save(
            root / "activity.json",
            {
                "phase": "baseline" if state is None else "candidates",
                "batch": active_batch.relative_to(root).as_posix() if active_batch else None,
                "tool": tool,
                "log": log.relative_to(root).as_posix(),
            },
        )

    evaluator.progress = tool_progress

    def baseline_progress(status: str) -> None:
        save(
            summary_path,
            {
                "state": status,
                "phase": "baseline",
                "mode": mode,
                "baseline_bins_checked": len(baseline),
                "original_bins": len(catalog.originals),
                "last_published_selection": previous_selection,
            },
        )

    try:
        if source is None:
            folders = []
            for index, names in enumerate(_chunks(sorted(catalog.originals), evaluator.batch_size), 1):
                folder = root / "baseline" / f"batch-{index:06d}"
                active_batch = folder
                if not (folder / "complete.json").is_file():
                    baseline_progress("IN_PROGRESS")
                manifest, results = evaluate_batch(
                    folder,
                    catalog.batch([], names),
                    identity,
                    catalog,
                    evaluator,
                    limits,
                    paused.is_set,
                    baseline=True,
                )
                _add_baseline(baseline, manifest, results)
                folders.append(folder)
            freeze(
                root / "baseline" / "complete.json",
                {folder.name: sha(folder / "complete.json") for folder in folders},
            )
        write_baseline(root, baseline)
        state = Assessment(catalog, baseline)
        for manifest, results, context in imports:
            state.add(manifest, results, context, True)
        batches = len(imports)

        def check_batch(folder: Path, keys: list[str]) -> None:
            nonlocal batches, active_batch
            assert state is not None
            active_batch = folder
            manifest, results = evaluate_batch(
                folder,
                catalog.batch(keys, []),
                identity,
                catalog,
                evaluator,
                limits,
                paused.is_set,
                baseline=False,
            )
            state.add(manifest, results, folder.relative_to(root).as_posix(), False)
            batches += 1

        def progress() -> None:
            assert state is not None
            write_summary(
                root, state, mode, "IN_PROGRESS", rounds, batches, last_selection=previous_selection
            )

        # Replay completed receipts to reconstruct state before publishing progress.
        # Rewriting the entire catalog for each replayed batch is quadratic on resume.
        replaying = True

        def check_and_publish(folder: Path, keys: list[str]) -> None:
            nonlocal replaying
            if replaying and not (folder / "complete.json").is_file():
                progress()
                replaying = False
            check_batch(folder, keys)

        reason = "catalog_exhausted"
        if max_rounds is not None:
            while keys := state.next_round():
                if rounds == max_rounds:
                    reason = "round_limit"
                    break
                for index, chunk in enumerate(_chunks(keys, evaluator.batch_size), 1):
                    folder = root / "batches" / f"round-{rounds + 1:06d}" / f"batch-{index:06d}"
                    check_and_publish(folder, chunk)
                    if not replaying and index * evaluator.batch_size < len(keys):
                        progress()
                state.accept(keys)
                rounds += 1
                if not replaying:
                    progress()
        else:
            for index, keys in enumerate(_chunks(state.full_order(), evaluator.batch_size), 1):
                check_and_publish(root / "batches" / f"batch-{index:06d}", keys)
                if not replaying:
                    progress()
        if paused.is_set():
            raise Paused
        selection = state.publish_selection(root, mode, evaluator.solver_seconds, limits)
        write_reports(root, state, mode)
        status = "STOPPED" if reason == "round_limit" else "COMPLETE"
        write_summary(root, state, mode, status, rounds, batches, selection, reason, last_selection=selection)
        save(root / "activity.json", {"phase": "finished", "state": status})
    except (Paused, KeyboardInterrupt):
        if state is None:
            baseline_progress("PAUSED")
        else:
            write_reports(root, state, mode)
            write_summary(
                root,
                state,
                mode,
                "PAUSED",
                rounds,
                batches,
                reason="interrupted",
                last_selection=previous_selection,
            )
        save(root / "activity.json", {"phase": "paused"})
    return root / "summary.json"


def assess(
    experiment: Path,
    config: Path,
    output: Path,
    *,
    mode: str,
    max_rounds: int | None = None,
    reuse_from: Path | None = None,
    memory_gib: float = 4.0,
) -> Path:
    """Run or resume an opt-in CheckM2 and RNA assessment."""
    if mode not in {"greedy", "full"}:
        raise ValueError("Assessment mode must be greedy or full")
    if mode == "greedy" and (type(max_rounds) is not int or max_rounds < 1):
        raise ValueError("Greedy assessment requires a positive --max-rounds")
    if (mode == "greedy" and reuse_from is not None) or (mode == "full" and max_rounds is not None):
        raise ValueError("Use --reuse-from with full mode and --max-rounds with greedy mode")
    experiment, output = experiment.resolve(), output.resolve()
    source = reuse_from.resolve() if reuse_from is not None else None
    for original in (experiment, source):
        if original is not None and (output.is_relative_to(original) or original.is_relative_to(output)):
            raise ValueError("Use an assessment output directory separate from all inputs")
    if source is not None and not (source / "contract.json").is_file():
        raise ValueError("Reuse requires a saved bincomplete assess directory")
    evaluator = Evaluator.load(config)
    evaluator.validate_paths(output)
    limits = ResourceLimits(memory_gib=memory_gib)
    catalog = Catalog.load(experiment, limits)
    for path in (catalog.dataset.path, *(item.path for item in catalog.artifacts)):
        path = path.resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Use an assessment output directory separate from completed artifacts")
    with (
        FileLock(output.with_name(output.name + ".assess.lock"), blocking=False) as lock,
        _pause_signal() as (paused, terminating),
    ):
        evaluator.paused, evaluator.terminating = paused.is_set, terminating.is_set
        evaluator.lock_fd = lock.stream.fileno()
        contract_path = output / "contract.json"
        if output.exists() and any(output.iterdir()) and not contract_path.is_file():
            raise ValueError("Assessment output exists without a BinComplete assessment contract")
        output.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="probe-", dir=output) as temp:
                tool_identity = evaluator.identity(Path(temp))
        except (Paused, KeyboardInterrupt):
            summary_path = output / "summary.json"
            if contract_path.exists():
                previous = read(summary_path) if summary_path.is_file() else {}
                save(summary_path, previous | {"state": "PAUSED", "phase": "preflight"})
            raise Paused from None
        evidence = {
            "input": catalog.identity,
            "evaluator": tool_identity,
            "implementation": {
                file.name: sha(file) for file in sorted(Path(__file__).parent.glob("assess*.py"))
            }
            | {
                name: sha(Path(__file__).with_name(name))
                for name in ("export.py", "reconcile.py", "sequence_io.py")
            },
        }
        contract = {
            "schema_version": 1,
            "evidence": evidence,
            "mode": mode,
            "batch_size": evaluator.batch_size,
            "reuse_from": str(source) if source else None,
            "bincomplete_version": __version__,
        }
        freeze(contract_path, contract)
        if max_rounds is not None:
            started = [
                int(path.name.removeprefix("round-"))
                for path in (output / "batches").glob("round-*")
                if path.name.removeprefix("round-").isdigit()
            ]
            if started and max_rounds < max(started):
                raise ValueError("--max-rounds must cover rounds already started")
        try:
            return _execute(output, catalog, evaluator, limits, mode, max_rounds, source, evidence, paused)
        except Exception as error:
            summary = output / "summary.json"
            previous = read(summary) if summary.is_file() else {}
            save(summary, previous | {"state": "FAILED", "mode": mode, "error": str(error)})
            activity = output / "activity.json"
            current = read(activity) if activity.is_file() else {}
            save(activity, current | {"state": "FAILED"})
            raise
