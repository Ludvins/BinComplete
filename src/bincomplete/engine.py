"""Train portable membership models and stream deterministic proposal artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import complete, fingerprint, read, save, sha, verify_outputs
from .context import Context
from .data import Dataset
from .methods import recipe
from .pipeline.examples import example_artifact
from .resources import FileLock, ResourceLimits
from .retrieval.queries import EmbeddingQueries


@dataclass(frozen=True)
class ModelArtifact:
    """Reference to a fitted model or zero-fit analytic configuration.

    Parameters
    ----------
    path : Path
        Completed model artifact directory.
    """

    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the portable model contract and dataset-specific provenance."""
        verify_outputs(self.path, read(self.path / "complete.json"))
        return read(self.path / "model.json")


@dataclass(frozen=True)
class ProposalArtifact:
    """Reference to full scores and membership proposals, without verdicts.

    Parameters
    ----------
    path : Path
        Completed proposal artifact directory.
    """

    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the verified proposal artifact manifest."""
        verify_outputs(self.path, read(self.path / "complete.json"))
        return read(self.path / "proposals.json")


def _new_or_same(root: Path, identity: str) -> bool:
    marker = root / "complete.json"
    if marker.exists():
        receipt = read(marker)
        verify_outputs(root, receipt)
        if receipt["identity"] != identity:
            raise ValueError("Output contains a different completed artifact")
        return False
    if root.exists() and any(root.iterdir()):
        raise ValueError("Incomplete output exists; preserve it and choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    return True


def fit(
    dataset: Dataset,
    output: str | Path,
    *,
    method: str,
    settings: dict[str, Any] | None = None,
    limits: ResourceLimits | None = None,
    cache: str | Path | None = None,
) -> ModelArtifact:
    """Fit one method using supplied original memberships, without quality labels.

    Parameters
    ----------
    dataset : Dataset
        Original multi-contig training bins and their representations.
    output : path-like
        Model output directory. Existing artifacts are reused only by exact identity.
    method : str
        Explicit method profile; ``cosine`` records zero optimizer calls.
    settings : dict, optional
        Explicit overrides for method parameters.
    limits : ResourceLimits, optional
        Training allocation, thread and cache limits.
    cache : path-like, optional
        Rebuildable representation cache directory.

    Returns
    -------
    ModelArtifact
        Model, complete method contract and immutable source provenance.
    """
    limits = limits or ResourceLimits()
    spec = recipe(method, limits, settings)
    method = spec["method"]
    root = Path(output).resolve()
    with FileLock(root.with_name(root.name + ".fit.lock")):
        context = Context(dataset, spec, root, cache, limits)
        try:
            compatibility = context.compatibility()
            runtime = {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "joblib")}
            if method == "single":
                from .numerical import torch_execution

                runtime["torch"] = version("torch")
                runtime["torch_execution"] = torch_execution(spec["scorer"]["params"]["device"])
            identity = fingerprint(
                {
                    "dataset": context.store.manifest["identity"],
                    "recipe": spec,
                    "compatibility": compatibility,
                    "runtime": runtime,
                    "implementation": context.code,
                }
            )
            if not _new_or_same(root, identity):
                return ModelArtifact(root)
            sources, groups = context.store.sources()
            if method != "cosine" and len({groups[r["genome_id"]] for r in sources}) < 2:
                raise ValueError("Training requires multi-contig sources from at least two unrelated groups")
            scorer = context.components["scorer"]
            stats = {"rows": 0, "episodes": 0}
            if method != "cosine":
                example_path, stats = example_artifact(context, sources, groups)
                scorer.fit(context, example_path, stats, root, sources, groups)
                # These matrices are rebuildable fit scratch, not the trained model.
                for name in ("features.npy", "labels.npy", "weights.npy"):
                    (root / name).unlink(missing_ok=True)
            joblib.dump(scorer, root / "scorer.partial.joblib", compress=0)
            (root / "scorer.partial.joblib").replace(root / "scorer.joblib")
            trained_rows = (
                np.unique(np.concatenate([context.store.indices(r["members"]) for r in sources]))
                if sources
                else np.array([], int)
            )
            source_hashes = (
                sorted(
                    {
                        bytes(value).decode("ascii")
                        for value in context.store.maps["sequence_hashes"][trained_rows]
                    }
                )
                if method != "cosine"
                else []
            )
            manifest = {
                "schema_version": 2,
                "method": method,
                "recipe": spec,
                "compatibility": compatibility,
                "runtime": runtime,
                "implementation": context.code,
                "training_dataset": str(dataset.path.resolve()),
                "training_identity": context.store.manifest["identity"],
                "training_sources": [r["genome_id"] for r in sources] if method != "cosine" else [],
                "training_groups": sorted({groups[r["genome_id"]] for r in sources})
                if method != "cosine"
                else [],
                "training_sequence_hashes": source_hashes,
                "training_groups_declared": context.store.manifest.get("groups_declared", False),
                "learned": method != "cosine",
                "optimizer_calls": 0 if method == "cosine" else None,
                "labels": "original within-bin membership",
                "training_statistics": stats,
                "scorer_sha256": sha(root / "scorer.joblib"),
            }
            save(root / "model.json", manifest)
            context.status("complete", method=method)
            complete(root, {"kind": "model", "identity": identity})
            return ModelArtifact(root)
        finally:
            context.close()


def _overlap(model: dict[str, Any], context: Context, recipients: list[str]) -> dict[str, Any]:
    train_hashes = set(model["training_sequence_hashes"])
    shared = {}
    for name in recipients:
        rows = context.store.indices(context.store.manifest["bins"][name])
        count = sum(
            bytes(value).decode("ascii") in train_hashes
            for value in context.store.maps["sequence_hashes"][rows]
        )
        if count:
            shared[name] = count
    shared_groups = (
        sorted(
            set(model["training_groups"]) & {context.store.manifest["groups"][name] for name in recipients}
        )
        if model.get("training_groups_declared") and context.store.manifest.get("groups_declared")
        else []
    )
    return {
        "mode": "transductive" if shared or shared_groups else "no_exact_sequence_overlap_detected",
        "recipient_overlap": shared,
        "overlapping_recipients": len(shared),
        "shared_declared_groups": shared_groups,
        "note": "Overlap is allowed and reported; absence of exact matches does not prove biological independence.",
    }


def propose(
    model: ModelArtifact,
    dataset: Dataset,
    output: str | Path,
    *,
    recipients: list[str] | None = None,
    limits: ResourceLimits | None = None,
    cache: str | Path | None = None,
) -> ProposalArtifact:
    """Rank donors and save membership proposals without invoking evaluation tools.

    Parameters
    ----------
    model : ModelArtifact
        Completed model; no fitting occurs during proposal generation.
    dataset : Dataset
        Recipient bins and candidate pool, possibly from another assembly.
    output : path-like
        New proposal artifact directory.
    recipients : list of str, optional
        Explicit recipient bin IDs; default is every supplied bin.
    limits : ResourceLimits, optional
        Current allocation and thread limits.
    cache : path-like, optional
        Rebuildable representation cache directory.

    Returns
    -------
    ProposalArtifact
        Full saved score vectors, ordered donors and finite addition proposals.
    """
    limits = limits or ResourceLimits()
    metadata = model.manifest
    spec = deepcopy(metadata["recipe"])
    spec["resources"].update(
        memory_gib=limits.memory_gib,
        threads=limits.threads,
        cache_gib=limits.cache_gib,
        chunk_size=limits.chunk_size,
    )
    spec["retriever"]["params"]["chunk_size"] = limits.chunk_size
    root = Path(output).resolve()
    with FileLock(root.with_name(root.name + ".propose.lock")):
        context = Context(dataset, spec, root, cache, limits)
        try:
            if context.compatibility() != metadata["compatibility"]:
                raise ValueError("Model feature formulas or encoder-space contract differ from this dataset")
            selected = sorted(context.store.manifest["bins"] if recipients is None else recipients)
            if len(selected) != len(set(selected)) or not set(selected) <= set(
                context.store.manifest["bins"]
            ):
                raise ValueError("Recipients must be unique supplied bin IDs")
            identity = fingerprint(
                {
                    "model": sha(model.path / "model.json"),
                    "dataset": context.store.manifest["identity"],
                    "recipients": selected,
                    "implementation": context.code,
                }
            )
            if not _new_or_same(root, identity):
                return ProposalArtifact(root)
            scorer = joblib.load(model.path / "scorer.joblib")
            pool = context.store.maps["pool"]
            overlap = _overlap(metadata, context, selected)
            counts = {}
            proposals = 0
            with (root / "memberships.partial.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for number, name in enumerate(selected):
                    core = context.store.indices(context.store.manifest["bins"][name])
                    limits.require(
                        len(core) * 136 * 64 + limits.chunk_size * 136 * 128, "Feature and scoring buffers"
                    )
                    queries = None
                    if spec["method"] == "single":
                        import torch

                        old_threads = torch.get_num_threads()
                        torch.set_num_threads(min(2, limits.threads))
                        try:
                            limits.require(
                                len(core) ** 2 * scorer.heads * 16 + len(core) * scorer.width * 64,
                                "Single-query attention",
                            )
                            queries = scorer.predict_queries(core, context)
                        finally:
                            torch.set_num_threads(old_threads)
                        retriever = EmbeddingQueries(
                            **{
                                k: v
                                for k, v in spec["retriever"]["params"].items()
                                if k not in {"metric", "summary", "weights"}
                            }
                        )
                        candidates = retriever.retrieve_queries(
                            queries,
                            core,
                            pool,
                            context.representation("embedding"),
                            context.store,
                            context.representation_id("embedding"),
                        ).indices
                    elif spec["pool_policy"] == "full":
                        candidates = context.components["retriever"].eligible(core, pool, context.store)
                        if spec["method"] == "cosine":
                            queries = scorer.predict_queries(core, context)
                    else:
                        candidates = context.retrieve(core, pool).indices
                    folder = root / "rankings" / fingerprint(name)
                    folder.mkdir(parents=True, exist_ok=True)
                    np.save(folder / "rows.npy", candidates)
                    scores = np.lib.format.open_memmap(
                        folder / "scores.npy", mode="w+", dtype="float64", shape=(len(candidates),)
                    )
                    best, best_scores = np.array([], np.int64), np.array([], float)
                    try:
                        with threadpool_limits(limits=limits.threads):
                            for start in range(0, len(candidates), limits.chunk_size):
                                block = candidates[start : start + limits.chunk_size]
                                if queries is not None:
                                    q = np.asarray(queries.values)
                                    x = np.asarray(context.representation("embedding")[block], np.float32)
                                    if spec["method"] == "cosine":
                                        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
                                        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
                                        values = (x @ q.T).max(1)
                                    else:
                                        old_threads = torch.get_num_threads()
                                        torch.set_num_threads(min(2, limits.threads))
                                        try:
                                            with torch.inference_mode():
                                                tensor = torch.as_tensor(x, device=scorer.device)
                                                query = torch.as_tensor(q, device=scorer.device)
                                                values = (tensor @ query.T).amax(1).cpu().numpy()
                                        finally:
                                            torch.set_num_threads(old_threads)
                                else:
                                    values = np.asarray(scorer.score(core, block, context))
                                if values.shape != (len(block),) or not np.isfinite(values).all():
                                    raise ValueError("Scorer violated its finite-vector contract")
                                scores[start : start + len(block)] = values
                                joined, costs = np.r_[best, block], np.r_[best_scores, values]
                                retained = max(20, context.components["search"].fallback_singletons)
                                limits.require(
                                    retained * 128 + context.reserved_bytes, "Retained ranking buffers"
                                )
                                order = np.lexsort((context.store.ids[joined], -costs))[:retained]
                                best, best_scores = joined[order], costs[order]
                        scores.flush()
                    finally:
                        scores._mmap.close()
                    emitted = 0
                    for proposal in context.components["search"].propose(core, best, context.store):
                        members = sorted(str(context.store.ids[i]) for i in proposal.pop("indices"))
                        proposal["added"] = sorted(str(context.store.ids[i]) for i in proposal["added"])
                        proposal.update(
                            recipient=name,
                            members=members,
                            method=spec["method"],
                            dataset_identity=context.store.manifest["identity"],
                        )
                        proposal["proposal_id"] = fingerprint(
                            {"artifact": identity, "recipient": name, "members": members}
                        )
                        stream.write(json.dumps(proposal, sort_keys=True) + "\n")
                        proposals += 1
                        emitted += 1
                    counts[name] = {
                        "scored": len(candidates),
                        "proposals": emitted,
                        "ranking": folder.relative_to(root).as_posix(),
                    }
                    context.status(
                        "proposing", completed_recipients=number + 1, total_recipients=len(selected)
                    )
            (root / "memberships.partial.jsonl").replace(root / "memberships.jsonl")
            save(
                root / "proposals.json",
                {
                    "schema_version": 2,
                    "dataset": str(dataset.path.resolve()),
                    "dataset_identity": context.store.manifest["identity"],
                    "model": str(model.path.resolve()),
                    "model_sha256": sha(model.path / "model.json"),
                    "method": spec["method"],
                    "search": spec["search"],
                    "coverage": counts,
                    "proposal_count": proposals,
                    "overlap": overlap,
                    "implementation": context.code,
                    "verdicts": "external tools only",
                },
            )
            context.status("complete", proposal_count=proposals)
            complete(root, {"kind": "proposals", "identity": identity})
            return ProposalArtifact(root)
        finally:
            context.close()


def run(
    training: Dataset,
    recipients: Dataset,
    output: str | Path,
    *,
    method: str,
    settings: dict[str, Any] | None = None,
    limits: ResourceLimits | None = None,
    cache: str | Path | None = None,
) -> ProposalArtifact:
    """Fit and propose one explicit method, releasing training buffers before inference.

    Parameters
    ----------
    training, recipients : Dataset
        Training and recipient datasets; overlap is permitted and disclosed.
    output : path-like
        Parent for the model and proposal artifacts.
    method : str
        Explicit method name.
    settings : dict, optional
        Explicit training and proposal profile settings.
    limits : ResourceLimits, optional
        Recruitment resource limits.
    cache : path-like, optional
        Derived representation cache directory.

    Returns
    -------
    ProposalArtifact
        Membership proposals ready for separate external evaluation.
    """
    root = Path(output)
    fitted = fit(training, root / "model", method=method, settings=settings, limits=limits, cache=cache)
    return propose(fitted, recipients, root / "proposals", limits=limits, cache=cache)
