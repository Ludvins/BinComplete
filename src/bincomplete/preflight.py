"""Read-only input/readiness checks; no preparation, encoder or training jobs."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read
from .data import Dataset, _composition, _coverage, _membership_rows, bin_fasta_paths, fasta_records
from .defaults import uses_embedding
from .experiments.config import load_plan
from .resources import ResourceLimits


def _mapping(path: str, names: set[str], kind: str, *, unique: bool = False) -> dict[str, str]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        pairs = [line.rstrip("\n").split("\t") for line in stream if line.strip()]
    headers = (["bin_id", f"{kind}_id"], ["bin", kind])
    if pairs and pairs[0] in headers:
        pairs = pairs[1:]
    if any(len(row) != 2 for row in pairs):
        raise ValueError(f"{kind} table needs two tab-separated columns")
    result = dict(pairs)
    if (
        len(result) != len(pairs)
        or set(result) != names
        or not all(result.values())
        or (unique and len(set(result.values())) != len(result))
    ):
        raise ValueError(f"{kind} table must map every bin exactly once to a nonempty ID")
    return result


def _supplied_vectors(spec: dict, ids: list[str], limits: ResourceLimits) -> dict | None:
    keys = ("embeddings", "embedding_ids", "embedding_space")
    if not any(key in spec for key in keys):
        return None
    if not all(key in spec for key in keys):
        raise ValueError("Embeddings require the array, row IDs and embedding-space manifest together")
    matrix = np.load(spec["embeddings"], mmap_mode="r", allow_pickle=False)
    try:
        rows = Path(spec["embedding_ids"]).read_text(encoding="utf-8").splitlines()
        limits.require(len(rows) * 512 + len(ids) * 1024, "Preflight embedding metadata")
        if (
            matrix.ndim != 2
            or matrix.dtype not in (np.dtype("float16"), np.dtype("float32"))
            or len(rows) != len(matrix)
            or len(set(rows)) != len(rows)
        ):
            raise ValueError("Embedding array must be float16/float32 with one unique ID per row")
        if not set(ids) <= set(rows):
            raise ValueError("Embedding population omits assembly contigs")
        space = read(spec["embedding_space"])
        required = {
            "encoder",
            "encoder_sha256",
            "tokenizer_sha256",
            "pooling",
            "window_bp",
            "windows",
            "max_tokens",
            "normalization",
            "dimension",
        }
        if (
            not required <= set(space)
            or space["dimension"] != matrix.shape[1]
            or space["normalization"] != "l2"
        ):
            raise ValueError("Supply the complete encoder-space contract with L2 normalization")
        width = min(limits.chunk_size, max(1, limits.memory_bytes // max(1, matrix.shape[1] * 32)))
        tolerance = 5e-4 if matrix.dtype == np.float16 else 1e-4
        for start in range(0, len(rows), width):
            block = np.asarray(matrix[start : start + width], np.float32)
            if not np.isfinite(block).all() or not np.allclose(
                np.linalg.norm(block, axis=1), 1, atol=tolerance, rtol=0
            ):
                raise ValueError("Embedding rows must be finite and already L2 normalized")
        return {"provider": "supplied", "dimension": matrix.shape[1], "space": space}
    finally:
        matrix._mmap.close()


def _raw(spec: dict[str, Any], limits: ResourceLimits) -> dict[str, Any]:
    ids: list[str] = []
    hashes: dict[str, bytes] = {}
    for name, sequence, _, _ in fasta_records(spec["assembly"], limits):
        limits.require((len(ids) + 1) * 1024, "Preflight assembly metadata")
        if name in hashes:
            raise ValueError(f"Duplicate assembly ID: {name}")
        ids.append(name)
        hashes[name] = hashlib.sha256(sequence).digest()
    if not ids:
        raise ValueError("Empty assembly")
    bins: dict[str, list[str]] = {}
    assigned = set()

    def assign(name: str, bin_id: str, *, table: bool = False) -> None:
        if table and bin_id in {"", "unbinned", "unassigned", "-", "NA", "0"}:
            return
        if name not in hashes or name in assigned:
            raise ValueError(f"Unknown or multiply assigned bin contig: {name}")
        assigned.add(name)
        bins.setdefault(bin_id, []).append(name)

    if "memberships" in spec:
        for name, bin_id in _membership_rows(Path(spec["memberships"])):
            assign(name, bin_id, table=True)
    else:
        for path in bin_fasta_paths(spec["bins"]):
            if path.stem in bins:
                raise ValueError(f"Duplicate bin ID: {path.stem}")
            for name, sequence, _, _ in fasta_records(path, limits):
                if name not in hashes or hashlib.sha256(sequence).digest() != hashes[name]:
                    raise ValueError(f"Bin sequence differs from the assembly: {name}")
                assign(name, path.stem)
            if path.stem not in bins:
                raise ValueError(f"Empty bin FASTA: {path}; no original bin may be silently omitted")
    if not bins:
        raise ValueError("No assigned bins")
    groups = (
        _mapping(spec["groups"], set(bins), "group") if "groups" in spec else {name: name for name in bins}
    )
    if "source_ids" in spec:
        _mapping(spec["source_ids"], set(bins), "source", unique=True)
    pool = set(ids) - assigned
    if "donor_ids" in spec:
        donors = Path(spec["donor_ids"]).read_text(encoding="utf-8").splitlines()
        if len(donors) != len(set(donors)) or not set(donors) <= pool:
            raise ValueError("Explicit donors must be unique, unbinned assembly contigs")
        pool = set(donors)
    coverage = _coverage(Path(spec["coverage"]), ids, spec.get("coverage_profile"))
    variance = bool(np.isfinite(coverage[:, 1]).all())
    del coverage
    if "composition" in spec:
        _composition(Path(spec["composition"]), ids, limits)
    embedding = _supplied_vectors(spec, ids, limits)
    if "encoder_path" in spec:
        if embedding is not None:
            raise ValueError("Choose precomputed embeddings or an explicit local encoder")
        from .representations.dna import local_contract

        space, files = local_contract(spec["encoder_path"], spec.get("encoder_options"))
        embedding = {"provider": "frozen_local", "space": space, "dimension": space["dimension"]}
        weight_bytes = sum(
            Path(p).stat().st_size for p in files if Path(p).suffix in {".bin", ".safetensors"}
        )
        limits.require(len(ids) * 768 + weight_bytes * 3 + 128 * 1024**2, "Frozen encoder workspace")
    elif spec.get("encoder_options"):
        raise ValueError("Encoder options require an explicit encoder_path")
    return {
        "bins": bins,
        "groups": groups,
        "groups_declared": "groups" in spec,
        "contigs": len(ids),
        "protected_contigs": len(assigned),
        "candidate_contigs": len(pool),
        "coverage_variance": variance,
        "embedding": embedding,
        "composition_formula": "supplied_table_normalized"
        if "composition" in spec
        else "valid_window_frequency",
        "candidate_policy": "explicit donor_ids"
        if "donor_ids" in spec
        else "all unassigned assembly contigs",
    }


def _prepared(spec: dict[str, Any]) -> dict[str, Any]:
    dataset = Dataset(Path(spec["prepared"]))
    manifest = dataset.verify()
    return {
        "bins": manifest["bins"],
        "groups": manifest["groups"],
        "groups_declared": manifest["groups_declared"],
        "contigs": manifest["contigs"],
        "protected_contigs": sum(len(m) for m in manifest["bins"].values()),
        "candidate_contigs": manifest["candidate_contigs"],
        "coverage_variance": manifest["coverage_variance"],
        "embedding": manifest["embedding"],
        "composition_formula": "supplied_table_normalized"
        if manifest.get("composition_source") is not None
        else "valid_window_frequency",
        "candidate_policy": "recorded prepared pool (verified immutable artifact)",
    }


def preflight(config: str | Path) -> dict[str, Any]:
    """Check full input populations, dependencies and declared resources without writes.

    This reads FASTAs and supplied vectors but never loads encoder weights into a
    model, fits, encodes, downloads, evaluates quality, or creates run/cache files.
    A ready report is not a prediction that optimization will finish or improve HQ.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "ready": False,
        "status": "INPUTS_NEED_ATTENTION",
        "quality_status": "NOT_EVALUATED_EXTERNAL_TOOLS_REQUIRED",
        "errors": [],
        "warnings": [],
        "datasets": {},
        "scope": "Input/readiness checks only; no training, encoding or HQ evaluation. No run/cache writes.",
    }
    errors, warnings = report["errors"], report["warnings"]
    try:
        plan = load_plan(config)
    except (ValueError, OSError, ImportError) as error:
        errors.append(str(error))
        return report
    document = plan.document
    methods = document["methods"]
    report.update(
        methods=[row["id"] for row in methods],
        training=document["training"],
        evaluation=document["evaluation"],
        output=document["output"],
        resources=document["resources"],
        training_row_caps={row["id"]: row["recipe"]["training"]["max_training_rows"] for row in methods},
    )
    needs_embedding = any(uses_embedding(row["recipe"]) for row in methods)
    needs_variance = any(row["recipe"]["scorer"]["name"] == "feature_classifier" for row in methods)
    inspected = {}
    for name, spec in document["datasets"].items():
        try:
            info = _prepared(spec) if "prepared" in spec else _raw(spec, plan.limits)
            inspected[name] = info
            if needs_embedding:
                embedding = info["embedding"]
                if embedding is None:
                    errors.append(
                        f"{name}: selected methods need local DNABERT-S weights or supplied embeddings"
                    )
                elif embedding["space"]["encoder"] != "dnabert_s":
                    errors.append(f"{name}: selected methods require the declared DNABERT-S vector space")
            if needs_variance and not info["coverage_variance"]:
                errors.append(
                    f"{name}: feature-classifier methods need measured coverage variance; it cannot be fabricated"
                )
            sources = [bin_id for bin_id, members in info["bins"].items() if len(members) >= 2]
            groups = {info["groups"][bin_id] for bin_id in sources}
            report["datasets"][name] = {
                key: info[key]
                for key in (
                    "contigs",
                    "protected_contigs",
                    "candidate_contigs",
                    "candidate_policy",
                    "coverage_variance",
                    "groups_declared",
                )
            } | {
                "bins": len(info["bins"]),
                "multi_contig_sources": len(sources),
                "training_groups": len(groups),
                "inputs": spec,
                "metadata_estimated_bytes": info["contigs"] * 768,
                "embedding_provider": (info["embedding"] or {}).get(
                    "provider", "supplied" if info["embedding"] else None
                ),
            }
            if not info["groups_declared"]:
                warnings.append(
                    f"{name}: bins are assumed unrelated; supply groups when related bins are known"
                )
            if info["candidate_policy"] == "all unassigned assembly contigs":
                warnings.append(
                    f"{name}: all unassigned contigs are donors, including any upstream-excluded contigs; "
                    "use donor_ids to restrict the pool explicitly"
                )
            if not info["candidate_contigs"]:
                warnings.append(f"{name}: no unassigned donor contigs; proposals cannot add contigs")
            if name == document["training"]:
                for row in methods:
                    scorer = row["recipe"]["scorer"]
                    if row["method"] != "cosine" and len(groups) < 2:
                        errors.append(
                            f"{row['id']}: training needs at least two unrelated multi-contig groups"
                        )
                    if (
                        scorer["name"] == "set_model"
                        and scorer["params"].get("validation_fraction", 0.2)
                        and len(groups) < 4
                    ):
                        errors.append(
                            f"{row['id']}: attention epoch selection needs at least four training groups"
                        )
        except (ValueError, OSError, MemoryError, ImportError) as error:
            errors.append(f"{name}: {error}")
    training = inspected.get(document["training"])
    evaluation = inspected.get(document["evaluation"])
    if training is not None and evaluation is not None:
        if training["composition_formula"] != evaluation["composition_formula"]:
            errors.append(
                "Training/evaluation composition contracts differ: use the same supplied-table or "
                "sequence-derived composition policy for both datasets"
            )
        train_embedding, eval_embedding = training["embedding"], evaluation["embedding"]
        if (
            needs_embedding
            and train_embedding is not None
            and eval_embedding is not None
            and train_embedding["space"] != eval_embedding["space"]
        ):
            errors.append(
                "Training/evaluation embedding-space contracts differ: use vectors produced with "
                "the same encoder, tokenizer, pooling, window, strand and normalization settings"
            )
    if evaluation is not None:
        recipients = document["recipients"] or list(evaluation["bins"])
        missing = set(recipients) - set(evaluation["bins"])
        if missing:
            errors.append("Unknown recipient bins: " + ", ".join(sorted(missing)))
        report["recipients"] = len(recipients)
    dependencies = {"numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"}
    devices = set()
    for row in methods:
        scorer = row["recipe"]["scorer"]
        if scorer["name"] == "set_model":
            dependencies.add("torch")
            devices.add(scorer["params"].get("device", "cpu"))
    for name, info in inspected.items():
        embedding = info["embedding"] or {}
        if embedding.get("provider") in {"frozen_local"}:
            dependencies.update({"torch", "transformers", "einops"})
            spec = document["datasets"][name]
            devices.add(embedding.get("execution", spec.get("encoder_options", {})).get("device", "cpu"))
    report["dependencies"] = {}
    for package in sorted(dependencies):
        try:
            report["dependencies"][package] = version(package)
        except PackageNotFoundError:
            errors.append(
                f"Missing dependency {package}; install with -c requirements.txt '.[sets,embeddings]'"
            )
    if "torch" in report["dependencies"]:
        from .numerical import torch_execution

        for device in sorted(devices):
            try:
                torch_execution(device)
            except (ValueError, RuntimeError, AssertionError, ImportError, OSError) as error:
                errors.append(f"{device}: {error}")
    if document["training"] == document["evaluation"]:
        warnings.append(
            "Training and evaluation use the same dataset: transductive use, not independent validation"
        )
    report["ready"] = not errors
    report["status"] = "READY_FOR_RUN" if report["ready"] else "INPUTS_NEED_ATTENTION"
    report["quality_status"] = "NOT_EVALUATED_EXTERNAL_TOOLS_REQUIRED"
    return report
