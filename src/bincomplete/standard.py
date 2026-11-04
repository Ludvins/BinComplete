"""Portable, packaged six-method experiment recipe; no dataset-specific decisions."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .methods import normalize_method

STANDARD_METHODS = ("cosine", "kmers", "dnabert_s", "single", "hgb_full_pool", "logistic_full_pool")


def standard_settings(method: str, *, device: str = "cuda:0") -> dict[str, Any]:
    """Return the standard recipe, with an explicit attention execution device.

    CPU execution is supported but is a different numerical profile from the
    reported CUDA experiment. No automatic device fallback is performed.
    """
    method = normalize_method(method)
    if device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("Choose cpu, cuda or cuda:0 explicitly")
    value = json.loads(files("bincomplete").joinpath(f"profiles/standard_v1/{method}.json").read_text())
    if method == "single":
        # Attention's preserved model contract names logical GPU zero "cuda";
        # the frozen encoder independently records that same device as "cuda:0".
        value["scorer"]["params"]["device"] = "cuda" if device == "cuda:0" else device
    return value


def initialize(output: str | Path, *, device: str = "cuda:0", embedding_mode: str = "supplied") -> Path:
    """Create a new portable experiment directory without overwriting existing work.

    Paths in the emitted configuration are relative to that file. Users supply
    original memberships, measured coverage, groups and recipient IDs. No quality
    reference or expected outcome is an input to the recipe.
    """
    settings = {method: standard_settings(method, device=device) for method in STANDARD_METHODS}
    if embedding_mode not in {"supplied", "encode"}:
        raise ValueError("embedding_mode must be supplied or encode")
    dataset: dict[str, Any] = {
        "assembly": "data/assembly.fasta",
        "memberships": "data/memberships.tsv",
        "coverage": "data/coverage.tsv",
        "groups": "data/groups.tsv",
    }
    if embedding_mode == "supplied":
        dataset.update(
            embeddings="data/embeddings.npy",
            embedding_ids="data/embedding_ids.txt",
            embedding_space="data/embedding_space.json",
        )
    else:
        dataset.update(
            encoder_path="models/dnabert_s",
            encoder_options={
                "pooling": "non_special_bp_weighted_windows",
                "window_bp": 6000,
                "windows": 4,
                "max_tokens": 2048,
                "reverse_complement": True,
                "storage_dtype": "float16",
                "device": device,
                "precision": "float32" if device == "cpu" else "bfloat16",
                "batch_size": 8,
            },
        )
    document = {
        "schema_version": 1,
        "output": "results",
        "cache": "cache",
        "resources": {"threads": 2, "memory_gib": 6, "cache_gib": 10, "chunk_size": 4096},
        "training": "sample",
        "evaluation": "sample",
        "on_error": "stop",
        "datasets": {"sample": dataset},
        "recipients": "data/recipients.txt",
        "methods": [{"method": method, "settings": f"settings/{method}.json"} for method in STANDARD_METHODS],
    }
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "settings").mkdir()
    (root / "data").mkdir()
    for method, value in settings.items():
        (root / f"settings/{method}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    path = root / "experiment.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (root / "data/README.txt").write_text(
        "Supply original assembly and memberships, not accepted HQ repairs.\n"
        "membership TSV: contig_id, bin_id; coverage TSV: contig_id, mean, variance.\n"
        "groups TSV: bin_id, group_id; recipients.txt: one bin ID per line.\n"
        "All TSV fields are tab-separated. Coverage variance must be measured.\n"
        "Keep data/weights external if preferred: edit the paths in experiment.json.\n"
        "Supplied embeddings require row IDs and a matching encoder-space JSON.\n"
        "Optional composition input must declare the same feature contract.\n"
        "See docs/portable_recipe.md for formats, separate training data and cohorts.\n"
        "No HQ evaluation is performed by this experiment.\n",
        encoding="utf-8",
    )
    return path


def initialize_compact(
    output: str | Path,
    *,
    input_root: str | Path | None = None,
    global_config: str | Path | None = None,
    recipients: str | Path | None = None,
    device: str | None = None,
    inputs: dict[str, Any] | None = None,
    methods: list[str] | None = None,
) -> Path:
    """Create a dataset-only experiment inheriting the global preset at execution."""
    if device is not None:
        standard_settings("cosine", device=device)
    inputs = {key: value for key, value in (inputs or {}).items() if value is not None}
    if "bins" in inputs and "memberships" in inputs:
        raise ValueError("Supply exactly one of --bins or --memberships")
    if input_root is not None:
        dataset = {"input_format": "mmlong2", "input_root": str(Path(input_root).expanduser().resolve())}
    elif inputs:
        if not {"assembly", "coverage"} <= set(inputs) or not ({"bins", "memberships"} & set(inputs)):
            raise ValueError(
                "Generic init needs --assembly, --coverage and exactly one of --bins/--memberships"
            )
        dataset = {}
    else:
        dataset = {
            "assembly": "data/assembly.fasta",
            "memberships": "data/memberships.tsv",
            "coverage": "data/coverage.tsv",
        }
    for key, value in inputs.items():
        dataset[key] = str(value) if key == "coverage_profile" else str(Path(value).expanduser().resolve())
    document: dict[str, Any] = {"schema_version": 2, "output": "results", "datasets": {"sample": dataset}}
    if methods is not None:
        resolved_methods = [normalize_method(method) for method in methods]
        if not methods or len(set(resolved_methods)) != len(methods):
            raise ValueError("Select unique supported methods")
        document["methods"] = methods
    if global_config is not None:
        document["global_config"] = str(Path(global_config).expanduser().resolve())
    if recipients is not None:
        document["recipients"] = str(Path(recipients).expanduser().resolve())
    if device is not None:
        document["device"] = device
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    path = root / "experiment.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (root / "README.txt").write_text(
        "Train and propose on original memberships from the same dataset (transductive).\n"
        "Run bincomplete preflight --config experiment.json before experiment.\n"
        "Use the full assembly including unbinned donors, original bins/memberships and measured coverage.\n"
        "Feature classifiers require measured variance. All supplied bin members remain protected.\n"
        "mmlong2 handoff: results/ plus the full primary tmp/binning/mapping/1-<profile>.tsv.\n"
        "The default donor pool is all unassigned assembly contigs; use donor_ids to restrict it.\n"
        "Evaluation compares proposals; HQ requires external assessment, approval and reconciliation.\n"
        "See docs/global_configuration.md, docs/generic_workflow.md and docs/evaluation.md.\n",
        encoding="utf-8",
    )
    return path
