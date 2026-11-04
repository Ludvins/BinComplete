"""A bounded, deterministic synthetic installation demo; no biological claims."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import read, save, sha
from .engine import ProposalArtifact
from .experiments import run_experiment
from .inventory import candidate_table
from .resources import ResourceLimits
from .standard import STANDARD_METHODS, standard_settings


def demo(output: str | Path) -> Path:
    """Generate four original bins and six donors, then fit all six tiny profiles.

    The vectors are artificial, HGB uses three iterations and attention one epoch.
    This deliberately reduced installation check never downloads, encodes DNA or
    establishes HQ. Existing exact demo inputs/results are verified on resumption.
    """
    root = Path(output).resolve()
    if not root.exists():
        (root / "data").mkdir(parents=True)
        rng = np.random.default_rng(19)
        ids = [f"contig_{i}" for i in range(14)]
        with (root / "data/assembly.fa").open("w", encoding="utf-8") as stream:
            for name in ids:
                stream.write(f">{name}\n{''.join(rng.choice(list('ACGT'), 2400))}\n")
        (root / "data/memberships.tsv").write_text(
            "contig_id\tbin_id\n" + "".join(f"{ids[i]}\tbin_{i // 2}\n" for i in range(8)), encoding="utf-8"
        )
        (root / "data/coverage.tsv").write_text(
            "contig_id\tmean\tvariance\n"
            + "".join(f"{name}\t{40 + i % 3}\t9\n" for i, name in enumerate(ids)),
            encoding="utf-8",
        )
        (root / "data/groups.tsv").write_text(
            "bin_id\tgroup_id\n" + "".join(f"bin_{i}\tgroup_{i}\n" for i in range(4)), encoding="utf-8"
        )
        vectors = rng.normal(size=(14, 8)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(root / "data/vectors.npy", vectors)
        (root / "data/vector_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        save(
            root / "data/space.json",
            {
                "encoder": "dnabert_s",
                "encoder_sha256": "SYNTHETIC_DEMO",
                "tokenizer_sha256": "SYNTHETIC_DEMO",
                "pooling": "synthetic",
                "window_bp": 6000,
                "windows": 4,
                "max_tokens": 512,
                "normalization": "l2",
                "dimension": 8,
            },
        )
        methods = []
        for method in STANDARD_METHODS:
            settings = standard_settings(method, device="cpu")
            settings["retriever"]["params"].update(min_length=0, top_k=4)
            settings["masker"]["params"]["extra_masks"] = 0
            settings["negative_sampler"]["params"].update(hard=3, random=2)
            if method in {"kmers", "dnabert_s", "hgb_full_pool"}:
                settings["classifier"]["params"].update(max_iter=3, min_samples_leaf=1)
            if method == "single":
                settings["masker"] = {"params": {}}
                settings["scorer"]["params"].update(
                    max_epochs=1,
                    width=16,
                    heads=2,
                    layers=1,
                    feedforward=24,
                    batch_size=2,
                    repeats_per_source=1,
                )
            methods.append({"method": method, "settings": settings})
        config = {
            "schema_version": 1,
            "output": "results",
            "cache": "cache",
            "on_error": "stop",
            "training": "sample",
            "evaluation": "sample",
            "resources": {"threads": 1, "memory_gib": 0.5, "cache_gib": 0.05, "chunk_size": 3},
            "methods": methods,
            "datasets": {
                "sample": {
                    "assembly": "data/assembly.fa",
                    "memberships": "data/memberships.tsv",
                    "coverage": "data/coverage.tsv",
                    "groups": "data/groups.tsv",
                    "embeddings": "data/vectors.npy",
                    "embedding_ids": "data/vector_ids.txt",
                    "embedding_space": "data/space.json",
                }
            },
        }
        save(root / "experiment.json", config)
        save(
            root / "demo.json",
            {
                "kind": "synthetic_installation_demo",
                "inputs": {
                    p.relative_to(root).as_posix(): sha(p)
                    for p in [*sorted((root / "data").iterdir()), root / "experiment.json"]
                },
            },
        )
    if not (root / "demo.json").exists():
        raise ValueError("Output is not a completed demo setup; preserve it and choose a new directory")
    for name, digest in read(root / "demo.json")["inputs"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Demo input changed: {name}; use a new directory")
    artifact = run_experiment(root / "experiment.json")
    summary = artifact.manifest
    if summary["status"] != "SUCCEEDED":
        raise ValueError(f"Demo failed; inspect {artifact.path / 'report.md'}")
    proposals = [ProposalArtifact(Path(row["proposals"])) for row in summary["methods"].values()]
    candidate_table(proposals, root / "candidates.tsv", limits=ResourceLimits(threads=1, memory_gib=0.5))
    save(
        root / "demo_result.json",
        {
            "status": "PASSED",
            "methods": len(proposals),
            "bins": 4,
            "encoding_jobs": 0,
            "quality_jobs": 0,
            "known_hq_gains": None,
            "scope": "Synthetic installation check with artificial vectors and reduced fit settings",
        },
    )
    return root / "demo_result.json"
