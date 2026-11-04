"""Bounded installed-wheel smoke with real, read-only local DNABERT-S weights.

This uses ten 256-bp artificial contigs, CPU FP32, one thread and a 4 GiB
allocation budget. All six toy methods fit after one shared real encoding.
No metagenomic study, expected HQ fixture or quality tool is opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def smoke(encoder_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "hf_home", "hf_modules"):
        (output / name).mkdir()
    os.environ.update(
        HF_HOME=str(output / "hf_home"),
        HF_MODULES_CACHE=str(output / "hf_modules"),
        TMPDIR=str(output / "tmp"),
        TEMP=str(output / "tmp"),
        TMP=str(output / "tmp"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    started = time.monotonic()
    import bincomplete
    from bincomplete.artifacts import read, save
    from bincomplete.preflight import preflight
    from bincomplete.standard import STANDARD_METHODS

    assert "site-packages" in Path(bincomplete.__file__).parts, "Test the installed wheel, not a checkout"
    rng = np.random.default_rng(19)
    ids = [f"test_contig_{i}" for i in range(10)]
    with (output / "assembly.fa").open("w", encoding="utf-8") as stream:
        for name in ids:
            stream.write(f">{name}\n{''.join(rng.choice(list('ACGT'), 256))}\n")
    (output / "members.tsv").write_text(
        "contig_id\tbin_id\n" + "".join(f"{ids[i]}\tbin_{i // 2}\n" for i in range(8)), encoding="utf-8"
    )
    (output / "coverage.tsv").write_text(
        "contig_id\tmean\tvariance\n" + "".join(f"{name}\t40\t9\n" for name in ids), encoding="utf-8"
    )
    (output / "groups.tsv").write_text(
        "bin_id\tgroup_id\n" + "".join(f"bin_{i}\tgroup_{i}\n" for i in range(4)), encoding="utf-8"
    )
    methods = []
    for method in STANDARD_METHODS:
        settings = {
            "retriever": {"params": {"min_length": 0, "top_k": 4}},
            "negative_sampler": {"params": {"hard": 3, "random": 2}},
        }
        if method != "single":
            settings["masker"] = {"params": {"extra_masks": 0}}
        if method in {"kmers", "dnabert_s", "hgb_full_pool"}:
            settings["classifier"] = {"params": {"max_iter": 3, "min_samples_leaf": 1}}
        if method == "single":
            settings["scorer"] = {
                "params": {
                    "max_epochs": 1,
                    "width": 16,
                    "heads": 2,
                    "layers": 1,
                    "feedforward": 24,
                    "batch_size": 2,
                    "repeats_per_source": 1,
                }
            }
        methods.append({"method": method, "settings": settings})
    save(
        output / "global.json",
        {
            "device": "cpu",
            "encoder_path": str(encoder_path),
            "cache": str(output / "cache"),
            "resources": {"threads": 1, "memory_gib": 4, "cache_gib": 1, "chunk_size": 8},
            "encoder": {"batch_size": 2},
        },
    )
    save(
        output / "experiment.json",
        {
            "schema_version": 2,
            "global_config": "global.json",
            "output": "results",
            "methods": methods,
            "datasets": {
                "sample": {
                    "assembly": "assembly.fa",
                    "memberships": "members.tsv",
                    "coverage": "coverage.tsv",
                    "groups": "groups.tsv",
                }
            },
        },
    )
    report = preflight(output / "experiment.json")
    save(output / "preflight.json", report)
    assert report["ready"], report["errors"]
    print(json.dumps({"stage": "raw_fasta_to_six_methods", "contigs": 10}), flush=True)
    run = bincomplete.run_experiment(
        output / "experiment.json", progress=lambda row: print(json.dumps(row), flush=True)
    )
    summary = run.manifest
    assert summary["status"] == "SUCCEEDED" and len(summary["methods"]) == 6
    artifacts = [bincomplete.ProposalArtifact(Path(row["proposals"])) for row in summary["methods"].values()]
    bincomplete.candidate_table(artifacts, output / "candidates.tsv")
    proposal = artifacts[0]
    first = json.loads((proposal.path / "memberships.jsonl").read_text().splitlines()[0])
    exported = bincomplete.export(proposal, output / "exported", proposal_ids=[first["proposal_id"]])
    assert exported.with_name("bins.tsv").is_file()
    assert read(exported)["quality_verdicts"].startswith("Supplied externally")
    encoded = list((output / "cache/arrays").glob("*/values.npy"))
    # Other numerical arrays may coexist; assert a completed real encoder cache identity exists.
    assert any(
        '"local_dnabert_s_v2"' in p.read_text() for p in (output / "cache/arrays").glob("*/complete.json")
    )
    result = {
        "status": "PASSED",
        "version": bincomplete.__version__,
        "python": sys.version,
        "elapsed_seconds": time.monotonic() - started,
        "contigs_encoded": 10,
        "methods": 6,
        "cached_arrays": len(encoded),
        "quality_jobs": 0,
        "scientific_metagenomes": 0,
        "encoder": str(encoder_path),
        "scope": "Artificial DNA, actual frozen encoder, reduced toy fits; no HQ or cross-device equality claim",
    }
    save(output / "encoder_smoke_result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.encoder_path.resolve(), args.output.resolve())
