"""Exercise the installed portable CLI on two small synthetic populations.

This script must import the installed wheel, not a checkout via PYTHONPATH.
It performs fresh membership fits and resumes them; no DNA or quality jobs run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def command(*arguments: str) -> None:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LOKY_MAX_CPU_COUNT="2")
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-B", "-m", "bincomplete", *arguments], env=environment, check=True)


def smoke(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    global_path = output / "global.json"
    command("config", "init", "--output", str(global_path), "--device", "cpu")
    defaults = json.loads(global_path.read_text())
    defaults["cache"] = str(output / "shared_cache")
    global_path.write_text(json.dumps(defaults), encoding="utf-8")
    populations = []
    for seed in (19, 23):
        work = output / f"sample_{seed}"
        if seed == 19:
            command("init", "--output", str(work), "--device", "cpu")
        else:
            command(
                "init",
                "--output",
                str(work),
                "--global-config",
                str(global_path),
                "--assembly",
                str(work / "data/assembly.fasta"),
                "--memberships",
                str(work / "data/memberships.tsv"),
                "--coverage",
                str(work / "data/coverage.tsv"),
                "--groups",
                str(work / "data/groups.tsv"),
                "--recipients",
                str(work / "data/recipients.txt"),
                "--embeddings",
                str(work / "data/embeddings.npy"),
                "--embedding-ids",
                str(work / "data/embedding_ids.txt"),
                "--embedding-space",
                str(work / "data/embedding_space.json"),
            )
        data = work / "data"
        data.mkdir(exist_ok=True)
        rng = np.random.default_rng(seed)
        ids = [f"sample{seed}_c{i}" for i in range(14)]
        with (data / "assembly.fasta").open("w", encoding="utf-8") as stream:
            for name in ids:
                stream.write(f">{name}\n{''.join(rng.choice(list('ACGT'), 2400))}\n")
        (data / "memberships.tsv").write_text(
            "contig_id\tbin_id\n" + "".join(f"{ids[i]}\tb{i // 2}\n" for i in range(8)), encoding="utf-8"
        )
        (data / "coverage.tsv").write_text(
            "contig_id\tmean\tvariance\n"
            + "".join(f"{name}\t{40 + i % 3}\t9\n" for i, name in enumerate(ids)),
            encoding="utf-8",
        )
        (data / "groups.tsv").write_text(
            "bin_id\tgroup_id\n" + "".join(f"b{i}\tg{i}\n" for i in range(4)), encoding="utf-8"
        )
        (data / "recipients.txt").write_text("\n".join(f"b{i}" for i in range(4)) + "\n", encoding="utf-8")
        vectors = rng.normal(size=(14, 8)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(data / "embeddings.npy", vectors)
        (data / "embedding_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        space = {
            "encoder": "dnabert_s",
            "encoder_sha256": "synthetic_encoder",
            "tokenizer_sha256": "synthetic_tokenizer",
            "pooling": "masked_mean",
            "window_bp": 6000,
            "windows": 4,
            "max_tokens": 512,
            "normalization": "l2",
            "dimension": 8,
        }
        (data / "embedding_space.json").write_text(json.dumps(space), encoding="utf-8")
        command("doctor", "--device", "cpu", "--output", str(work / "environment.json"))
        config = work / "experiment.json"
        command("preflight", "--config", str(config))
        assert not (work / "results").exists(), "Preflight must not create run outputs"
        command("experiment", "--config", str(config))
        status = work / "results/status.json"
        before = status.read_bytes()
        command("experiment", "--config", str(config))
        assert before == status.read_bytes(), "Completed experiment should resume without fitting"
        summary = json.loads((work / "results/summary.json").read_text())
        assert summary["status"] == "SUCCEEDED" and len(summary["methods"]) == 6
        for row in summary["methods"].values():
            proposals = json.loads((Path(row["proposals"]) / "proposals.json").read_text())
            assert proposals["search"]["name"] == "expanded_ranked_alternatives"
            assert max(value["proposals"] for value in proposals["coverage"].values()) <= 128
        populations.append(
            {
                "seed": seed,
                "methods_succeeded": 6,
                "resume_verified": True,
                "configuration": "standalone" if seed == 19 else "global_defaults",
            }
        )
    result = {"status": "PASSED", "populations": populations, "quality_evaluations": 0, "new_encoding": 0}
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(smoke(parser.parse_args().output.resolve()), indent=2))
