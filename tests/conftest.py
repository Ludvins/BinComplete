"""Small synthetic populations; no scientific study paths are used."""

from pathlib import Path

import numpy as np
import pytest

from bincomplete import ResourceLimits, prepare_dataset
from bincomplete.artifacts import save


def population(root: Path, *, seed=19, prefix="", variance=True):
    root.mkdir()
    rng = np.random.default_rng(seed)
    names = [f"{prefix}c{i}" for i in range(14)]
    with (root / "assembly.fa").open("w") as stream:
        for name in names:
            sequence = "".join(rng.choice(list("ACGT"), 2400))
            stream.write(f">{name} description\n{sequence[:1200]}\n{sequence[1200:]}\n")
    (root / "members.tsv").write_text(
        "contig_id\tbin_id\n" + "".join(f"{names[i]}\tb{i // 2}\n" for i in range(8)), encoding="utf-8"
    )
    (root / "coverage.tsv").write_text(
        "contig_id\tmean"
        + ("\tvariance" if variance else "")
        + "\tcompleteness\n"
        + "".join(
            f"{name}\t{40 + i % 3}" + ("\t9" if variance else "") + "\t999\n" for i, name in enumerate(names)
        ),
        encoding="utf-8",
    )
    embedding = rng.normal(size=(len(names), 8)).astype("float32")
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    # Store a deliberately different row order without copying it at load time.
    np.save(root / "embedding.npy", embedding[::-1])
    (root / "embedding_ids.txt").write_text("\n".join(names[::-1]) + "\n", encoding="utf-8")
    save(
        root / "space.json",
        {
            "encoder": "dnabert_s",
            "encoder_sha256": "synthetic_encoder_v1",
            "tokenizer_sha256": "synthetic_tokenizer_v1",
            "pooling": "masked_mean",
            "window_bp": 6000,
            "windows": 4,
            "max_tokens": 512,
            "normalization": "l2",
            "dimension": 8,
        },
    )
    return prepare_dataset(
        root / "prepared",
        assembly=root / "assembly.fa",
        memberships=root / "members.tsv",
        coverage=root / "coverage.tsv",
        embeddings=root / "embedding.npy",
        embedding_ids=root / "embedding_ids.txt",
        embedding_space=root / "space.json",
    )


@pytest.fixture
def dataset(tmp_path):
    return population(tmp_path / "data")


@pytest.fixture
def limits():
    return ResourceLimits(memory_gib=0.5, threads=1, cache_gib=0.01, chunk_size=3)


@pytest.fixture
def settings():
    return {
        "masker": {"params": {"extra_masks": 0}},
        "classifier": {"params": {"max_iter": 3, "min_samples_leaf": 1}},
        "retriever": {"params": {"min_length": 0, "top_k": 4}},
        "negative_sampler": {"params": {"hard": 3, "random": 2}},
    }
