"""Reject incompatible inputs before fitting, using only synthetic populations."""

import shutil

import numpy as np
import pytest

from bincomplete import prepare_dataset
from bincomplete.artifacts import read, save
from bincomplete.preflight import preflight


def raw_inputs(dataset):
    return {
        key: str(dataset.path.parent / filename)
        for key, filename in {
            "assembly": "assembly.fa",
            "memberships": "members.tsv",
            "coverage": "coverage.tsv",
            "embeddings": "embedding.npy",
            "embedding_ids": "embedding_ids.txt",
            "embedding_space": "space.json",
        }.items()
    }


def config(tmp_path, training, evaluation=None, method="cosine"):
    path = tmp_path / "experiment.json"
    save(
        path,
        {
            "schema_version": 1,
            "output": str(tmp_path / "never-created"),
            "training": "train",
            "evaluation": "apply" if evaluation is not None else "train",
            "datasets": {"train": training} | ({"apply": evaluation} if evaluation is not None else {}),
            "methods": [{"method": method}],
            "resources": {"threads": 1, "memory_gib": 0.5, "cache_gib": 0.01, "chunk_size": 3},
        },
    )
    return path


def inspection_spec(spec, prepared, root):
    if not prepared:
        return spec
    return {"prepared": str(prepare_dataset(root / "prepared", **spec).path)}


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("method", ["cosine", "hgb_full_pool", "single"])
def test_missing_embeddings_is_an_actionable_read_only_report(tmp_path, dataset, prepared, method):
    spec = {key: value for key, value in raw_inputs(dataset).items() if not key.startswith("embedding")}
    spec = inspection_spec(spec, prepared, tmp_path)
    path = config(tmp_path, spec, method=method)
    before = set(tmp_path.rglob("*"))
    result = preflight(path)
    assert not result["ready"]
    assert "supplied embeddings" in " ".join(result["errors"])
    assert before == set(tmp_path.rglob("*"))


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "field", ["dimension", "encoder_sha256", "tokenizer_sha256", "pooling", "reverse_complement"]
)
def test_preflight_rejects_transfer_space_changes(tmp_path, dataset, prepared, field):
    training = raw_inputs(dataset)
    copy = tmp_path / "other"
    copy.mkdir()
    evaluation = {}
    for key, value in training.items():
        destination = copy / value.replace("\\", "/").rsplit("/", 1)[-1]
        shutil.copy2(value, destination)
        evaluation[key] = str(destination)
    space = read(evaluation["embedding_space"])
    if field == "dimension":
        space[field] = 3
        values = np.load(evaluation["embeddings"])[:, :3]
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        np.save(evaluation["embeddings"], values)
    else:
        space[field] = True if field == "reverse_complement" else "different_protocol"
    save(evaluation["embedding_space"], space)
    training = inspection_spec(training, prepared, tmp_path / "train")
    evaluation = inspection_spec(evaluation, prepared, tmp_path / "apply")
    path = config(tmp_path, training, evaluation)
    result = preflight(path)
    assert not result["ready"]
    assert "embedding-space contracts differ" in " ".join(result["errors"])
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("prepared", [False, True])
def test_matching_transfer_spaces_are_ready(tmp_path, dataset, prepared):
    spec = inspection_spec(raw_inputs(dataset), prepared, tmp_path)
    result = preflight(config(tmp_path, spec, spec))
    assert result["ready"], result["errors"]


@pytest.mark.parametrize("prepared", [False, True])
def test_composition_only_ignores_unneeded_embedding_space(tmp_path, dataset, prepared):
    training = raw_inputs(dataset)
    evaluation = {key: value for key, value in training.items() if not key.startswith("embedding")}
    training = inspection_spec(training, prepared, tmp_path / "train")
    evaluation = inspection_spec(evaluation, prepared, tmp_path / "apply")
    result = preflight(config(tmp_path, training, evaluation, method="kmers"))
    assert result["ready"], result["errors"]


@pytest.mark.parametrize("prepared", [False, True])
def test_preflight_rejects_composition_source_changes(tmp_path, dataset, prepared):
    training = raw_inputs(dataset)
    composition = tmp_path / "composition.tsv"
    ids = (dataset.path.parent / "embedding_ids.txt").read_text().splitlines()
    composition.write_text(
        "contig_id\t"
        + "\t".join(map(str, range(136)))
        + "\n"
        + "".join(name + "\t" + "\t".join(["1"] * 136) + "\n" for name in ids),
        encoding="utf-8",
    )
    evaluation = training | {"composition": str(composition)}
    training = inspection_spec(training, prepared, tmp_path / "train")
    evaluation = inspection_spec(evaluation, prepared, tmp_path / "apply")
    result = preflight(config(tmp_path, training, evaluation, method="kmers"))
    assert not result["ready"]
    assert "composition contracts differ" in " ".join(result["errors"])
