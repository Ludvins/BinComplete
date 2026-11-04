import json

import numpy as np
import pytest
from conftest import population

from bincomplete import export, fit, prepare_dataset, propose, reconcile, run
from bincomplete.artifacts import sha
from bincomplete.context import Context, RowView
from bincomplete.methods import recipe


@pytest.mark.parametrize("method", ["kmers", "dnabert_s", "hgb_full_pool", "cosine", "single"])
def test_all_methods_fit_then_propose(dataset, tmp_path, limits, settings, method):
    if method in {"single", "cosine"}:
        settings.pop("classifier")
    if method == "single":
        pytest.importorskip("torch")
        settings.pop("masker")
        settings["scorer"] = {
            "params": {
                "max_epochs": 1,
                "validation_fraction": 0,
                "width": 16,
                "heads": 2,
                "layers": 1,
                "feedforward": 24,
                "batch_size": 2,
                "repeats_per_source": 1,
            }
        }
    before = {path: sha(path) for path in dataset.path.iterdir() if path.is_file()}
    artifact = run(
        dataset,
        dataset,
        tmp_path / method,
        method=method,
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    manifest = artifact.manifest
    assert manifest["proposal_count"] > 0
    assert manifest["overlap"]["overlapping_recipients"] == (0 if method == "cosine" else 4)
    assert manifest["overlap"]["mode"] == (
        "no_exact_sequence_overlap_detected" if method == "cosine" else "transductive"
    )
    model_manifest = json.loads((tmp_path / method / "model/model.json").read_text())
    if method == "cosine":
        assert model_manifest["optimizer_calls"] == 0
        assert not (tmp_path / method / "model/examples").exists()
    else:
        assert model_manifest["training_statistics"]["positives"] > 0
    for entry in manifest["coverage"].values():
        rows = np.load(artifact.path / entry["ranking"] / "rows.npy")
        scores = np.load(artifact.path / entry["ranking"] / "scores.npy")
        assert len(rows) == len(scores) == entry["scored"]
        assert np.isfinite(scores).all()
    assert not any(path.name.endswith("partial.npy") for path in artifact.path.rglob("*"))
    assert all(sha(path) == digest for path, digest in before.items())


def test_transfer_contract_uses_space_not_array_hash(dataset, tmp_path, settings, limits):
    other = population(tmp_path / "other", seed=39, prefix="other_")
    model = fit(
        dataset,
        tmp_path / "model",
        method="dnabert_s",
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    proposals = propose(model, other, tmp_path / "proposals", limits=limits, cache=tmp_path / "cache")
    assert proposals.manifest["overlap"]["overlapping_recipients"] == 0
    assert model.manifest["training_identity"] != other.manifest["identity"]


def test_repeated_identical_operations_reuse_without_refit(dataset, tmp_path, settings, limits, monkeypatch):
    first = fit(
        dataset,
        tmp_path / "model",
        method="kmers",
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    from bincomplete.scorers.vector import FeatureClassifier

    monkeypatch.setattr(FeatureClassifier, "fit", lambda *args: pytest.fail("unexpected refit"))
    second = fit(
        dataset,
        tmp_path / "model",
        method="kmers",
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    assert first.path == second.path


def test_changed_encoder_rejected_before_scoring(dataset, tmp_path, settings, limits):
    from bincomplete.artifacts import complete, read, save

    model = fit(
        dataset,
        tmp_path / "model",
        method="dnabert_s",
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    other = population(tmp_path / "other", seed=39)
    metadata = read(other.path / "dataset.json")
    metadata["embedding"]["space"]["encoder_sha256"] = "different"
    save(other.path / "dataset.json", metadata)
    (other.path / "complete.json").unlink()
    complete(other.path, {"kind": "dataset", "identity": metadata["identity"]})
    with pytest.raises(ValueError, match="encoder-space"):
        propose(model, other, tmp_path / "bad", limits=limits, cache=tmp_path / "cache")


def test_input_and_model_corruption_are_detected(dataset, tmp_path, settings, limits):
    model = fit(
        dataset,
        tmp_path / "model",
        method="kmers",
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    with (model.path / "scorer.joblib").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        _ = model.manifest
    coverage = dataset.path.parent / "coverage.tsv"
    coverage.write_text(coverage.read_text().replace("999", "998"))
    with pytest.raises(ValueError, match="Input changed"):
        dataset.verify()


def test_embeddings_are_row_views_and_not_copied(dataset, tmp_path, limits):
    context = Context(dataset, recipe("cosine", limits), tmp_path / "work", tmp_path / "cache", limits)
    try:
        array = context.representation("embedding")
        assert isinstance(array, RowView)
        assert isinstance(array.matrix, np.memmap)
        expected = np.load(dataset.path.parent / "embedding.npy")[-1]
        np.testing.assert_array_equal(array[[0]][0], expected)
    finally:
        context.close()


def test_missing_variance_never_fabricated(tmp_path, limits):
    dataset = population(tmp_path / "data", variance=False)
    with pytest.raises(ValueError, match="measured coverage variance"):
        fit(dataset, tmp_path / "model", method="kmers", limits=limits, cache=tmp_path / "cache")
    model = fit(dataset, tmp_path / "cosine", method="cosine", limits=limits, cache=tmp_path / "cache")
    assert not model.manifest["learned"]


def test_external_approval_and_manifest_only_unchanged_bins(dataset, tmp_path, limits, settings):
    model = fit(
        dataset,
        tmp_path / "model",
        method="kmers",
        settings=settings,
        limits=limits,
        cache=tmp_path / "cache",
    )
    proposals = propose(model, dataset, tmp_path / "proposals", limits=limits, cache=tmp_path / "cache")
    rows = [json.loads(line) for line in (proposals.path / "memberships.jsonl").read_text().splitlines()]
    approved = [rows[0]["proposal_id"]]
    selection = reconcile([proposals], approved, tmp_path / "selection", limits=limits)
    result = export(selection, tmp_path / "export", limits=limits)
    manifest = json.loads(result.read_text())
    assert len(manifest["changed"]) == 1
    assert len(manifest["unchanged"]) == 3
    assert not (result.parent / "bins").exists()
    assert len(list((result.parent / "objects").glob("*.fa"))) == 1
    output = export(selection, tmp_path / "all", complete_directory=True, limits=limits)
    assert len(list((output.parent / "bins").glob("*.fa"))) == 4
    with pytest.raises(ValueError, match="Explicit proposal_ids"):
        export(proposals, tmp_path / "implicit")


def test_preparation_identity_and_bins_in_place(dataset):
    parent = dataset.path.parent
    again = prepare_dataset(
        dataset.path,
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        embeddings=parent / "embedding.npy",
        embedding_ids=parent / "embedding_ids.txt",
        embedding_space=parent / "space.json",
    )
    assert again == dataset
    assert not list(dataset.path.glob("*.fa"))
    assert not (dataset.path / "embedding.npy").exists()


def test_no_implicit_all_methods_and_no_evaluator_import():
    import sys

    from bincomplete.cli import main

    with pytest.raises(SystemExit):
        main(["fit", "--dataset", "somewhere", "--output", "elsewhere"])
    assert not any(name.startswith("bincomplete.evaluation") for name in sys.modules)
