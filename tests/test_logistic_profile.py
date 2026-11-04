"""Only the selected logistic family joins the maintained method portfolio."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from conftest import population
from sklearn.exceptions import ConvergenceWarning

from bincomplete import fit, propose
from bincomplete.classifiers.logistic import ScaledLogistic
from bincomplete.cli import main
from bincomplete.core import registered
from bincomplete.methods import METHODS, recipe


def test_weighted_scaler_is_training_only(monkeypatch):
    x = np.array([[0.0, 100.0], [1.0, 200.0], [2.0, 400.0], [4.0, 800.0]])
    weights = np.array([4.0, 1.0, 1.0, 2.0])
    model = ScaledLogistic().fit(x, np.array([0, 0, 1, 1]), sample_weight=weights)
    assert np.allclose(model.scaler.mean_, np.average(x, axis=0, weights=weights))
    prior = model.scaler.mean_.copy()
    monkeypatch.setattr(model.scaler, "fit", lambda *a, **k: pytest.fail("inference refit scaler"))
    scores = model.predict_scores(np.array([[100.0, 1e9], [-100.0, -1e9]]))
    assert np.isfinite(scores).all() and np.array_equal(prior, model.scaler.mean_)


def test_failed_logistic_convergence_is_not_silently_accepted(monkeypatch):
    model = ScaledLogistic()

    def failed_fit(*args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning)

    monkeypatch.setattr(model.estimator, "fit", failed_fit)
    with pytest.raises(ConvergenceWarning):
        model.fit(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0, 1]))


def test_logistic_saved_model_proposals_are_immutable_and_protected(dataset, limits, tmp_path):
    model = fit(
        dataset,
        tmp_path / "model",
        method="logistic_full_pool",
        limits=limits,
        settings={
            "masker": {"params": {"extra_masks": 0}},
            "negative_sampler": {"params": {"hard": 3, "random": 2}},
            "retriever": {"params": {"min_length": 0, "coverage_fold": 100}},
        },
        cache=tmp_path / "cache",
    )
    saved = (model.path / "scorer.joblib").read_bytes()
    output = propose(model, dataset, tmp_path / "proposals", limits=limits, cache=tmp_path / "cache")
    assert (model.path / "scorer.joblib").read_bytes() == saved
    assert "classifiers/logistic.py" in model.manifest["compatibility"]["numerical_code"]
    assert model.manifest["labels"] == "original within-bin membership"
    classifier = joblib.load(model.path / "scorer.joblib").classifier
    assert classifier.parameters["C"] == 1.0 and classifier.parameters["random_state"] == 73019
    protected = set().union(*map(set, dataset.manifest["bins"].values()))
    counts = {}
    for line in (output.path / "memberships.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert set(row["added"]).isdisjoint(protected)
        assert set(dataset.manifest["bins"][row["recipient"]]) <= set(row["members"])
        counts[row["recipient"]] = counts.get(row["recipient"], 0) + 1
    assert counts and max(counts.values()) <= 20


def test_logistic_rejects_missing_measured_variance(tmp_path, limits):
    data = population(tmp_path / "data", variance=False)
    with pytest.raises(ValueError, match="measured coverage variance"):
        fit(data, tmp_path / "model", method="logistic_full_pool", limits=limits)


def test_six_method_recipe_and_cli_include_only_selected_addition(limits, capsys, tmp_path):
    expected = ("kmers", "dnabert_s", "single", "cosine", "hgb_full_pool", "logistic_full_pool")
    assert METHODS == expected
    with pytest.raises(SystemExit) as exit_code:
        main(["fit", "--help"])
    assert exit_code.value.code == 0 and "logistic_full_pool" in capsys.readouterr().out
    assert "extra_trees" not in registered("classifier")
    assert not any("vae" in name for name in registered("scorer"))
    for rejected in ["extra_trees_full_pool", "vae_full_pool"]:
        with pytest.raises(ValueError, match="Unknown method"):
            recipe(rejected, limits)
    import tomllib

    folder = Path(__file__).resolve().parents[1]
    config = tomllib.loads((folder / "configs/experiment.mmlong2.example.toml").read_text())
    from bincomplete.defaults import expand_experiment, write_defaults

    # The current example inherits its six profiles from an explicit global file.
    # Resolve recipes without trying to discover the placeholder sample's files.
    config["global_config"] = str(write_defaults(tmp_path / "global.toml"))
    resolved, settings, _ = expand_experiment(config, folder / "configs")
    assert {entry["method"] for entry in resolved["methods"]} == set(expected)
    assert set(settings) == set(expected)
    for method in expected:
        assert (folder / "configs/fresh" / f"{method}.json").is_file()
