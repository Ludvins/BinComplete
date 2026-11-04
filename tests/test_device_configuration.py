"""Validate GPU aliases through public plan construction without running a GPU job."""

import json

import pytest

from bincomplete.artifacts import read, save
from bincomplete.defaults import write_defaults
from bincomplete.experiments.config import load_plan
from bincomplete.standard import initialize, standard_settings


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_materialized_recipe_accepts_every_advertised_device(tmp_path, dataset, device):
    initialized = initialize(tmp_path / "materialized", device=device)
    value = read(initialized)
    value["datasets"] = {"sample": {"prepared": str(dataset.path)}}
    value.pop("recipients")
    save(initialized, value)
    plan = load_plan(initialized).document
    attention = next(row for row in plan["methods"] if row["method"] == "single")
    assert attention["recipe"]["scorer"]["params"]["device"] == ("cpu" if device == "cpu" else "cuda")
    assert standard_settings("single", device=device)["scorer"]["params"]["device"] in {"cpu", "cuda"}


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_compact_recipe_accepts_every_advertised_device(tmp_path, dataset, monkeypatch, device):
    global_path = write_defaults(tmp_path / "shared.toml", device=device)
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(global_path))
    config = tmp_path / "experiment.json"
    save(config, {"schema_version": 2, "datasets": {"sample": {"prepared": str(dataset.path)}}})
    plan = load_plan(config).document
    assert len(plan["methods"]) == 6
    attention = next(row for row in plan["methods"] if row["method"] == "single")
    expected = "cpu" if device == "cpu" else "cuda"
    assert attention["recipe"]["scorer"]["params"]["device"] == expected
    # This is constructor/configuration validation, with no CUDA context or model execution.
    json.dumps(plan)
