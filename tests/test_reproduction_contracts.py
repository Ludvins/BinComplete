"""Regression checks for validation protocols and physical numerical identity."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from bincomplete.artifacts import fingerprint
from bincomplete.methods import recipe


def test_single_default_retains_historical_epoch_selection_task(limits):
    spec = recipe("single", limits)
    assert spec["masker"]["name"] == "leave_one_out"
    assert spec["validation"] == {
        "candidate_policy": "all_known",
        "masker": {
            "name": "leave_one_out_and_count",
            "params": {"max_hidden": 4, "extra_masks": 20, "decoy_every": 0},
        },
    }
    assert spec["profile_version"] == 2
    assert recipe("kmers", limits)["validation"] == {"candidate_policy": "retrieved", "masker": None}


def test_explicit_validation_changes_recipe_without_changing_training(limits):
    baseline = recipe("single", limits)
    override = {"validation": {"candidate_policy": "retrieved", "masker": None}}
    changed = recipe("single", limits, override)
    assert changed["training"] == baseline["training"]
    assert changed["masker"] == baseline["masker"]
    assert changed["validation"] == override["validation"]
    assert fingerprint(changed) != fingerprint(baseline)
    changed["validation"]["candidate_policy"] = "all_known"
    assert override["validation"]["candidate_policy"] == "retrieved"


@pytest.mark.parametrize(
    "value",
    [
        {"candidate_policy": "quality_selected"},
        {"candidate_policy": []},
        {"unknown": True},
        {"masker": {"params": {}}},
        {"masker": {"name": "leave_one_out", "params": []}},
    ],
)
def test_invalid_validation_settings_are_rejected(limits, value):
    with pytest.raises(ValueError, match="validation"):
        recipe("single", limits, {"validation": value})


def test_encoder_cache_key_distinguishes_actual_gpu(limits, monkeypatch):
    pytest.importorskip("torch")
    from bincomplete import numerical
    from bincomplete.representations import dna

    state = {"cuda": {"name": "GPU A", "uuid": "physical-a"}}
    monkeypatch.setattr(numerical, "torch_execution", lambda device: deepcopy(state))
    monkeypatch.setattr(dna, "version", lambda name: "test-version")
    cache = SimpleNamespace(array=lambda identity, writer, **kwargs: identity)
    context = SimpleNamespace(
        reserved_bytes=0,
        limits=limits,
        cache=cache,
        store=SimpleNamespace(
            ids=["a"],
            manifest={"assembly": "assembly.fa", "inputs_sha256": {"assembly.fa": "same-sequences"}},
        ),
    )
    spec = {
        "space": {"dimension": 8},
        "model_files": {},
        "execution": {"device": "cuda"},
    }
    first = dna.frozen_array(context, spec)
    state["cuda"] = {"name": "GPU B", "uuid": "physical-b"}
    second = dna.frozen_array(context, spec)
    assert first["execution"] == second["execution"]
    assert fingerprint(first) != fingerprint(second)
    assert first["numerical_runtime"]["cuda"]["uuid"] == "physical-a"
    assert second["numerical_runtime"]["cuda"]["uuid"] == "physical-b"
