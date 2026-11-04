"""Readable method names and proposal budgets preserve stored scientific recipes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bincomplete import fit, propose
from bincomplete.artifacts import read, save
from bincomplete.cli import main
from bincomplete.defaults import expand_experiment, load_defaults
from bincomplete.experiments.config import load_plan
from bincomplete.methods import METHOD_ALIASES, METHODS, method_catalog, recipe
from bincomplete.search.expanded import ExpandedRankedAlternatives
from bincomplete.search.ranked_alternatives import RankedAlternatives
from bincomplete.standard import initialize_compact, standard_settings


def test_catalogue_explains_six_distinct_profiles(capsys):
    assert main(["methods", "--json"]) == 0
    catalogue = json.loads(capsys.readouterr().out)
    assert catalogue == method_catalog()
    assert {row["profile_id"] for row in catalogue} == set(METHODS)
    attention = next(row for row in catalogue if row["name"] == "attention_query")
    assert attention["profile_id"] == "single" and "attention" in attention["description"]


@pytest.mark.parametrize("alias,stored", METHOD_ALIASES.items())
def test_method_alias_has_exact_stored_recipe(alias, stored, limits):
    assert recipe(alias, limits) == recipe(stored, limits)
    assert standard_settings(alias, device="cpu") == standard_settings(stored, device="cpu")


@pytest.mark.parametrize("alias,stored", METHOD_ALIASES.items())
def test_public_alias_fit_and_propose_preserves_profile_identity(
    dataset, tmp_path, settings, limits, alias, stored
):
    if stored == "single":
        pytest.importorskip("torch")
        settings.pop("classifier")
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
    model = fit(
        dataset, tmp_path / "model", method=alias, settings=settings, limits=limits, cache=tmp_path / "cache"
    )
    assert model.manifest["method"] == stored
    assert model.manifest["recipe"]["method"] == stored
    if stored == "single":
        assert "torch_execution" in model.manifest["runtime"]
    proposals = propose(model, dataset, tmp_path / "proposals", limits=limits, cache=tmp_path / "cache")
    assert proposals.manifest["method"] == stored
    assert proposals.manifest["proposal_count"] > 0


def test_aliases_resolve_global_and_compact_overrides(dataset, tmp_path, limits):
    path = tmp_path / "global.json"
    save(
        path,
        {
            "schema_version": 1,
            "device": "cpu",
            "enabled_methods": ["hgb_composition_shortlist"],
            "search": {"max_proposals": 128},
            "methods": {"hgb_composition_shortlist": {"search": {"params": {"max_proposals": 128}}}},
        },
    )
    values, _ = load_defaults(path)
    assert values["enabled_methods"] == ["kmers"]
    assert values["search"]["max_checks"] == 128
    raw = {
        "schema_version": 2,
        "global_config": str(path),
        "output": "results",
        "datasets": {"sample": {"prepared": str(dataset.path)}},
        "methods": [
            {
                "method": "hgb_composition_shortlist",
                "settings": {
                    "search": {"name": "expanded_ranked_alternatives", "params": {"max_proposals": 64}}
                },
            }
        ],
    }
    expanded, settings, _ = expand_experiment(raw, tmp_path)
    assert expanded["methods"][0]["method"] == "kmers"
    save(tmp_path / "experiment.json", raw)
    plan = load_plan(tmp_path / "experiment.json")
    resolved = plan.document["methods"][0]
    assert resolved["recipe"]["search"]["params"]["max_checks"] == 64
    assert "max_proposals" not in resolved["recipe"]["search"]["params"]
    assert recipe("kmers", limits, settings["kmers"])["search"]["params"]["max_checks"] == 128
    standalone = raw | {"schema_version": 1, "training": "sample", "evaluation": "sample"}
    standalone.pop("global_config")
    save(tmp_path / "standalone.json", standalone)
    assert load_plan(tmp_path / "standalone.json").document["methods"][0]["method"] == "kmers"
    initialized = initialize_compact(tmp_path / "compact", methods=["attention_query"])
    assert read(initialized)["methods"] == ["attention_query"]


@pytest.mark.parametrize("search,default", [(RankedAlternatives, 20), (ExpandedRankedAlternatives, 128)])
def test_explicit_proposal_cap_is_an_exact_compatibility_alias(search, default):
    store = SimpleNamespace(
        ids=np.array([f"c{i}" for i in range(81)]),
        lengths=np.array([100000] + [1000] * 80),
        protected=lambda: {"c0"},
    )
    core, ranked = np.array([0]), np.arange(1, 81)
    old, preferred = search(max_checks=default), search(max_proposals=default)
    assert list(old.propose(core, ranked, store)) == list(preferred.propose(core, ranked, store))
    assert preferred.max_proposals == preferred.max_checks == default
    with pytest.raises(ValueError, match="must agree"):
        search(max_checks=20, max_proposals=21)


def test_conflicting_aliases_fail_before_any_work(tmp_path, limits):
    with pytest.raises(ValueError, match="must agree"):
        recipe("cosine", limits, {"search": {"params": {"max_checks": 20, "max_proposals": 128}}})
    path = tmp_path / "global.json"
    save(path, {"enabled_methods": ["single", "attention_query"]})
    with pytest.raises(ValueError, match="unique"):
        load_defaults(path)
