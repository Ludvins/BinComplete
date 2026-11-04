"""Portable recipe, inherited proposals and environment diagnostics contracts."""

import json
from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pytest

from bincomplete import fit, propose
from bincomplete.artifacts import read, sha
from bincomplete.cli import main
from bincomplete.context import Context
from bincomplete.environment import environment_report
from bincomplete.methods import recipe
from bincomplete.search.expanded import ExpandedRankedAlternatives
from bincomplete.search.ranked_alternatives import RankedAlternatives
from bincomplete.standard import STANDARD_METHODS, initialize, standard_settings


def test_standard_is_packaged_and_has_six_uniform_searches(tmp_path, limits):
    assert main(["init", "--output", str(tmp_path / "work"), "--device", "cpu"]) == 0
    config = read(tmp_path / "work/experiment.json")
    assert [row["method"] for row in config["methods"]] == list(STANDARD_METHODS)
    assert config["resources"]["threads"] == 2
    assert config["resources"]["memory_gib"] == 6
    for entry in config["methods"]:
        settings = read(tmp_path / "work" / entry["settings"])
        spec = recipe(entry["method"], limits, settings)
        assert spec["search"] == {
            "name": "expanded_ranked_alternatives",
            "params": {
                "max_checks": 128,
                "singleton_depth": 50,
                "boundary_width": 2,
                "max_added_fraction": 1.0,
            },
        }
        assert spec["seed"] == 73019
        assert spec["preprocessing"]["coverage_dtype"] == "float32"
    assert standard_settings("single", device="cpu")["scorer"]["params"]["device"] == "cpu"
    assert standard_settings("single", device="cuda")["scorer"]["params"]["device"] == "cuda"
    with pytest.raises(FileExistsError):
        initialize(tmp_path / "work")
    # Relative inputs/settings/output, including the encoder path, contain no drive IDs.
    path = initialize(tmp_path / "encode", embedding_mode="encode")
    assert read(path)["datasets"]["sample"]["encoder_options"]["batch_size"] == 8
    assert "R:" not in path.read_text() and "M:" not in path.read_text()


def test_reference_dependencies_and_profile_copies():
    root = files("bincomplete").joinpath("profiles/standard_v1")
    environment = json.loads(root.joinpath("tested_environment.json").read_text())
    assert environment["schema_version"] == 1
    assert environment["packages"]["numpy"]
    assert set(environment) == {"schema_version", "recipe", "python", "packages", "scope"}
    # Each call returns an independent object, so callers cannot mutate shipped defaults.
    settings = standard_settings("hgb_full_pool")
    settings["seed"] = 1
    assert standard_settings("hgb_full_pool")["seed"] == 73019


def test_expanded_search_retains_legacy_and_guards_ownership():
    store = SimpleNamespace(
        ids=np.array([str(i) for i in range(81)]),
        lengths=np.array([100000] + [1000] * 80),
        protected=lambda: {"0"},
    )
    core, ranked = np.array([0]), np.arange(1, 81)
    old = list(RankedAlternatives().propose(core, ranked, store))
    new = list(ExpandedRankedAlternatives().propose(core, ranked, store))

    def memberships(proposals):
        return {tuple(sorted(row["indices"])) for row in proposals}

    assert memberships(old) <= memberships(new)
    assert len(new) <= 128 and len(new) > len(old)
    assert len(new) == len(memberships(new))
    assert all(set(core) <= set(row["indices"]) and row["added_bp"] <= 100000 for row in new)
    assert any(row.get("donor_ranks") == [50] for row in new)
    with pytest.raises(ValueError, match="Protected donor"):
        list(ExpandedRankedAlternatives().propose(core, np.array([0, 1]), store))
    with pytest.raises(ValueError, match="Duplicate"):
        list(ExpandedRankedAlternatives().propose(core, np.array([1, 1]), store))


def test_named_search_is_saved_and_legacy_model_remains_immutable(dataset, tmp_path, limits, settings):
    # Preserve the explicit lower-level legacy profile and validate compatibility
    # with the new search registration; old manifests are never relabelled.
    cache = tmp_path / "cache"
    model = fit(dataset, tmp_path / "legacy", method="kmers", settings=settings, limits=limits, cache=cache)
    before = sha(model.path / "model.json")
    legacy = propose(model, dataset, tmp_path / "legacy_proposals", limits=limits, cache=cache).manifest
    assert legacy["search"]["name"] == "ranked_alternatives"
    assert max(row["proposals"] for row in legacy["coverage"].values()) <= 20
    updated = dict(settings, search=standard_settings("kmers")["search"])
    candidate = fit(
        dataset, tmp_path / "standard", method="kmers", settings=updated, limits=limits, cache=cache
    )
    result = propose(candidate, dataset, tmp_path / "standard_proposals", limits=limits, cache=cache).manifest
    assert result["search"]["name"] == "expanded_ranked_alternatives"
    assert max(row["proposals"] for row in result["coverage"].values()) <= 128
    assert sha(model.path / "model.json") == before
    context = Context(dataset, candidate.manifest["recipe"], tmp_path / "context", cache, limits)
    try:
        assert context.compatibility() == model.manifest["compatibility"]
    finally:
        context.close()


def test_doctor_records_only_relevant_packages_and_never_overwrites(tmp_path):
    report = environment_report("cpu")
    assert report["packages"]["numpy"]
    assert "PATH" not in report and "environment_variables" not in report
    destination = tmp_path / "environment.json"
    assert main(["doctor", "--output", str(destination)]) == 0
    assert read(destination)["python"] == report["python"]
    before = destination.read_bytes()
    assert main(["doctor", "--output", str(destination)]) == 2
    assert destination.read_bytes() == before
