"""The documented production profiles must fit new models through the normal CLI."""

import json

import pytest

from bincomplete.cli import main
from bincomplete.engine import ModelArtifact, ProposalArtifact
from bincomplete.methods import METHODS
from bincomplete.standard import standard_settings


@pytest.mark.parametrize("method", METHODS)
def test_fresh_profile_runs_without_historical_artifacts(dataset, tmp_path, method):
    if method == "single":
        pytest.importorskip("torch")
    # CI has an explicit CPU profile; do not silently fall back from the configured GPU.
    profile = tmp_path / (method + ".json")
    profile.write_text(json.dumps(standard_settings(method, device="cpu")), encoding="utf-8")
    output = tmp_path / method
    assert (
        main(
            [
                "run",
                "--training",
                str(dataset.path),
                "--dataset",
                str(dataset.path),
                "--method",
                method,
                "--settings",
                str(profile),
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
                "--threads",
                "2",
                "--memory-gib",
                "0.5",
                "--cache-gib",
                "0.01",
            ]
        )
        == 0
    )
    model = ModelArtifact(output / "model").manifest
    proposals = ProposalArtifact(output / "proposals").manifest
    assert model["learned"] == (method != "cosine")
    assert proposals["proposal_count"] > 0
    assert model["recipe"]["training"]["max_training_rows"] == 8000000
    if method != "cosine":
        assert model["training_statistics"]["sources"] == 4
    json.dumps(model)
