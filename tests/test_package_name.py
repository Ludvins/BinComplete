"""Guard the package rename's saved-model and configuration boundaries."""

import importlib
import pickle

import pytest

from bincomplete.cache import default_cache
from bincomplete.cli import main


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("classifiers.sklearn_models", "SklearnClassifier"),
        ("scorers.vector", "FeatureClassifier"),
        ("scorers.centroid", "CentroidQueries"),
        ("scorers.sets", "SetScorer"),
        ("pipeline.training", "ModelBundle"),
        ("core.types", "FeatureBatch"),
    ],
)
def test_saved_class_references_resolve_to_canonical_implementations(module, name):
    # Protocol-zero GLOBAL records use the same historical class paths as joblib.
    reference = f"cbin_recruitment.{module}\n{name}\n.".encode("ascii")
    canonical = getattr(importlib.import_module(f"bincomplete.{module}"), name)
    assert pickle.loads(reference) is canonical
    assert canonical.__module__.startswith("bincomplete.")


def test_cache_configuration_prefers_new_name_and_accepts_old_name(monkeypatch, tmp_path):
    monkeypatch.delenv("BINCOMPLETE_CACHE", raising=False)
    monkeypatch.delenv("BIN_RECRUITMENT_CACHE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_cache() == tmp_path / "bincomplete"
    monkeypatch.setenv("BIN_RECRUITMENT_CACHE", str(tmp_path / "old"))
    assert default_cache() == tmp_path / "old"
    monkeypatch.setenv("BINCOMPLETE_CACHE", str(tmp_path / "new"))
    assert default_cache() == tmp_path / "new"


def test_cli_identifies_bincomplete(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.startswith("BinComplete ")
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "usage: bincomplete" in capsys.readouterr().out
