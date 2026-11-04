"""Check public model class paths, cache configuration and CLI identification."""

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
        ("core.types", "FeatureBatch"),
    ],
)
def test_saved_class_references_resolve_to_canonical_implementations(module, name):
    reference = f"cbincomplete.{module}\n{name}\n.".encode("ascii")
    canonical = getattr(importlib.import_module(f"bincomplete.{module}"), name)
    assert pickle.loads(reference) is canonical
    assert canonical.__module__.startswith("bincomplete.")


def test_cache_configuration_uses_public_environment_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("BINCOMPLETE_CACHE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_cache() == tmp_path / "bincomplete"
    monkeypatch.setenv("BINCOMPLETE_CACHE", str(tmp_path / "custom"))
    assert default_cache() == tmp_path / "custom"


def test_cli_identifies_bincomplete(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.startswith("BinComplete ")
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "usage: bincomplete" in capsys.readouterr().out
