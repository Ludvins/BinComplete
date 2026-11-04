"""Shared settings resolve explicitly, freeze per run and never alter schema 1."""

import json
from pathlib import Path

import pytest

from bincomplete import run_experiment
from bincomplete.artifacts import read, save
from bincomplete.cli import main
from bincomplete.defaults import (
    DNABERT_REVISION,
    config_path,
    load_defaults,
    method_defaults,
    write_defaults,
)
from bincomplete.experiments.config import load_plan
from bincomplete.standard import initialize_compact, standard_settings


def configuration(tmp_path, dataset, monkeypatch):
    global_path = tmp_path / "global.json"
    values = {
        "schema_version": 1,
        "device": "cpu",
        "cache": str(tmp_path / "cache"),
        "resources": {"threads": 1, "memory_gib": 0.5, "cache_gib": 0.01, "chunk_size": 3},
    }
    save(global_path, values)
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(global_path))
    config = tmp_path / "experiment.json"
    save(
        config,
        {"schema_version": 2, "output": "results", "datasets": {"sample": {"prepared": str(dataset.path)}}},
    )
    return config, global_path


def test_write_global_toml_and_show(tmp_path, monkeypatch, capsys):
    path = tmp_path / "installation/config.toml"
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(path))
    assert config_path() == path
    assert main(["config", "init", "--encoder-path", str(tmp_path / "weights"), "--device", "cuda:0"]) == 0
    capsys.readouterr()
    values, proof = load_defaults()
    assert values["device"] == "cuda:0" and Path(values["encoder_path"]) == tmp_path / "weights"
    assert values["search"]["max_checks"] == 128
    assert values["search"]["singleton_depth"] == 50
    assert values["methods"]["hgb_full_pool"]["classifier"]["params"]["max_leaf_nodes"] == 7
    assert proof["sha256"] and proof["path"] == str(path)
    assert main(["config", "show", "--resolved"]) == 0
    view = json.loads(capsys.readouterr().out)
    assert len(view["method_recipes"]) == 6
    assert view["method_recipes"]["single"]["scorer"]["params"]["device"] == "cuda"
    with pytest.raises(FileExistsError):
        write_defaults()


def test_compact_inherits_all_six_and_global_overrides(tmp_path, dataset, monkeypatch):
    config, global_path = configuration(tmp_path, dataset, monkeypatch)
    original = load_plan(config).document
    assert len(original["methods"]) == 6
    assert all(row["recipe"]["search"]["params"]["max_checks"] == 128 for row in original["methods"])
    global_value = read(global_path)
    global_value.update(
        search={"max_checks": 256, "singleton_depth": 100},
        methods={"hgb_full_pool": {"classifier": {"params": {"max_iter": 200}}}},
    )
    save(global_path, global_value)
    value = read(config)
    value["methods"] = [
        {"method": "hgb_full_pool", "settings": {"classifier": {"params": {"max_iter": 250}}}}
    ]
    value["resources"] = {"threads": 2}
    save(config, value)
    updated = load_plan(config).document
    assert updated["resources"]["threads"] == 2
    assert updated["methods"][0]["recipe"]["classifier"]["params"]["max_iter"] == 250
    assert updated["methods"][0]["recipe"]["search"]["params"]["max_checks"] == 256
    assert updated["defaults"]["sha256"] != original["defaults"]["sha256"]


def test_global_relative_paths_and_explicit_file_precedence(tmp_path, dataset, monkeypatch):
    config, _ = configuration(tmp_path, dataset, monkeypatch)
    other = tmp_path / "installation"
    other.mkdir()
    save(other / "global.json", {"encoder_path": "weights", "cache": "arrays", "device": "cpu"})
    value = read(config)
    value["global_config"] = "installation/global.json"
    value["methods"] = ["cosine"]
    save(config, value)
    plan = load_plan(config).document
    assert plan["cache"] == str(other / "arrays")
    assert plan["defaults"]["resolved"]["encoder_path"] == str(other / "weights")


def test_schema_one_is_insulated_from_global_settings(tmp_path, dataset, monkeypatch):
    config, global_path = configuration(tmp_path, dataset, monkeypatch)
    value = read(config)
    value.update(schema_version=1, training="sample", evaluation="sample", methods=[{"method": "cosine"}])
    save(config, value)
    first = load_plan(config).document
    # Invalid ambient globals must also be ignored by an existing explicit recipe.
    global_path.write_text("not json", encoding="utf-8")
    second = load_plan(config).document
    assert first == second and "defaults" not in second
    assert second["methods"][0]["recipe"]["search"]["params"]["max_checks"] == 20


def test_global_change_blocks_completed_resume(tmp_path, dataset, monkeypatch):
    config, global_path = configuration(tmp_path, dataset, monkeypatch)
    value = read(config)
    value["methods"] = ["cosine"]
    save(config, value)
    result = run_experiment(config)
    assert result.manifest["status"] == "SUCCEEDED"
    assert run_experiment(config).manifest["status"] == "SUCCEEDED"
    frozen = (result.path / "experiment.json").read_bytes()
    globals_value = read(global_path)
    globals_value["search"] = {"singleton_depth": 60}
    save(global_path, globals_value)
    with pytest.raises(ValueError, match="identity changed"):
        run_experiment(config)
    assert (result.path / "experiment.json").read_bytes() == frozen


def test_compact_init_declares_only_dataset_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(tmp_path / "not_created.json"))
    path = initialize_compact(tmp_path / "sample", input_root=tmp_path / "original")
    value = read(path)
    assert value["schema_version"] == 2 and "methods" not in value
    assert value["datasets"]["sample"]["input_root"] == str(tmp_path / "original")
    assert not (path.parent / "settings").exists()
    assert main(["init", "--input-root", str(tmp_path / "original"), "--output", str(tmp_path / "cli")]) == 0
    assert "device" not in read(tmp_path / "cli/experiment.json")


def test_compact_default_encoder_options_and_cpu_variant(tmp_path, dataset, monkeypatch):
    config, global_path = configuration(tmp_path, dataset, monkeypatch)
    values = read(global_path)
    values.update(device="cuda:0", encoder_path=str(tmp_path / "weights"))
    save(global_path, values)
    # Expansion alone does not load a model or require the input paths to exist.
    from bincomplete.defaults import expand_experiment

    raw = {
        "schema_version": 2,
        "datasets": {
            "sample": {"assembly": "assembly.fa", "memberships": "members.tsv", "coverage": "coverage.tsv"}
        },
    }
    expanded, settings, _ = expand_experiment(raw, tmp_path)
    assert expanded["datasets"]["sample"]["encoder_options"]["precision"] == "bfloat16"
    assert settings["single"]["scorer"]["params"]["device"] == "cuda"
    raw["device"] = "cpu"
    expanded, settings, _ = expand_experiment(raw, tmp_path)
    assert expanded["datasets"]["sample"]["encoder_options"]["precision"] == "float32"
    assert settings["single"]["scorer"]["params"]["device"] == "cpu"
    assert DNABERT_REVISION in str(values.get("encoder_path", "")) or values["encoder_path"].endswith(
        "weights"
    )


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": True},
        {"device": "auto"},
        {"preset": "typo"},
        {"search": {"typo": 1}},
        {"enabled_methods": ["cosine", "cosine"]},
        {"methods": {"vae": {}}},
    ],
)
def test_invalid_global_settings_fail_clearly(tmp_path, value):
    path = tmp_path / "bad.json"
    save(path, value)
    with pytest.raises(ValueError):
        load_defaults(path)


def test_preset_values_match_standalone_at_same_device(tmp_path):
    path = write_defaults(tmp_path / "global.toml", device="cuda")
    values, _ = load_defaults(path)
    inherited = method_defaults(values)
    for method, settings in inherited.items():
        assert settings == standard_settings(method, device="cuda")
