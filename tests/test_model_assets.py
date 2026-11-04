"""Pinned downloads, retry preservation and offline reuse without real model jobs."""

from pathlib import Path

import pytest

from bincomplete.defaults import DNABERT_REPO, DNABERT_REVISION
from bincomplete.model_assets import install_dnabert


def test_offline_inspection_exports_actual_contract_without_executing_model(tmp_path, capsys):
    from bincomplete.artifacts import fingerprint, read, sha
    from bincomplete.cli import main
    from bincomplete.representations.dna import local_contract

    model = tmp_path / "model"
    checkpoint(model)
    options = tmp_path / "options.json"
    options.write_text(
        '{"pooling":"non_special_bp_weighted_windows","reverse_complement":true,"max_tokens":2048}',
        encoding="utf-8",
    )
    output = tmp_path / "space.json"
    arguments = [
        "models",
        "inspect",
        "dnabert_s",
        "--path",
        str(model),
        "--encoder-options",
        str(options),
        "--output",
        str(output),
    ]
    before = {p.name: sha(p) for p in model.iterdir()}
    assert main(arguments) == 0
    assert read(output) == local_contract(model, read(options))[0]
    assert read(output)["reverse_complement"] and read(output)["dimension"] == 8
    assert read(output)["encoder_sha256"] == fingerprint(
        {name: before[name] for name in ("config.json", "pytorch_model.bin", "bert_layers.py")}
    )
    assert read(output)["tokenizer_sha256"] == fingerprint({"tokenizer.json": before["tokenizer.json"]})
    assert main(arguments) == 0
    capsys.readouterr()
    (model / "pytorch_model.bin").write_bytes(b"changed")
    assert main(arguments) == 2
    assert "Immutable artifact identity changed" in capsys.readouterr().err


def checkpoint(target):
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text('{"hidden_size":8}', encoding="utf-8")
    (target / "tokenizer.json").write_text('{"synthetic":true}', encoding="utf-8")
    (target / "pytorch_model.bin").write_bytes(b"synthetic-checkpoint-not-used-for-inference")
    (target / "bert_layers.py").write_text(
        'raise RuntimeError("model code must not execute during download")', encoding="utf-8"
    )


def test_pinned_install_is_idempotent_and_receipt_is_external(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(tmp_path / "absent.toml"))
    calls = []

    def download(**values):
        calls.append(values)
        checkpoint(Path(values["local_dir"]))

    monkeypatch.setattr(hub, "snapshot_download", download)
    target = tmp_path / "model"
    first = install_dnabert(target)
    second = install_dnabert(target)
    assert len(calls) == 1 and calls[0]["repo_id"] == DNABERT_REPO
    assert calls[0]["revision"] == DNABERT_REVISION and calls[0]["max_workers"] == 2
    assert not first["reused"] and second["reused"]
    assert (tmp_path / "model.download.json").exists()
    assert not (target / "model.download.json").exists()
    (target / "pytorch_model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        install_dnabert(target)


def test_user_managed_checkpoint_is_never_overwritten(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: pytest.fail("unexpected download"))
    checkpoint(tmp_path / "existing")
    result = install_dnabert(tmp_path / "existing")
    assert result["status"] == "EXISTING_LOCAL_MODEL_REUSED" and not result["downloaded"]
    assert not (tmp_path / "existing.download.json").exists()


def test_failed_attempt_is_preserved_and_only_same_revision_can_retry(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("BINCOMPLETE_CONFIG", str(tmp_path / "absent.toml"))
    target = tmp_path / "model"

    def failed(**kwargs):
        target.mkdir()
        (target / "partial").write_text("preserve", encoding="utf-8")
        raise OSError("synthetic connection failure")

    monkeypatch.setattr(hub, "snapshot_download", failed)
    with pytest.raises(OSError, match="connection failure"):
        install_dnabert(target)
    assert (target / "partial").read_text() == "preserve"
    assert (tmp_path / "model.download-attempt.json").exists()
    with pytest.raises(ValueError, match="different contract"):
        install_dnabert(target, revision="f" * 40)
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: checkpoint(target))
    assert install_dnabert(target)["status"] == "MODEL_INSTALLED"
    assert (target / "partial").read_text() == "preserve"


def test_model_revision_cannot_float_with_main(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        install_dnabert(tmp_path / "model", revision="main")
