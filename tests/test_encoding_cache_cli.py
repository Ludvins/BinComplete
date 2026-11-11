import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bincomplete.artifacts import fingerprint, save
from bincomplete.cache import DerivedCache
from bincomplete.cli import main
from bincomplete.data import prepare_dataset
from bincomplete.preflight import preflight
from bincomplete.resources import FileLock, ResourceLimits


@pytest.fixture
def encode_args(tmp_path):
    assembly = tmp_path / "assembly.fa"
    assembly.write_text(">z description\nACGT\n>a\nGGCC\n", encoding="utf-8")
    model = tmp_path / "weights"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    save(model / "config.json", {"hidden_size": 2})
    config = tmp_path / "global.json"
    save(config, {"encoder_path": "weights", "cache": "cache"})
    return [
        "encode",
        "--assembly",
        str(assembly),
        "--output",
        str(tmp_path / "embeddings"),
        "--global-config",
        str(config),
    ]


@pytest.mark.parametrize("rename_files", [False, True], ids=["original", "renamed"])
def test_encode_bundle_keeps_fasta_rows_and_survives_cache_clean(
    tmp_path, monkeypatch, capsys, encode_args, rename_files
):
    from bincomplete import encoding

    assembly = tmp_path / "assembly.fa"
    output = tmp_path / "embeddings"
    calls = []

    def synthetic_encoder(context, spec):
        windows = list(context.store.sequence_windows([1, 0], window_bp=3, count=1))
        assert windows == [[b"GGC"], [b"ACG"]]
        assert spec["execution"]["device"] == "cpu"
        calls.append(windows)
        return context.cache.array(
            {"kind": "synthetic_encoder"},
            lambda path: np.save(path, np.eye(2, dtype=np.float16)),
            expected_bytes=8,
        )

    monkeypatch.setattr(encoding, "frozen_array", synthetic_encoder)
    assert main(encode_args) == 0
    assert main(encode_args) == 0
    assert len(calls) == 1
    assert (output / "embedding_ids.txt").read_text(encoding="utf-8").splitlines() == ["z", "a"]
    capsys.readouterr()
    assert main(["cache", "clean", "--global-config", str(tmp_path / "global.json"), "--apply"]) == 0
    assert len(json.loads(capsys.readouterr().out)["entries"]) == 1
    assert not any((tmp_path / "cache" / "arrays").iterdir())
    memberships = tmp_path / "members.tsv"
    memberships.write_text("contig_id\tbin_id\nz\tbin\n", encoding="utf-8")
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text("contig_id\tmean\nz\t3\na\t2\n", encoding="utf-8")
    inputs = {
        "assembly": assembly,
        "memberships": memberships,
        "coverage": coverage,
        "embeddings": output / "embeddings.npy",
        "embedding_ids": output / "embedding_ids.txt",
        "embedding_space": output / "embedding_space.json",
    }
    if rename_files:
        inputs["embeddings"] = inputs["embeddings"].rename(output / "vectors.npy")
        inputs["embedding_ids"] = inputs["embedding_ids"].rename(output / "row_ids.txt")
        inputs["embedding_space"] = inputs["embedding_space"].rename(output / "space.json")
    dataset = prepare_dataset(tmp_path / "prepared", **inputs)
    assert dataset.verify()["embedding"]["space"]["dimension"] == 2
    assert str(output / "complete.json") in dataset.verify()["inputs_sha256"]
    later = tmp_path / "later"
    init = ["init", "--device", "cpu", "--methods", "cosine", "--output", str(later)]
    for name, path in inputs.items():
        init.extend((f"--{name.replace('_', '-')}", str(path)))
    assert main(init) == 0

    original = assembly.read_bytes()
    receipt = (output / "complete.json").read_bytes()
    assembly.write_bytes(original.replace(b"ACGT", b"TGCA"))
    assert "Embedding bundle belongs to a different assembly" in str(
        preflight(later / "experiment.json")["errors"]
    )
    with pytest.raises(ValueError, match="Embedding bundle belongs to a different assembly"):
        prepare_dataset(tmp_path / "mismatched", **inputs)
    assert main(encode_args) == 2
    assert "Embedding output belongs to another request" in capsys.readouterr().err
    assert (output / "complete.json").read_bytes() == receipt
    assert len(calls) == 1
    assembly.write_bytes(original)
    inputs["embedding_ids"].write_text("a\nz\n", encoding="utf-8")
    mismatch = f"Embedding bundle file changed: {inputs['embedding_ids'].name}"
    assert mismatch in str(preflight(later / "experiment.json")["errors"])
    with pytest.raises(ValueError, match=mismatch):
        prepare_dataset(tmp_path / "mismatched", **inputs)
    assert main(encode_args) == 2
    assert "Artifact output changed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("assembly.fa", "Assembly changed during encoding"),
        ("weights/pytorch_model.bin", "Encoder file changed during encoding"),
    ],
)
def test_encode_retries_without_reusing_a_cache_entry_from_changed_inputs(
    tmp_path, monkeypatch, capsys, encode_args, filename, message
):
    from bincomplete import encoding

    source = tmp_path / filename
    original = source.read_bytes()
    changed = original.replace(b"ACGT", b"TGCA") if filename == "assembly.fa" else original.upper()
    writes = []

    def synthetic_encoder(context, spec):
        def write(path):
            np.save(path, np.eye(2, dtype=np.float16))
            if not writes:
                source.write_bytes(changed)
            writes.append(path)

        return context.cache.array({"kind": "synthetic_encoder"}, write, expected_bytes=8)

    monkeypatch.setattr(encoding, "frozen_array", synthetic_encoder)
    assert main(encode_args) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "embeddings").exists()
    assert not list((tmp_path / "cache" / "arrays").glob("*/complete.json"))
    source.write_bytes(original)
    assert main(encode_args) == 0
    assert len(writes) == 2
    assert (tmp_path / "embeddings" / "complete.json").is_file()


def test_encode_retry_after_failed_export_reuses_completed_vectors(
    tmp_path, monkeypatch, capsys, encode_args
):
    from bincomplete import encoding

    writes = []

    def synthetic_encoder(context, spec):
        def write(path):
            writes.append(path)
            np.save(path, np.eye(2, dtype=np.float16))

        return context.cache.array({"kind": "synthetic_encoder"}, write, expected_bytes=8)

    def fail_copy(source, destination):
        destination.write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(encoding, "frozen_array", synthetic_encoder)
    with monkeypatch.context() as failing_export:
        failing_export.setattr(encoding.shutil, "copyfile", fail_copy)
        assert main(encode_args) == 2
    assert "simulated write failure" in capsys.readouterr().err
    assert not (tmp_path / "embeddings").exists()
    assert not list(tmp_path.glob(".embeddings.encode-*"))
    assert main(encode_args) == 0
    assert len(writes) == 1
    assert (tmp_path / "embeddings" / "complete.json").is_file()


def test_cache_clean_respects_leases_unknown_files_and_interrupted_entries(tmp_path, capsys):
    root = tmp_path / "cache"
    active = DerivedCache(root, ResourceLimits())
    legacy = DerivedCache(root, ResourceLimits())
    legacy_id = {"kind": "local_dnabert_s_v2"}
    active.array({"active": True}, lambda path: np.save(path, np.ones(2)), expected_bytes=16)
    legacy.array(legacy_id, lambda path: np.save(path, np.ones(2)), expected_bytes=16)
    legacy.close()
    arrays = root / "arrays"
    legacy_dir = arrays / fingerprint(legacy_id)
    save(
        legacy_dir / "encoder.checkpoint.progress.json",
        {
            "phase": "encoding",
            "completed_contigs": 2,
            "total_contigs": 2,
        },
    )
    (legacy_dir / "encoder.checkpoint.progress.json.12345.partial").write_text("{", encoding="utf-8")
    interrupted = DerivedCache(root, ResourceLimits())

    def interrupted_writer(path):
        path.with_name("encoder.checkpoint.npy").write_bytes(b"unfinished")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.array({"unfinished": True}, interrupted_writer, expected_bytes=16)
    unfinished = arrays / fingerprint({"unfinished": True})
    foreign = arrays / ("f" * 64)
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep", encoding="utf-8")
    unmarked = arrays / ("e" * 64)
    unmarked.mkdir()
    (unmarked / "encoder.checkpoint.npy").write_bytes(b"unknown owner")
    args = ["cache", "clean", "--cache", str(root)]
    try:
        assert main(args) == 0
        report = json.loads(capsys.readouterr().out)
        assert {entry["key"] for entry in report["entries"]} == {legacy_dir.name}
        assert report["skipped_in_use"] == report["skipped_incomplete"] == 1
        assert legacy_dir.exists() and unfinished.exists()
        assert main([*args, "--apply"]) == 0
        assert not legacy_dir.exists() and unfinished.exists()
        assert main([*args, "--apply", "--include-incomplete"]) == 0
        assert not unfinished.exists() and (foreign / "notes.txt").is_file()
        assert (unmarked / "encoder.checkpoint.npy").is_file()
        assert (arrays / fingerprint({"active": True}) / "values.npy").is_file()
    finally:
        active.close()
        interrupted.close()


def test_cache_clean_cannot_apply_while_another_process_holds_the_quota_lock(tmp_path):
    root = tmp_path / "cache"
    cache = DerivedCache(root, ResourceLimits())
    cache.array({"ready": True}, lambda path: np.save(path, np.ones(2)), expected_bytes=16)
    cache.close()
    values = next((root / "arrays").glob("*/values.npy"))
    command = [sys.executable, "-B", "-m", "bincomplete", "cache", "clean", "--cache", str(root), "--apply"]

    with FileLock(root / "quota.lock"):
        blocked = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        assert blocked.returncode == 2, blocked.stderr
        assert "Cache is busy" in blocked.stderr
        assert "preflight" not in blocked.stderr
        assert values.is_file()

    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    assert len(json.loads(result.stdout)["entries"]) == 1
    assert not values.exists()


@pytest.mark.parametrize("blocked_name", ["complete.json", "values.npy"])
def test_failed_cache_clean_can_retry_without_stranding_the_array(tmp_path, monkeypatch, blocked_name):
    root = tmp_path / "cache"
    cache = DerivedCache(root, ResourceLimits())

    def write(path):
        np.save(path, np.ones(2))

    identity = {"test": "cleanup recovery"}
    folder = Path(cache.array(identity, write, expected_bytes=16).filename).parent
    cache.close()
    unlink = Path.unlink

    def locked_file(path, *args, **kwargs):
        if path == folder / blocked_name:
            raise PermissionError("simulated locked cache file")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "unlink", locked_file)
        assert main(["cache", "clean", "--cache", str(root), "--apply"]) == 2
    assert (folder / "values.npy").is_file()
    assert (folder / "complete.json").is_file() == (blocked_name == "complete.json")
    assert main(["cache", "clean", "--cache", str(root), "--apply", "--include-incomplete"]) == 0
    assert not folder.exists()
    assert cache.array(identity, write, expected_bytes=16).tolist() == [1.0, 1.0]
    cache.close()


@pytest.fixture
def checkpoint_encoder(tmp_path, monkeypatch, encode_args):
    pytest.importorskip("torch")
    from bincomplete.representations import strand_pooling

    assembly = tmp_path / "assembly.fa"
    assembly.write_text("".join(f">c{i}\nACGT\n" for i in range(17)), encoding="utf-8")
    save(tmp_path / "weights" / "config.json", {"hidden_size": 768})

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            return self

    monkeypatch.setattr(strand_pooling, "load_local_encoder", lambda _: (None, Model()))
    return assembly, strand_pooling


def test_encode_resumes_within_original_cache_quota(tmp_path, monkeypatch, encode_args, checkpoint_encoder):
    _, pooling = checkpoint_encoder
    calls = []

    def encode_group(tokenizer, model, group, space, execution):
        calls.append(len(group))
        if len(calls) == 2:
            raise OSError("interrupted")
        vectors = np.zeros((len(group), space["dimension"]), dtype=np.float16)
        vectors[:, 0] = 1
        return vectors

    monkeypatch.setattr(pooling, "pooled_windows", encode_group)
    args = [*encode_args, "--cache-gib", str(38000 / 1024**3)]
    assert main(args) == 2
    assert not (tmp_path / "embeddings").exists()
    bitmap = next((tmp_path / "cache" / "arrays").glob("*/encoder.done.npy"))
    partial = bitmap.with_name("encoder.checkpoint.progress.json.12345.partial")
    partial.write_text("{", encoding="utf-8")
    assert main(args) == 0
    assert calls == [16, 1, 1]
    assert not partial.exists()


def test_interrupted_encode_never_commits_a_group_from_changed_assembly(
    tmp_path, monkeypatch, capsys, encode_args, checkpoint_encoder
):
    assembly, pooling = checkpoint_encoder
    original = assembly.read_bytes()
    calls = []

    def encode_group(tokenizer, model, group, space, execution):
        calls.append(len(group))
        if len(calls) == 1:
            assembly.write_bytes(original.replace(b"ACGT", b"TGCA", 1))
        vectors = np.zeros((len(group), space["dimension"]), dtype=np.float16)
        vectors[:, 1 if len(calls) == 1 else 0] = 1
        return vectors

    monkeypatch.setattr(pooling, "pooled_windows", encode_group)
    assert main(encode_args) == 2
    assert "Encoder input changed during encoding" in capsys.readouterr().err
    done_path = next((tmp_path / "cache" / "arrays").glob("*/encoder.done.npy"))
    assert not np.load(done_path).any()
    assembly.write_bytes(original)
    assert main(encode_args) == 0
    assert calls == [16, 16, 1]
    assert np.load(tmp_path / "embeddings" / "embeddings.npy")[0, :2].tolist() == [1.0, 0.0]
