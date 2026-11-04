import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bincomplete
from bincomplete import prepare_dataset
from bincomplete.artifacts import save
from bincomplete.context import Context
from bincomplete.methods import recipe


def test_public_operation_docs_and_annotations():
    for name in ("prepare_dataset", "fit", "propose", "export", "reconcile", "run", "run_experiment"):
        function = getattr(bincomplete, name)
        doc = inspect.getdoc(function)
        assert "Parameters\n----------" in doc
        assert "Returns\n-------" in doc
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(
            parameter.annotation is not inspect.Parameter.empty for parameter in signature.parameters.values()
        )


def test_package_has_no_evaluator_or_tool_invocation_modules():
    package = Path(bincomplete.__file__).parent
    assert not (package / "evaluation").exists()
    assert not any(
        path.name in {"quality.py", "rna_worker.py", "native.py"} for path in package.rglob("*.py")
    )
    assert not any(name.startswith("bincomplete.evaluation") for name in sys.modules)


def test_explicit_frozen_local_encoder_is_lazy_and_reuses_cached_vectors(
    dataset, tmp_path, limits, monkeypatch
):
    torch = pytest.importorskip("torch")
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    save(encoder / "config.json", {"hidden_size": 8})
    save(encoder / "tokenizer.json", {"synthetic": True})
    (encoder / "weights.bin").write_bytes(b"synthetic")
    parent = dataset.path.parent
    prepared = prepare_dataset(
        tmp_path / "prepared",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        encoder_path=encoder,
    )
    calls = []

    class Encoder(torch.nn.Module):
        def forward(self, **tokens):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 4, 8))

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.ones(1, 4, dtype=torch.int64),
                "attention_mask": torch.ones(1, 4, dtype=torch.int64),
            }

    def model_loader(path, **kwargs):
        assert kwargs["local_files_only"] is True
        calls.append(path)
        return Encoder()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
        ),
    )
    from bincomplete.representations import dna

    monkeypatch.setattr(dna, "version", lambda name: "synthetic-runtime")
    assert calls == []
    for _ in range(2):
        context = Context(prepared, recipe("cosine", limits), tmp_path / "work", tmp_path / "cache", limits)
        try:
            vectors = context.representation("embedding")
            assert isinstance(vectors, np.memmap)
            np.testing.assert_allclose(np.linalg.norm(vectors[:], axis=1), 1, atol=1e-6)
        finally:
            context.close()
    assert len(calls) == 1
