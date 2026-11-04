"""Frozen strand-aware, token-complete sequence pooling with bounded batches."""

from __future__ import annotations

import importlib
from contextlib import contextmanager

import numpy as np


def token_pieces(sequence, tokenizer, limit):
    """Split bases recursively until each complete tokenization fits its limit."""
    if len(tokenizer(sequence, add_special_tokens=True, truncation=False)["input_ids"]) <= limit:
        return [sequence]
    if len(sequence) < 2:
        raise ValueError("Tokenizer cannot fit one base within the declared token limit")
    middle = len(sequence) // 2
    return token_pieces(sequence[:middle], tokenizer, limit) + token_pieces(
        sequence[middle:], tokenizer, limit
    )


def pooled_vectors(tokenizer, model, sequences, space, execution):
    """Pool both strands and all token-complete pieces without truncation.

    Parameters
    ----------
    tokenizer, model : object
        Explicitly loaded frozen local tokenizer and encoder.
    sequences : list of bytes
        Bounded group of original contig sequences.
    space, execution : dict
        Semantic window/pooling contract and declared numerical execution policy.

    Returns
    -------
    numpy.ndarray
        L2-normalized float32 vectors, before declared storage quantization.
    """
    from ..sequence_io import window_starts

    windows = []
    for raw in sequences:
        starts = window_starts(len(raw), space["window_bp"], space["windows"])
        windows.append([raw[start : start + space["window_bp"]] for start in starts])
    return pooled_windows(tokenizer, model, windows, space, execution)


def pooled_windows(tokenizer, model, windows, space, execution):
    """Pool already-read windows with unchanged strand, token and reduction order.

    Parameters
    ----------
    tokenizer, model : object
        Frozen local tokenizer and encoder.
    windows : list of list of bytes
        Ordered windows per contig, exactly as sliced from the original sequence.
    space, execution : dict
        The same semantic and numerical contracts as ``pooled_vectors``.

    Returns
    -------
    numpy.ndarray
        Identical normalized vectors for identical window bytes and batching.
    """
    import torch

    pieces, owners, weights = [], [], []
    for owner, contig_windows in enumerate(windows):
        for raw in contig_windows:
            window = raw.decode("ascii")
            orientations = [window]
            if space["reverse_complement"]:
                orientations.append(window.translate(str.maketrans("ACGTN", "TGCAN"))[::-1])
            for orientation in orientations:
                for piece in token_pieces(orientation, tokenizer, space["max_tokens"]):
                    pieces.append(piece)
                    owners.append(owner)
                    weights.append(len(piece))
    output = [[] for _ in windows]
    mass = [[] for _ in windows]
    device = execution["device"]
    batch_size = execution["batch_size"]
    for start in range(0, len(pieces), batch_size):
        tokens = tokenizer(
            pieces[start : start + batch_size], return_tensors="pt", padding=True, truncation=False
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        if tokens["input_ids"].shape[1] > space["max_tokens"]:
            raise ValueError("Token-complete window exceeded its declared bound")
        special = torch.isin(tokens["input_ids"], torch.tensor(tokenizer.all_special_ids, device=device))
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda") and execution["precision"] == "bfloat16",
            ),
        ):
            result = model(**tokens)
            hidden = result.last_hidden_state if hasattr(result, "last_hidden_state") else result[0]
            mask = tokens.get("attention_mask", torch.ones_like(tokens["input_ids"])).bool() & ~special
            vector = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
        for offset, value in enumerate(vector.cpu().numpy()):
            index = start + offset
            output[owners[index]].append(value)
            mass[owners[index]].append(weights[index])
    vectors = np.asarray(
        [np.average(values, axis=0, weights=mass[i]) for i, values in enumerate(output)], np.float32
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms <= 1e-12).any():
        raise ValueError("Encoder returned invalid pooled sequence vectors")
    return vectors / norms


@contextmanager
def local_dynamic_imports():
    """Use a checkpoint's optional Triton fallback and restore Transformers state."""
    import transformers.dynamic_module_utils as dynamic

    previous = dynamic.check_imports

    def check(filename):
        for name in dynamic.get_imports(filename):
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as error:
                if name != "triton" or error.name != "triton":
                    raise
        return dynamic.get_relative_imports(filename)

    dynamic.check_imports = check
    try:
        yield
    finally:
        dynamic.check_imports = previous


def load_local_encoder(path):
    """Load exact local DNABERT-S tensors without downloading or altering them."""
    from pathlib import Path

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    with local_dynamic_imports():
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=True, model_max_length=2048
        )
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], str(path), local_files_only=True
        )
        model = model_class(config)
        state = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return tokenizer, model
