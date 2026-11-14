"""Optional frozen local DNABERT-S inference, independent of model training."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import fingerprint, read, sha
from . import validation


def local_contract(
    path: str | Path, options: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Describe frozen local weights and tokenizer without loading their model."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(
            "encoder_path must reference an existing local DNABERT-S directory. "
            "Run bincomplete models install dnabert_s or configure the path to your local weights."
        )
    files = {str(p): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    config = read(root / "config.json")
    if "hidden_size" not in config or type(config["hidden_size"]) is not int:
        raise ValueError("Local encoder config must declare hidden_size")
    weights = {
        Path(p).relative_to(root).as_posix(): digest
        for p, digest in files.items()
        if Path(p).suffix in {".bin", ".safetensors", ".py"} or Path(p).name == "config.json"
    }
    tokenizer = {
        Path(p).relative_to(root).as_posix(): digest
        for p, digest in files.items()
        if "token" in Path(p).name or Path(p).name in {"vocab.txt", "merges.txt"}
    }
    if not any(Path(name).suffix in {".bin", ".safetensors"} for name in weights) or not tokenizer:
        raise ValueError("Local encoder needs both frozen weights and tokenizer files")
    options = options or {}
    if set(options) - {
        "pooling",
        "window_bp",
        "windows",
        "max_tokens",
        "reverse_complement",
        "storage_dtype",
        "device",
        "precision",
        "batch_size",
    }:
        raise ValueError("Unknown encoder option")
    pooling = options.get("pooling", "masked_mean_uniform_windows")
    if pooling not in {"masked_mean_uniform_windows", "non_special_bp_weighted_windows"}:
        raise ValueError("Unknown frozen encoder pooling protocol")
    for key in ("window_bp", "windows", "max_tokens", "batch_size"):
        if key in options and (type(options[key]) is not int or options[key] < 1):
            raise ValueError("Encoder window and batch sizes must be positive integers")
    if options.get("storage_dtype", "float32") not in {"float16", "float32"}:
        raise ValueError("Encoder storage dtype must be float16 or float32")
    if options.get("precision", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("Encoder precision must be float32 or bfloat16")
    device = options.get("device", "cpu")
    if not isinstance(device, str) or (
        device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("Encoder device must be cpu, cuda or cuda:N")
    if "reverse_complement" in options and type(options["reverse_complement"]) is not bool:
        raise ValueError("reverse_complement must be a boolean")
    if pooling == "masked_mean_uniform_windows" and (
        device != "cpu"
        or options.get("reverse_complement", False)
        or options.get("precision", "float32") != "float32"
    ):
        raise ValueError("Strand/device/precision options require non_special_bp_weighted_windows pooling")
    if options.get("precision", "float32") == "bfloat16" and device == "cpu":
        raise ValueError("The declared bfloat16 encoder protocol requires CUDA")
    return (
        {
            "encoder": "dnabert_s",
            "encoder_sha256": fingerprint(weights),
            "tokenizer_sha256": fingerprint(tokenizer),
            "pooling": pooling,
            "window_bp": options.get("window_bp", 6000),
            "windows": options.get("windows", 4),
            "max_tokens": options.get("max_tokens", 512),
            **(
                {"reverse_complement": options.get("reverse_complement", False)}
                if pooling == "non_special_bp_weighted_windows"
                else {}
            ),
            "normalization": "l2",
            "dimension": config["hidden_size"],
        },
        files,
    )


def _source_stamp(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def frozen_array(context, spec: dict[str, Any]):
    """Compute or reuse a bounded cached array using explicitly supplied local code."""
    space = spec["space"]
    weights = sum(
        Path(path).stat().st_size
        for path in spec["model_files"]
        if Path(path).suffix in {".bin", ".safetensors"}
    )
    context.limits.require(
        context.reserved_bytes + weights * 3 + 128 * 1024**2,
        "Frozen encoder parameters and activation workspace",
    )
    from ..numerical import torch_execution

    assembly = Path(context.store.manifest["assembly"])
    assembly_hash = context.store.manifest["inputs_sha256"][str(assembly)]
    ids, limits = context.store.ids, context.limits
    dimension = space["dimension"]
    dtype = np.dtype(spec.get("execution", {}).get("storage_dtype", "float32"))
    tolerance = validation.l2_tolerance(dtype)
    identity = {
        "kind": "local_dnabert_s_v2",
        "space": space,
        "implementation": sha(__file__),
        "assembly": assembly_hash,
        "runtime": {name: version(name) for name in ("torch", "transformers")},
        "execution": spec.get("execution", {}),
        "numerical_runtime": torch_execution(spec.get("execution", {}).get("device", "cpu")),
        "contigs_per_group": 16,
        "strand_pooling_implementation": sha(Path(__file__).with_name("strand_pooling.py")),
        "embedding_validation": sha(validation.__file__),
    }

    def validate(array, source):
        validation.validate_encoder_matrix(array, ids, dimension, dtype, limits, source)

    def writer(path: Path) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        previous = torch.get_num_threads()
        torch.set_num_threads(context.limits.threads)
        model, output, done = None, None, None
        try:
            sources = {Path(path): digest for path, digest in spec["model_files"].items()}
            sources[assembly] = assembly_hash
            stamps = {path: _source_stamp(path) for path in sources}

            def require_unchanged(path: Path) -> None:
                if _source_stamp(path) != stamps[path]:
                    raise ValueError(f"Encoder input changed during encoding: {path}")

            for source, expected in sources.items():
                if sha(source) != expected:
                    raise ValueError(f"Encoder input changed since preparation: {source}")
                require_unchanged(source)
            if space["pooling"] == "non_special_bp_weighted_windows":
                from itertools import islice
                from time import perf_counter

                from ..artifacts import save
                from .strand_pooling import load_local_encoder, pooled_windows

                execution = spec["execution"]
                device = execution["device"]
                if device.startswith("cuda") and not torch.cuda.is_available():
                    raise ValueError("Requested CUDA encoder is unavailable")
                tokenizer, model = load_local_encoder(spec["model_path"])
                model = model.to(device).eval()
                model.requires_grad_(False)
                for source in sources:
                    require_unchanged(source)
                checkpoint = path.with_name("encoder.checkpoint.npy")
                flags = path.with_name("encoder.done.npy")
                rows = len(ids)
                if checkpoint.exists() and flags.exists():
                    output = np.load(checkpoint, mmap_mode="r+", allow_pickle=False)
                    done = np.load(flags, mmap_mode="r+", allow_pickle=False)
                    if (
                        output.shape != (rows, dimension)
                        or output.dtype != dtype
                        or done.shape != (rows,)
                        or done.dtype != np.dtype(bool)
                    ):
                        raise ValueError("Encoder checkpoint identity or shape changed")
                    width = validation.embedding_block_size(dimension, limits)
                    reset = 0
                    for start in range(0, rows, width):
                        flags_block = done[start : start + width]
                        invalid = flags_block & ~(
                            validation.l2_errors(output[start : start + width]) <= tolerance
                        )
                        flags_block[invalid] = False
                        reset += int(invalid.sum())
                    if reset:
                        done.flush()
                        print(f"Re-encoding {reset} invalid checkpoint rows", flush=True)
                else:
                    output = np.lib.format.open_memmap(
                        checkpoint, mode="w+", dtype=dtype, shape=(rows, space["dimension"])
                    )
                    done = np.lib.format.open_memmap(flags, mode="w+", dtype="bool", shape=(rows,))
                    done[:] = False
                    done.flush()
                todo = np.flatnonzero(~done)
                sequences = context.store.sequence_windows(
                    todo, window_bp=space["window_bp"], count=space["windows"]
                )
                completed = int(done.sum())
                started = perf_counter()
                for start in range(0, len(todo), 16):
                    group = list(islice(sequences, 16))
                    context.limits.require(
                        sum(len(value) for windows in group for value in windows) * 4
                        + context.reserved_bytes,
                        "Encoder contig group",
                    )
                    indices = todo[start : start + len(group)]
                    names = [ids[index] for index in indices]
                    try:
                        vectors = pooled_windows(tokenizer, model, group, space, execution)
                    except ValueError as error:
                        raise ValueError(
                            f"Encoder batch for contig IDs {', '.join(names)}: {error}"
                        ) from error
                    with np.errstate(over="ignore", invalid="ignore"):
                        stored = np.asarray(vectors, dtype=dtype)
                    if stored.shape != (len(indices), dimension):
                        raise ValueError(f"Invalid encoder batch shape for contig IDs {', '.join(names)}")
                    validation.require_l2(validation.l2_errors(stored), tolerance, names, "encoder batch")
                    require_unchanged(assembly)
                    output[indices] = stored
                    output.flush()
                    done[indices] = True
                    done.flush()
                    completed += len(group)
                    progress = {
                        "phase": "encoding",
                        "completed_contigs": completed,
                        "total_contigs": rows,
                        "fresh_contigs": start + len(group),
                        "elapsed_seconds": perf_counter() - started,
                        "checkpoint": str(checkpoint),
                        "device": device,
                    }
                    context.status(**progress)
                    save(checkpoint.with_suffix(".progress.json"), progress)
                    if start % 160 == 0:
                        print(f"Encoding {completed}/{rows} contigs", flush=True)
                validate(output, "encoder matrix before cache publication")
                done._mmap.close()
                done = None
                output._mmap.close()
                output = None
                checkpoint.replace(path)
                flags.unlink()
                return
            tokenizer = AutoTokenizer.from_pretrained(
                spec["model_path"], local_files_only=True, trust_remote_code=True
            )
            model = (
                AutoModel.from_pretrained(spec["model_path"], local_files_only=True, trust_remote_code=True)
                .float()
                .cpu()
                .eval()
            )
            model.requires_grad_(False)
            for source in sources:
                require_unchanged(source)
            output = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=(len(ids), dimension),
            )
            with torch.inference_mode():
                windows = context.store.sequence_windows(
                    range(len(context.store.ids)), window_bp=space["window_bp"], count=space["windows"]
                )
                for row, contig_windows in enumerate(windows):
                    pooled = []
                    for window in contig_windows:
                        tokens = tokenizer(
                            window.decode("ascii"),
                            return_tensors="pt",
                            truncation=True,
                            max_length=space["max_tokens"],
                        )
                        result = model(**tokens)
                        hidden = (
                            result.last_hidden_state if hasattr(result, "last_hidden_state") else result[0]
                        )
                        mask = tokens["attention_mask"].unsqueeze(-1)
                        pooled.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()[0])
                    vector = np.mean(pooled, axis=0)
                    norm = np.linalg.norm(vector)
                    if (
                        vector.shape != (dimension,)
                        or not np.isfinite(vector).all()
                        or not np.isfinite(norm)
                        or norm <= 1e-12
                    ):
                        raise ValueError(f"Frozen encoder returned an invalid vector for contig {ids[row]!r}")
                    require_unchanged(assembly)
                    stored = np.asarray([vector / norm], dtype=dtype)
                    validation.require_l2(
                        validation.l2_errors(stored), tolerance, [ids[row]], "encoder batch"
                    )
                    output[row] = stored[0]
            output.flush()
            validate(output, "encoder matrix before cache publication")
        finally:
            if done is not None:
                done._mmap.close()
            if output is not None:
                output._mmap.close()
            del model
            torch.set_num_threads(previous)

    return context.cache.array(
        identity,
        writer,
        expected_bytes=len(ids) * dimension * dtype.itemsize + len(ids) + 4096,
        validate_cached=lambda matrix: validate(matrix, "completed encoder cache"),
    )
