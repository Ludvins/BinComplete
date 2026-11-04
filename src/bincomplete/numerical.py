"""Numerical execution identities for fresh representations and learned models."""

from __future__ import annotations

import os
import platform
from typing import Any


def torch_execution(device: str) -> dict[str, Any]:
    """Identify the actual numerical backend behind a logical device.

    Parameters
    ----------
    device : str
        Explicit Torch device, including any logical CUDA index.

    Returns
    -------
    dict
        JSON-compatible runtime, physical device and arithmetic settings.
        Models are not loaded and no inference is performed.
    """
    import torch

    runtime: dict[str, Any] = {
        "torch": str(torch.__version__),
        "device_type": torch.device(device).type,
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if runtime["device_type"] != "cuda":
        runtime["cpu"] = {"architecture": platform.machine(), "processor": platform.processor()}
        return runtime
    if not torch.cuda.is_available():
        raise ValueError("Requested CUDA runtime is unavailable")
    properties = torch.cuda.get_device_properties(torch.device(device))
    runtime["cuda"] = {
        "version": torch.version.cuda,
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "uuid": str(properties.uuid),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return runtime
