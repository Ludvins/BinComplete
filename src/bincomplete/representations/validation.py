"""Shared numerical checks for stored embedding vectors."""

from collections.abc import Sequence

import numpy as np

from ..resources import ResourceLimits


def l2_tolerance(dtype: np.dtype) -> float:
    return 5e-4 if dtype == np.float16 else 1e-4


def l2_errors(rows: np.ndarray) -> np.ndarray:
    """Measure stored rows in float32, including non-finite norms."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.abs(np.linalg.norm(np.asarray(rows, np.float32), axis=1) - 1)


def embedding_block_size(dimension: int, limits: ResourceLimits) -> int:
    return min(limits.chunk_size, max(1, limits.memory_bytes // max(1, dimension * 32)))


def require_l2(errors: np.ndarray, tolerance: float, ids: Sequence[str], source: str) -> None:
    invalid = np.flatnonzero(~(errors <= tolerance))
    if len(invalid):
        names = ", ".join(repr(str(ids[i])) for i in invalid[:5])
        raise ValueError(
            f"Embedding rows must be finite and already L2 normalized ({source}). Invalid contig IDs: {names}"
        )


def validate_l2(matrix: np.ndarray, ids: Sequence[str], limits: ResourceLimits, source: str) -> float:
    """Validate stored rows in bounded blocks and return the maximum norm error."""
    width = embedding_block_size(matrix.shape[1], limits)
    tolerance = l2_tolerance(matrix.dtype)
    maximum_error = 0.0
    for start in range(0, len(matrix), width):
        errors = l2_errors(matrix[start : start + width])
        require_l2(errors, tolerance, ids[start : start + width], source)
        maximum_error = max(maximum_error, float(errors.max()))
    return maximum_error


def validate_encoder_matrix(matrix, ids, dimension, dtype, limits, source):
    if matrix.shape != (len(ids), dimension) or matrix.dtype != np.dtype(dtype):
        raise ValueError(f"Unexpected embedding matrix shape or storage dtype ({source})")
    validate_l2(matrix, ids, limits, source)
