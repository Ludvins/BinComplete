"""Explicit recruitment resource limits and portable process locks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


@dataclass(frozen=True)
class ResourceLimits:
    """Bound recruitment buffers, numerical threads and rebuildable cache.

    Parameters
    ----------
    memory_gib : float
        Budget for recruitment allocations, excluding the operating system and
        separately managed external tools. Oversized fits fail before allocation.
    threads : int
        Maximum numerical thread count, between one and four.
    cache_gib : float
        Maximum size of completed rebuildable cache entries.
    chunk_size : int
        Upper bound on the number of candidate rows scored together.
    """

    memory_gib: float = 4.0
    threads: int = 4
    cache_gib: float = 10.0
    chunk_size: int = 4096

    def __post_init__(self) -> None:
        import math

        if not math.isfinite(self.memory_gib) or self.memory_gib <= 0:
            raise ValueError("memory_gib must be finite and positive")
        if not math.isfinite(self.cache_gib) or self.cache_gib <= 0:
            raise ValueError("cache_gib must be finite and positive")
        if type(self.threads) is not int or not 1 <= self.threads <= 4:
            raise ValueError("threads must be an integer in 1..4")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

    @property
    def memory_bytes(self) -> int:
        """Return the recruitment allocation budget in bytes."""
        return int(self.memory_gib * 1024**3)

    def require(self, amount: int, purpose: str) -> None:
        """Reject a declared allocation above the recruitment budget."""
        if amount > self.memory_bytes:
            raise MemoryError(
                f"{purpose} needs approximately {amount / 1024**3:.2f} GiB; "
                "increase the explicit budget or change the method configuration"
            )


class FileLock:
    """Hold a cross-process advisory lock until explicitly closed."""

    def __init__(self, path: str | Path, *, blocking: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if self.stream.read(1) == b"":
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                msvcrt.locking(self.stream.fileno(), mode, 1)
            else:
                import fcntl

                mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                fcntl.flock(self.stream.fileno(), mode)
        except OSError:
            self.stream.close()
            raise

    def close(self) -> None:
        """Release the operating-system lock."""
        if not self.stream.closed:
            self.stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()

    def __enter__(self) -> FileLock:
        return self

    def __exit__(
        self, exc_type: type | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
