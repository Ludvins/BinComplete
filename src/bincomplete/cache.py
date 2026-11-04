"""Bounded derived-array cache; user inputs and model artifacts are never evicted."""

from __future__ import annotations

import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import fingerprint, read, save, sha
from .resources import FileLock, ResourceLimits


def default_cache() -> Path:
    """Return an environment-configurable cache outside the source tree."""
    base = os.environ.get("BINCOMPLETE_CACHE") or os.environ.get("BIN_RECRUITMENT_CACHE")
    if base:
        return Path(base)
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "bincomplete"


class DerivedCache:
    """Manage rebuildable arrays with entry leases and a byte-limited feature LRU."""

    def __init__(self, root: str | Path | None, limits: ResourceLimits):
        self.root = Path(root) if root is not None else default_cache()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.hits = 0
        self._arrays: dict[str, np.memmap] = {}
        self._leases: dict[str, FileLock] = {}
        self._features: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feature_bytes = 0
        self._feature_limit = min(64 * 1024**2, limits.memory_bytes // 16)

    def feature(self, identity: Any, writer: Callable[[], np.ndarray]) -> np.ndarray:
        """Reuse a small feature block without creating per-proposal disk files."""
        key = fingerprint(identity)
        if key in self._features:
            self.hits += 1
            self._features.move_to_end(key)
            return self._features[key]
        result = writer()
        if result.nbytes <= self._feature_limit:
            while self._features and self._feature_bytes + result.nbytes > self._feature_limit:
                self._feature_bytes -= self._features.popitem(last=False)[1].nbytes
            self._features[key] = result
            self._feature_bytes += result.nbytes
        return result

    def array(self, identity: Any, writer: Callable[[Path], None], *, expected_bytes: int) -> np.memmap:
        """Map a verified derived array, reserving cache space before construction."""
        key = fingerprint(identity)
        if key in self._arrays:
            self.hits += 1
            return self._arrays[key]
        quota = int(self.limits.cache_gib * 1024**3)
        if expected_bytes + 4096 > quota:
            raise MemoryError("Derived array exceeds the configured cache ceiling")
        folder = self.root / "arrays" / key
        folder.parent.mkdir(exist_ok=True)
        with FileLock(self.root / "quota.lock"):
            lease = FileLock(self.root / "locks" / (key + ".lock"), blocking=False)
            try:
                marker = folder / "complete.json"
                destination = folder / "values.npy"
                if marker.exists():
                    proof = read(marker)
                    if proof["identity"] != identity or sha(destination) != proof["sha256"]:
                        raise ValueError("Derived cache integrity changed")
                    self.hits += 1
                else:
                    self._evict(expected_bytes + 4096, quota, key)
                    folder.mkdir(exist_ok=True)
                    temporary = folder / "values.partial.npy"
                    try:
                        writer(temporary)
                        if temporary.stat().st_size > expected_bytes + 4096:
                            raise MemoryError("Array writer exceeded its reserved cache space")
                        temporary.replace(destination)
                        save(marker, {"identity": identity, "sha256": sha(destination)})
                    finally:
                        temporary.unlink(missing_ok=True)
                os.utime(folder, None)
                result = np.load(destination, mmap_mode="r", allow_pickle=False)
                self._arrays[key], self._leases[key] = result, lease
                return result
            except BaseException:
                lease.close()
                raise

    def _evict(self, reserve: int, quota: int, current: str) -> None:
        entries = []
        total = 0
        for folder in (self.root / "arrays").iterdir():
            if folder.is_dir():
                size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                entries.append((folder.stat().st_mtime, folder, size))
                total += size
        for _, folder, size in sorted(entries):
            if total + reserve <= quota:
                break
            if folder.name == current or folder.name in self._leases:
                continue
            try:
                lock = FileLock(self.root / "locks" / (folder.name + ".lock"), blocking=False)
            except OSError:
                continue
            try:
                if not self._owned_entry(folder):
                    continue
                shutil.rmtree(folder)
                total -= size
            finally:
                lock.close()
        if total + reserve > quota:
            raise MemoryError(
                "Cache ceiling reached; entries are pinned, incomplete or not owned by BinComplete"
            )

    def _owned_entry(self, folder: Path) -> bool:
        """Recognize completed derived entries without following links or deleting foreign data."""
        if not re.fullmatch(r"[0-9a-f]{64}", folder.name):
            return False
        try:
            if folder.is_symlink() or folder.resolve() != (self.root / "arrays").resolve() / folder.name:
                return False
            files = list(folder.iterdir())
            if {path.name for path in files} != {"complete.json", "values.npy"}:
                return False
            if any(path.is_symlink() or not path.is_file() for path in files):
                return False
            proof = read(folder / "complete.json")
            return (
                isinstance(proof, dict)
                and set(proof) == {"identity", "sha256"}
                and fingerprint(proof["identity"]) == folder.name
                and isinstance(proof["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", proof["sha256"]) is not None
            )
        except (OSError, TypeError, ValueError):
            return False

    def close(self) -> None:
        """Release mapped arrays, feature buffers and all cache leases."""
        for array in self._arrays.values():
            array._mmap.close()
        self._arrays.clear()
        self._features.clear()
        self._feature_bytes = 0
        for lease in self._leases.values():
            lease.close()
        self._leases.clear()
