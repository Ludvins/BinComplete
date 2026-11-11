"""Bounded derived-array cache; user inputs and model artifacts are never evicted."""

from __future__ import annotations

import os
import re
import stat
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import fingerprint, read, save, sha
from .resources import FileLock, LockBusyError, ResourceLimits

_KEY = re.compile(r"[0-9a-f]{64}")
_PROGRESS = "encoder.checkpoint.progress.json"
_PROGRESS_TEMP = re.compile(rf"{re.escape(_PROGRESS)}\.\d+\.partial")
_PENDING = "pending.json"
_PARTIAL_FILES = {
    _PENDING,
    "values.npy",
    "values.partial.npy",
    "encoder.checkpoint.npy",
    "encoder.done.npy",
    _PROGRESS,
}
_PARTIAL_JSON = re.compile(r"(?:complete|pending|encoder\.checkpoint\.progress)\.json\.\d+\.partial")


def _plain(path: Path, *, directory: bool) -> bool:
    """Reject symlinks and Windows reparse points before touching owned entries."""
    info = path.lstat()
    return (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)) and not getattr(
        info, "st_file_attributes", 0
    ) & stat.FILE_ATTRIBUTE_REPARSE_POINT


def default_cache() -> Path:
    """Return an environment-configurable cache outside the source tree."""
    base = os.environ.get("BINCOMPLETE_CACHE")
    if base:
        return Path(base)
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "bincomplete"


class DerivedCache:
    """Manage rebuildable arrays with entry leases and a byte-limited feature LRU."""

    def __init__(
        self,
        root: str | Path | None,
        limits: ResourceLimits,
        *,
        before_publish: Callable[[], None] | None = None,
    ):
        self.root = Path(root) if root is not None else default_cache()
        self.limits = limits
        self.hits = 0
        self._arrays: dict[str, np.memmap] = {}
        self._leases: dict[str, FileLock] = {}
        self._features: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feature_bytes = 0
        self._feature_limit = min(64 * 1024**2, limits.memory_bytes // 16)
        self._before_publish = before_publish

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
        folder.parent.mkdir(parents=True, exist_ok=True)
        if not _plain(folder.parent, directory=True):
            raise ValueError(f"Cache arrays must be an ordinary directory: {folder.parent}")
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
                    existing = self._entry_files(folder) if folder.exists() or folder.is_symlink() else set()
                    if existing is None or any(
                        name not in _PARTIAL_FILES and not _PARTIAL_JSON.fullmatch(name) for name in existing
                    ):
                        raise ValueError(f"Unrecognized files in unfinished cache entry: {folder}")
                    pending = folder / _PENDING
                    is_encoder = isinstance(identity, dict) and identity.get("kind") == "local_dnabert_s_v2"
                    checkpoint_bytes = 0
                    if _PENDING in existing:
                        if read(pending) != {"identity": identity}:
                            raise ValueError(f"Unfinished cache identity changed: {folder}")
                        # A failed publication may have left a matrix without its completion proof.
                        destination.unlink(missing_ok=True)
                        (folder / "values.partial.npy").unlink(missing_ok=True)
                        checkpoints = {"encoder.checkpoint.npy", "encoder.done.npy"}
                        if is_encoder and checkpoints <= existing:
                            checkpoint_bytes = sum((folder / name).stat().st_size for name in checkpoints)
                    if is_encoder:
                        for name in existing:
                            if _PROGRESS_TEMP.fullmatch(name):
                                (folder / name).unlink()
                    reserve = expected_bytes + 4096 - min(expected_bytes, checkpoint_bytes)
                    self._evict(reserve, quota, key)
                    folder.mkdir(exist_ok=True)
                    save(pending, {"identity": identity})
                    temporary = folder / "values.partial.npy"
                    try:
                        writer(temporary)
                        if temporary.stat().st_size > expected_bytes + 4096:
                            raise MemoryError("Array writer exceeded its reserved cache space")
                        if self._before_publish is not None:
                            self._before_publish()
                        temporary.replace(destination)
                        save(marker, {"identity": identity, "sha256": sha(destination)})
                        if is_encoder:
                            (folder / _PROGRESS).unlink(missing_ok=True)
                        pending.unlink()
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
            if _plain(folder, directory=True):
                size = sum(p.stat().st_size for p in folder.iterdir() if _plain(p, directory=False))
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
                self._remove_entry(folder)
                total -= size
            finally:
                lock.close()
        if total + reserve > quota:
            raise MemoryError(
                "Cache ceiling reached; entries are pinned, incomplete or not owned by BinComplete"
            )

    def _entry_files(self, folder: Path) -> set[str] | None:
        if not _KEY.fullmatch(folder.name):
            return None
        try:
            if not _plain(folder, directory=True):
                return None
            files = list(folder.iterdir())
            if any(not _plain(path, directory=False) for path in files):
                return None
            return {path.name for path in files}
        except OSError:
            return None

    def _owned_entry(self, folder: Path) -> bool:
        """Recognize completed arrays, including legacy encoder progress files."""
        files = self._entry_files(folder)
        required = {"complete.json", "values.npy"}
        if files is None or not required <= files:
            return False
        extras = files - required - {_PROGRESS, _PENDING}
        if any(not _PROGRESS_TEMP.fullmatch(name) for name in extras):
            return False
        try:
            proof = read(folder / "complete.json")
            if (
                not isinstance(proof, dict)
                or set(proof) != {"identity", "sha256"}
                or fingerprint(proof["identity"]) != folder.name
                or not isinstance(proof["sha256"], str)
                or not _KEY.fullmatch(proof["sha256"])
            ):
                return False
            identity = proof["identity"]
            if _PENDING in files and read(folder / _PENDING) != {"identity": identity}:
                return False
            if (_PROGRESS in files or extras) and (
                not isinstance(identity, dict) or identity.get("kind") != "local_dnabert_s_v2"
            ):
                return False
            if _PROGRESS in files:
                progress = read(folder / _PROGRESS)
                if (
                    not isinstance(progress, dict)
                    or progress.get("phase") != "encoding"
                    or type(progress.get("total_contigs")) is not int
                    or progress.get("completed_contigs") != progress["total_contigs"]
                ):
                    return False
            return True
        except (OSError, TypeError, ValueError):
            return False

    def _owned_incomplete(self, folder: Path) -> bool:
        files = self._entry_files(folder)
        if (
            files is None
            or _PENDING not in files
            or any(name not in _PARTIAL_FILES and not _PARTIAL_JSON.fullmatch(name) for name in files)
        ):
            return False
        try:
            proof = read(folder / _PENDING)
            return (
                isinstance(proof, dict)
                and set(proof) == {"identity"}
                and fingerprint(proof["identity"]) == folder.name
            )
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _remove_entry(folder: Path) -> None:
        marker, pending = folder / "complete.json", folder / _PENDING
        if marker.exists():
            # Keep ownership recoverable and invalidate cache hits before removing the matrix.
            if not pending.exists():
                save(pending, {"identity": read(marker)["identity"]})
            marker.unlink()
        for path in folder.iterdir():
            if path != pending:
                path.unlink()
        pending.unlink()
        folder.rmdir()

    def clean(self, *, apply: bool = False, include_incomplete: bool = False) -> dict[str, Any]:
        """Preview or remove only verified, unleased entries in this cache."""
        arrays, locks, quota = self.root / "arrays", self.root / "locks", self.root / "quota.lock"
        entries: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "cache": str(self.root),
            "applied": apply,
            "entries": entries,
            "selected_bytes": 0,
            "skipped_in_use": 0,
            "skipped_incomplete": 0,
            "skipped_unrecognized": 0,
        }
        if not arrays.exists() and not arrays.is_symlink():
            return result
        if not _plain(arrays, directory=True):
            raise ValueError(f"Cache arrays must be an ordinary directory: {arrays}")
        for path, directory in ((locks, True), (quota, False)):
            if (path.exists() or path.is_symlink()) and not _plain(path, directory=directory):
                raise ValueError(f"Cache locking path is not an ordinary file or directory: {path}")
        try:
            quota_lock = FileLock(quota, blocking=False)
        except LockBusyError as error:
            raise OSError(
                f"Cache is busy: {self.root}. Retry after the active operation finishes."
            ) from error
        with quota_lock:
            for folder in sorted(arrays.iterdir()):
                completed = self._owned_entry(folder)
                if not completed:
                    if not self._owned_incomplete(folder):
                        result["skipped_unrecognized"] += 1
                        continue
                    if not include_incomplete:
                        result["skipped_incomplete"] += 1
                        continue
                lock_path = locks / (folder.name + ".lock")
                if (lock_path.exists() or lock_path.is_symlink()) and not _plain(lock_path, directory=False):
                    result["skipped_unrecognized"] += 1
                    continue
                try:
                    lease = FileLock(lock_path, blocking=False)
                except LockBusyError:
                    result["skipped_in_use"] += 1
                    continue
                with lease:
                    size = sum(path.stat().st_size for path in folder.iterdir())
                    if apply:
                        self._remove_entry(folder)
                    entries.append(
                        {
                            "key": folder.name,
                            "kind": "completed" if completed else "incomplete",
                            "bytes": size,
                        }
                    )
                    result["selected_bytes"] += size
        return result

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
