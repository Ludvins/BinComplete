"""Stream content-addressed FASTAs at the explicit materialization boundary."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import complete, fingerprint, read, save, sha
from .data import Dataset
from .engine import ProposalArtifact, _new_or_same
from .reconcile import SelectionArtifact, load_proposals
from .resources import FileLock, ResourceLimits


def _materialize(
    dataset: Dataset, members: list[str], root: Path, positions: dict[str, int], limits: ResourceLimits
) -> tuple[Path, str]:
    members = sorted(members)
    index = root / "objects" / "memberships" / (fingerprint(members) + ".json")
    if index.exists():
        record = read(index)
        path = root / "objects" / (record["sha256"] + ".fa")
        if sha(path) != record["sha256"]:
            raise ValueError("Content-addressed FASTA changed")
        return path, record["sha256"]
    temporary = root / "objects" / f"{fingerprint(members)}.partial.fa"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows = [positions[name] for name in members]
    try:
        with temporary.open("wb") as stream:
            for name, sequence in zip(members, dataset.sequences(rows, limits)):
                header = b">" + name.encode("utf-8") + b"\n"
                stream.write(header)
                digest.update(header)
                for start in range(0, len(sequence), 80):
                    block = sequence[start : start + 80] + b"\n"
                    stream.write(block)
                    digest.update(block)
        key = digest.hexdigest()
        path = temporary.with_name(key + ".fa")
        if path.exists():
            if sha(path) != key:
                raise ValueError("Content-addressed FASTA changed")
        else:
            temporary.replace(path)
        save(index, {"sha256": key})
        return path, key
    finally:
        temporary.unlink(missing_ok=True)


def export(
    source: SelectionArtifact | ProposalArtifact,
    output: str | Path,
    *,
    proposal_ids: list[str] | None = None,
    complete_directory: bool = False,
    limits: ResourceLimits | None = None,
) -> Path:
    """Materialize selected proposals once and reference unchanged original bins.

    Parameters
    ----------
    source : SelectionArtifact or ProposalArtifact
        Compatible final selection, or proposals for external evaluation.
    output : path-like
        New export directory, containing content-addressed FASTAs and a manifest.
    proposal_ids : list of str, optional
        Required for proposal exports; omitted for compatible final selections.
    complete_directory : bool
        Also expose every unchanged bin in ``bins/``. Hard links are preferred;
        copies are used only when the filesystem does not support a hard link.
    limits : ResourceLimits, optional
        Maximum sequence-buffer allocation.

    Returns
    -------
    Path
        Export manifest. Without ``complete_directory``, unchanged bins are only
        referenced by path or original assembly membership, never duplicated.
    """
    limits = limits or ResourceLimits()
    manifest = source.manifest
    if isinstance(source, SelectionArtifact):
        if proposal_ids is not None:
            raise ValueError("A final selection already declares its proposal IDs")
        rows = manifest["selected"]
    else:
        if proposal_ids is None:
            raise ValueError("Explicit proposal_ids are required to materialize proposals")
        all_rows, _ = load_proposals([source])
        if len(proposal_ids) != len(set(proposal_ids)) or not set(proposal_ids) <= set(all_rows):
            raise ValueError("Unknown or repeated proposal IDs")
        rows = [all_rows[key] for key in sorted(proposal_ids)]
    if complete_directory and not isinstance(source, SelectionArtifact):
        raise ValueError("A complete final bin directory requires a compatible selection")
    dataset = Dataset(Path(manifest["dataset"]))
    prepared = dataset.verify()
    identity = fingerprint(
        {
            "source": manifest,
            "ids": [row["proposal_id"] for row in rows],
            "complete_directory": complete_directory,
            "export_version": 2,
        }
    )
    root = Path(output).resolve()
    with FileLock(root.with_name(root.name + ".export.lock")):
        if not _new_or_same(root, identity):
            return root / "export.json"
        ids = np.load(dataset.array_path("ids.npy"), mmap_mode="r", allow_pickle=False)
        try:
            limits.require(len(ids) * 512, "Export row index")
            positions = {str(name): i for i, name in enumerate(ids)}
        finally:
            ids._mmap.close()
        changed = []
        selected_recipients = {row["recipient"] for row in rows}

        def expose(path: Path, name: str) -> dict[str, str]:
            destination = root / "bins" / (fingerprint(name) + ".fa")
            destination.parent.mkdir(exist_ok=True)
            try:
                os.link(path, destination)
                mode = "hardlink"
            except OSError:
                shutil.copyfile(path, destination)
                mode = "copy"
            return {"path": destination.relative_to(root).as_posix(), "storage": mode}

        for row in rows:
            path, digest = _materialize(dataset, row["members"], root, positions, limits)
            entry = {
                "recipient": row["recipient"],
                "proposal_id": row["proposal_id"],
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
            }
            if complete_directory:
                entry["final_bin"] = expose(path, row["recipient"])
            changed.append(entry)
        unchanged = []
        if isinstance(source, SelectionArtifact):
            for name, members in prepared["bins"].items():
                if name in selected_recipients:
                    continue
                original = prepared["bin_fastas"].get(name)
                entry: dict[str, Any] = {
                    "recipient": name,
                    "original_fasta": original,
                    "assembly": prepared["assembly"],
                    "members": members,
                }
                if complete_directory:
                    if original is not None:
                        path = Path(original)
                    else:
                        path, _ = _materialize(dataset, members, root, positions, limits)
                    entry["final_bin"] = expose(path, name)
                unchanged.append(entry)
        save(
            root / "export.json",
            {
                "schema_version": 1,
                "dataset": str(dataset.path.resolve()),
                "changed": changed,
                "unchanged": unchanged,
                "complete_directory": complete_directory,
                "quality_verdicts": "Supplied externally; no tool was invoked by export",
            },
        )
        with (root / "bins.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["recipient", "proposal_id", "fasta", "sha256", "changed", "quality_verdict"])
            for entry in changed:
                writer.writerow(
                    [
                        entry["recipient"],
                        entry["proposal_id"],
                        entry.get("final_bin", {}).get("path", entry["path"]),
                        entry["sha256"],
                        "true",
                        "EXTERNAL_APPROVAL" if isinstance(source, SelectionArtifact) else "UNASSESSED",
                    ]
                )
            for entry in unchanged:
                original_path = entry.get("final_bin", {}).get("path", entry["original_fasta"] or "")
                original_hash = sha(root / original_path) if original_path else ""
                writer.writerow(
                    [
                        entry["recipient"],
                        "",
                        original_path,
                        original_hash,
                        "false",
                        "UNCHANGED_CONTROL",
                    ]
                )
        complete(root, {"kind": "export", "identity": identity})
        return root / "export.json"
