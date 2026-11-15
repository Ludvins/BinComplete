"""Verify the proposal inventory and materialize bounded assessment batches."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import fingerprint, read, sha
from .data import Dataset
from .engine import ProposalArtifact
from .experiments.runner import ExperimentArtifact
from .export import _materialize
from .reconcile import load_proposals
from .resources import ResourceLimits


@dataclass
class Catalog:
    dataset: Dataset
    originals: dict[str, list[str]]
    artifacts: list[ProposalArtifact]
    rows: dict[str, dict[str, Any]]
    positions: dict[str, int]
    hashes: dict[str, str]
    identity: dict[str, Any]

    @classmethod
    def load(cls, experiment: Path, limits: ResourceLimits) -> Catalog:
        experiment = experiment.resolve()
        complete = experiment / "complete.json"
        if not complete.is_file() or read(complete).get("kind") != "experiment":
            raise ValueError("Assessment requires a completed BinComplete experiment")
        summary = ExperimentArtifact(experiment).manifest
        if summary["status"] != "SUCCEEDED":
            raise ValueError("Assessment requires a successful experiment")
        artifacts = [ProposalArtifact(Path(row["proposals"])) for row in summary["methods"].values()]
        proposals, population = load_proposals(artifacts, limits=limits)
        proposal_bytes = sum(
            1024 + sum(72 + len(member) * 4 for member in row["members"]) for row in proposals.values()
        )
        limits.require(proposal_bytes, "Source proposal index")
        dataset = Dataset(Path(population["dataset"]))
        manifest = dataset.verify()
        if manifest["identity"] != population["dataset_identity"]:
            raise ValueError("Assessment dataset identity differs from proposals")
        originals = manifest["bins"]
        assigned = {member for members in originals.values() for member in members}
        if len(assigned) != sum(map(len, originals.values())):
            raise ValueError("Original bin memberships overlap")

        ids = np.load(dataset.array_path("ids.npy"), mmap_mode="r", allow_pickle=False)
        hashes = np.load(dataset.array_path("sequence_hashes.npy"), mmap_mode="r", allow_pickle=False)
        try:
            limits.require(proposal_bytes + len(ids) * 768, "Assessment sequence index")
            positions = {str(name): index for index, name in enumerate(ids)}
            sequence_hashes = {
                str(name): value.decode("ascii") for name, value in zip(ids, hashes, strict=True)
            }
        finally:
            ids._mmap.close()
            hashes._mmap.close()

        table = experiment / "candidates.tsv"
        proof = experiment / "candidates.tsv.json"
        receipt = read(proof)
        if (
            receipt.get("schema_version") != 2
            or receipt.get("all_proposals_retained") is not True
            or receipt.get("sha256") != sha(table)
            or summary["assessment_priority"]["receipt_sha256"] != sha(proof)
        ):
            raise ValueError("Invalid assessment inventory receipt")
        methods = {str(item.path.resolve()): item.manifest["method"] for item in artifacts}
        rows: dict[str, dict[str, Any]] = {}
        seen = set()
        memory = proposal_bytes + len(ids) * 768
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, limits.memory_bytes // 8))
        try:
            with table.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    pid = row["proposal_id"]
                    original = proposals.get(pid)
                    name = row["recipient"]
                    members = json.loads(row["members_json"])
                    if (
                        pid in seen
                        or original is None
                        or name != original["recipient"]
                        or name not in originals
                        or members != sorted(set(original["members"]))
                        or methods.get(row["artifact"]) != row["method"]
                        or original["method"] != row["method"]
                    ):
                        raise ValueError(f"Inventory row differs from source proposal: {pid}")
                    core = set(originals[name])
                    if not core < set(members) or (set(members) - core) & assigned:
                        raise ValueError(f"Proposal changes protected original membership: {pid}")
                    key = fingerprint({"recipient": name, "members": members})
                    sequence = fingerprint({member: sequence_hashes[member] for member in members})
                    if (key, sequence) != (row["construction_id"], row["sequence_set_id"]):
                        raise ValueError(f"Inventory construction identity differs: {pid}")
                    tier, position = int(row["assessment_tier"]), int(row["priority_rank"])
                    if tier not in {1, 2} or position < 1:
                        raise ValueError(f"Invalid assessment rank: {pid}")
                    rank = [tier, position, pid]
                    seen.add(pid)
                    memory += 1024 + len(row["members_json"]) * 4
                    limits.require(memory, "Assessment inventory")
                    item = rows.setdefault(
                        key,
                        {
                            "recipient": name,
                            "members": members,
                            "sequence_set_id": sequence,
                            "construction_id": key,
                            "priority": rank,
                            "proposals": {},
                        },
                    )
                    item["priority"] = min(item["priority"], rank)
                    item["proposals"][pid] = row["method"]
        finally:
            csv.field_size_limit(previous_limit)
        if seen != proposals.keys() or len(seen) != receipt["rows"]:
            raise ValueError("Assessment inventory does not cover every proposal")
        return cls(
            dataset,
            originals,
            artifacts,
            rows,
            positions,
            sequence_hashes,
            {
                "experiment": str(experiment),
                "experiment_sha256": sha(complete),
                "inventory_sha256": receipt["sha256"],
                "dataset_identity": manifest["identity"],
            },
        )

    def batch(self, keys: list[str], originals: list[str]) -> dict[str, dict[str, Any]]:
        names = originals or sorted({self.rows[key]["recipient"] for key in keys})
        entries = {}
        for index, name in enumerate(names, 1):
            members = self.originals[name]
            entries[f"o{index:06d}"] = {
                "recipient": name,
                "members": members,
                "sequence_set_id": fingerprint({member: self.hashes[member] for member in members}),
            }
        entries.update({f"c{index:06d}": self.rows[key] for index, key in enumerate(keys, 1)})
        return entries

    def materialize(
        self,
        entries: dict[str, dict[str, Any]],
        scratch: Path,
        limits: ResourceLimits,
    ) -> tuple[Path, dict[str, Any]]:
        inputs = scratch / "bins"
        inputs.mkdir()
        manifest = {}
        for name, row in entries.items():
            path, digest = _materialize(
                self.dataset,
                row["members"],
                scratch,
                self.positions,
                limits,
                sequence_hashes=self.hashes,
            )
            target = inputs / f"{name}.fa"
            try:
                os.link(path, target)
            except OSError:
                shutil.copyfile(path, target)
            manifest[name] = row | {"fasta": target.name, "sha256": digest}
        return inputs, manifest
