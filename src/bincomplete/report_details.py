"""Recover inspector details from saved rankings and experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .artifacts import read, sha
from .data import Dataset
from .engine import ProposalArtifact
from .experiments.comparison import compare_proposals
from .report_quality import table, text_field
from .resources import ResourceLimits


class Donor(TypedDict):
    """Saved evidence for one added contig."""

    id: str
    length_bp: int
    mean_depth: float
    depth_variance: float | None
    score: float
    rank: int


def _array(path: Path, stack: ExitStack) -> Any:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    stack.callback(values._mmap.close)
    return values


def attach_donors(
    dataset: Dataset, manifests: dict[Path, dict[str, Any]], proposals: dict[str, dict[str, Any]]
) -> None:
    """Attach only added contigs and their saved scores, in donor rank order."""
    seen: set[str] = set()
    with ExitStack() as stack:
        names, lengths, coverage = (
            _array(dataset.array_path(name), stack) for name in ("ids.npy", "lengths.npy", "coverage.npy")
        )
        for path, manifest in manifests.items():
            group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            with (path / "memberships.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    group[row["recipient"]].append(row)
            artifact_id = read(path / "complete.json")["identity"]
            for recipient, records in group.items():
                folder = path / manifest["coverage"][recipient]["ranking"]
                with ExitStack() as ranking:
                    rows = _array(folder / "rows.npy", ranking)
                    scores = _array(folder / "scores.npy", ranking)
                    order = np.lexsort((names[rows], -scores))
                    required = {name for row in records for name in row["added"]}
                    donors: dict[str, Donor] = {}
                    for rank, index in enumerate(order, 1):
                        position = rows[index]
                        name = str(names[position])
                        if name not in required:
                            continue
                        variance = float(coverage[position, 1])
                        donors[name] = {
                            "id": name,
                            "length_bp": int(lengths[position]),
                            "mean_depth": float(coverage[position, 0]),
                            "depth_variance": variance if math.isfinite(variance) else None,
                            "score": float(scores[index]),
                            "rank": rank,
                        }
                        if len(donors) == len(required):
                            break
                for row in records:
                    key = row["proposal_id"]
                    proposal = proposals[key]
                    added = sorted((donors[name] for name in row["added"]), key=lambda donor: donor["rank"])
                    if (
                        key in seen
                        or proposal["artifact_id"] != artifact_id
                        or proposal["recipient"] != recipient
                        or len(added) != proposal["added_contig_count"]
                        or sum(donor["length_bp"] for donor in added) != proposal["added_bp"]
                        or math.fsum(donor["score"] for donor in added) / len(added)
                        != proposal["mean_added_score"]
                        or min(donor["score"] for donor in added) != proposal["min_added_score"]
                    ):
                        raise ValueError(f"Candidate inventory disagrees with saved donor ranking: {key}")
                    seen.add(key)
                    proposal["donors"] = added
    if seen != proposals.keys():
        raise ValueError("Candidate inventory does not match the saved proposal IDs")


def _rna_features(
    config: dict[str, Any], base: Path, selected: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str] | None]:
    if "rna_features" not in config:
        return {}, None
    raw = config["rna_features"]
    if not isinstance(raw, dict):
        raise ValueError("contig_evidence.rna_features must be an object")
    path = base / text_field(raw.get("report"), "RNA feature report")
    source = {key: text_field(raw.get(key), f"RNA feature {key}") for key in ("tool", "version", "settings")}
    source["report_sha256"] = sha(path)
    lengths = {
        proposal["id"]: {donor["id"]: donor["length_bp"] for donor in proposal["donors"]}
        for proposal in selected.values()
    }
    features = defaultdict(list)
    seen = set()
    for row in table(path, {"proposal_id", "contig_id", "feature", "start_bp", "end_bp", "strand"}):
        proposal, contig = row["proposal_id"], row["contig_id"]
        label = text_field(row["feature"], "RNA feature name")
        start, end, strand = int(row["start_bp"]), int(row["end_bp"]), row["strand"]
        length = lengths.get(proposal, {}).get(contig)
        key = (proposal, contig, label, start, end, strand)
        if length is None or not 1 <= start <= end <= length or strand not in {"+", "-"} or key in seen:
            raise ValueError(f"Invalid or repeated RNA feature for added contig {contig!r} in {proposal!r}")
        seen.add(key)
        features[proposal].append(
            {"contig_id": contig, "feature": label, "start_bp": start, "end_bp": end, "strand": strand}
        )
    return dict(features), source


def contig_evidence(
    dataset: Dataset,
    members: dict[str, list[str]],
    selected: dict[str, dict[str, Any]],
    config: Any,
    base: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None]:
    """Read sequence composition and optional saved RNA calls for selected bins."""
    if not isinstance(config, dict) or config.keys() - {"rna_features"}:
        raise ValueError("contig_evidence must be an object with an optional rna_features entry")
    features, source = _rna_features(config, base, selected)
    required = {name for recipient in selected for name in members[recipient]}
    required.update(donor["id"] for proposal in selected.values() for donor in proposal["donors"])
    if not required:
        return {}, source
    with ExitStack() as stack:
        names, coverage, hashes = (
            _array(dataset.array_path(name), stack)
            for name in ("ids.npy", "coverage.npy", "sequence_hashes.npy")
        )
        positions = {str(name): index for index, name in enumerate(names) if name in required}
        if positions.keys() != required:
            raise ValueError("Contig evidence does not match the prepared dataset")
        metadata = {}
        for (name, index), sequence in zip(
            positions.items(), dataset.sequences(list(positions.values()), ResourceLimits()), strict=True
        ):
            digest = hashlib.sha256(sequence).hexdigest()
            if digest != hashes[index].decode():
                raise ValueError(f"Assembly sequence changed after preparation: {name}")
            gc = sequence.count(b"G") + sequence.count(b"C")
            canonical = gc + sequence.count(b"A") + sequence.count(b"T")
            metadata[name] = (
                {
                    "length_bp": len(sequence),
                    "mean_depth": float(coverage[index, 0]),
                    "gc_fraction": gc / canonical if canonical else None,
                },
                digest,
            )
    return {
        recipient: {
            "original": [metadata[name][0] for name in members[recipient]],
            "added": [
                {
                    "id": name,
                    "gc_fraction": metadata[name][0]["gc_fraction"],
                    "sequence_sha256": metadata[name][1],
                }
                for name in (donor["id"] for donor in proposal["donors"])
            ],
            "rna_features": features.get(proposal["id"], []),
        }
        for recipient, proposal in selected.items()
    }, source


def _attention(manifest: dict[str, Any]) -> dict[str, Any] | None:
    root = Path(manifest["model"])
    if not (root / "complete.json").is_file():
        return None
    files = read(root / "complete.json")["outputs_sha256"]
    if files.get("model.json") != manifest["model_sha256"]:
        raise ValueError("Attention model metadata does not match its proposal artifact")
    for name in ("model.json", "summary.json", "selection_history.json"):
        if name in files and sha(root / name) != files[name]:
            raise ValueError(f"Attention model output changed: {name}")
    if "summary.json" not in files:
        return None
    epoch = read(root / "summary.json")["epochs"]
    if "selection_history.json" not in files:
        return {"epoch": epoch, "recall_at_5": None, "loss": None}
    selected = next(row for row in read(root / "selection_history.json") if row["epoch"] == epoch)
    return {"epoch": epoch, "recall_at_5": selected["validation_recall_at_5"], "loss": selected["loss"]}


def method_comparison(manifests: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    """Reuse the experiment comparison without exposing stored filesystem paths."""
    indexed = {
        read(path / "complete.json")["identity"]: (path, manifest) for path, manifest in manifests.items()
    }
    data = compare_proposals(
        {key: ProposalArtifact(path) for key, (path, _) in indexed.items()}, ResourceLimits()
    )
    for key, row in data["methods"].items():
        row.pop("artifact")
        if row["method"] == "single":
            row["attention_validation"] = _attention(indexed[key][1])
    return data
