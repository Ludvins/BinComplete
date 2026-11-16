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
from .report_quality import number, table, text_field
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


def _model_details(manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(manifest["model"])
    if not (root / "complete.json").is_file():
        return {}
    files = read(root / "complete.json")["outputs_sha256"]
    if files.get("model.json") != manifest["model_sha256"]:
        raise ValueError("Model metadata does not match its proposal artifact")
    for name in ("model.json", "summary.json", "selection_history.json"):
        if name in files and sha(root / name) != files[name]:
            raise ValueError(f"Model output changed: {name}")
    model = read(root / "model.json")
    summary = read(root / "summary.json") if "summary.json" in files else {}
    stats, recipe = model["training_statistics"], model["recipe"]
    output: dict[str, Any] = {
        "training": {
            "learned": model["learned"],
            "sources": len(model["training_sources"]),
            "groups": len(model["training_groups"]),
            "rows": stats.get("rows"),
            "positives": stats.get("positives"),
            "episodes": stats.get("episodes"),
            "features": len(summary["feature_schema"]) if "feature_schema" in summary else None,
            "weighting": summary.get("weighting"),
            "objective": summary.get("objective"),
            "scorer": recipe["scorer"]["name"],
            "retriever": recipe["retriever"]["name"],
            "classifier": (recipe.get("classifier") or {}).get("name"),
        }
    }
    if manifest["method"] == "single" and "epochs" in summary:
        epoch = summary["epochs"]
        history = read(root / "selection_history.json") if "selection_history.json" in files else []
        selected: dict[str, Any] = next((row for row in history if row["epoch"] == epoch), {})
        selection = summary.get("inner_group_selection") or {}
        output["attention_validation"] = {
            "epoch": epoch,
            "precision_at_5": selected.get("validation_precision_at_5"),
            "recall_at_5": selected.get("validation_recall_at_5"),
            "loss": selected.get("loss"),
            "held_out_sources": len(selection["validation_sources"])
            if "validation_sources" in selection
            else None,
            "held_out_groups": len(selection["validation_groups"])
            if "validation_groups" in selection
            else None,
        }
    return output


def method_comparison(
    manifests: dict[Path, dict[str, Any]], benchmark_results_path: Path | None = None
) -> dict[str, Any]:
    """Reuse the experiment comparison without exposing stored filesystem paths."""
    indexed = {
        read(path / "complete.json")["identity"]: (path, manifest) for path, manifest in manifests.items()
    }
    data = compare_proposals(
        {key: ProposalArtifact(path) for key, (path, _) in indexed.items()}, ResourceLimits()
    )
    for key, row in data["methods"].items():
        manifest = indexed[key][1]
        row.pop("artifact")
        row.update(_model_details(manifest))
        search = manifest.get("search", {})
        if search:
            row["search"] = {
                "name": search["name"],
                **{
                    field: search["params"][field]
                    for field in ("max_checks", "singleton_depth", "boundary_width", "max_added_fraction")
                    if field in search.get("params", {})
                },
            }
    if benchmark_results_path is not None:
        source_sha = sha(benchmark_results_path)
        seen = set()
        columns = {"artifact_id", "benchmark_id", "protocol", "precision_at_5", "recall_at_5"}
        for entry in table(benchmark_results_path, columns):
            identity = entry["artifact_id"]
            if identity not in data["methods"] or identity in seen:
                raise ValueError(f"Unknown or repeated benchmark artifact: {identity!r}")
            seen.add(identity)
            values = {
                key: number(entry[key], key) if entry[key].strip() else None
                for key in ("precision_at_5", "recall_at_5")
            }
            if all(value is None for value in values.values()) or any(
                not 0 <= value <= 1 for value in values.values() if value is not None
            ):
                raise ValueError(
                    "Benchmark results require Precision@5 or Recall@5 as a fraction from 0 to 1"
                )
            counts = {}
            for field in ("evaluated_bins", "evaluated_groups", "evaluated_episodes"):
                raw = entry.get(field, "").strip()
                if raw:
                    if not raw.isascii() or not raw.isdigit():
                        raise ValueError(f"Invalid benchmark count for {field}: {raw!r}")
                    counts[field] = int(raw)
            data["methods"][identity]["benchmark"] = {
                **values,
                **counts,
                "benchmark_id": text_field(entry["benchmark_id"], "benchmark ID"),
                "protocol": text_field(entry["protocol"], "benchmark protocol"),
                "run_id": entry.get("run_id", "").strip() or None,
                "report_sha256": source_sha,
            }
    return data
