"""Package immutable run outputs into one portable BinCompleteInspector report."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from . import __version__
from .artifacts import read, sha
from .data import Dataset
from .engine import ProposalArtifact
from .export import export_identity
from .inventory import inventory_identity
from .reconcile import SelectionArtifact, selection_identity
from .report_details import attach_donors, contig_evidence, method_comparison
from .report_quality import (
    RNA_MARKERS,
    checked_export,
    has_evidence,
    number,
    policy,
    quality_for,
    read_assessments,
    table,
    text_field,
)

RNA_FIELDS = {*RNA_MARKERS, "trna_types"}
HQ_FIELDS = {"completeness", "contamination", "hq_status"}

_INVENTORY_COLUMNS = {
    "proposal_id",
    "method",
    "recipient",
    "artifact",
    "sequence_set_id",
    "added_bp",
    "added_contig_count",
    "mean_added_score",
    "min_added_score",
    "within_size_rank",
    "priority_rank",
    "assessment_tier",
}


def _artifact_paths(base: Path, raw: Any, label: str) -> list[Path]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a nonempty list of proposal artifact paths")
    paths = [(base / text_field(item, label)).resolve() for item in raw]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Repeated artifact in {label}")
    return paths


def _candidates(
    path: Path, manifests: dict[Path, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = read(path.with_name(path.name + ".json"))
    if (
        receipt.get("schema_version") != 2
        or receipt.get("all_proposals_retained") is not True
        or receipt.get("sha256") != sha(path)
        or receipt.get("identity") != inventory_identity(list(manifests.values()), receipt.get("priority"))
    ):
        raise ValueError("Candidate inventory receipt does not match the TSV and ordered proposal artifacts")
    artifact_ids = {artifact: read(artifact / "complete.json")["identity"] for artifact in manifests}
    proposals: dict[str, dict[str, Any]] = {}
    for row in table(path, _INVENTORY_COLUMNS):
        key, recipient = row["proposal_id"], row["recipient"]
        artifact = Path(row["artifact"]).resolve()
        manifest = manifests.get(artifact)
        if (
            not key
            or key in proposals
            or not row["sequence_set_id"]
            or manifest is None
            or row["method"] != manifest["method"]
            or recipient not in manifest["coverage"]
        ):
            raise ValueError(f"Invalid or repeated proposal in candidate inventory: {key!r}")
        proposals[key] = {
            "id": key,
            "recipient": recipient,
            "method": row["method"],
            "artifact_id": artifact_ids[artifact],
            "sequence_set_id": row["sequence_set_id"],
            "mean_added_score": number(row["mean_added_score"], "mean added score"),
            "min_added_score": number(row["min_added_score"], "minimum added score"),
            **{
                field: int(row[field])
                for field in (
                    "added_bp",
                    "added_contig_count",
                    "within_size_rank",
                    "priority_rank",
                    "assessment_tier",
                )
            },
        }
    if len(proposals) != receipt["rows"] or len(proposals) != sum(
        m["proposal_count"] for m in manifests.values()
    ):
        raise ValueError("Candidate inventory does not cover every saved proposal")
    return proposals, receipt


def _selection(
    path: Path,
    manifests: list[dict[str, Any]],
    approvals: list[str],
    time_limit: float,
    dataset_identity: str,
    proposals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = SelectionArtifact(path).manifest
    receipt = read(path / "complete.json")
    saved_limits = [time_limit]
    if time_limit.is_integer():
        saved_limits.append(int(time_limit))
    if (
        receipt.get("kind") != "selection"
        or "selection.json" not in receipt["outputs_sha256"]
        or manifest["dataset_identity"] != dataset_identity
        or receipt.get("identity")
        not in {
            selection_identity(manifests, approvals, manifest["protected_bins"], limit)
            for limit in saved_limits
        }
    ):
        raise ValueError(
            "Selection receipt does not match the approved IDs, proposal artifacts, or time limit"
        )
    used_recipients: set[str] = set()
    for row in manifest["selected"]:
        proposal = proposals.get(row["proposal_id"])
        if (
            proposal is None
            or proposal["recipient"] != row["recipient"]
            or row["recipient"] in used_recipients
        ):
            raise ValueError("Selected proposal does not match the candidate inventory")
        used_recipients.add(row["recipient"])
    return manifest


def _verify_final_export(root: Path, selection: dict[str, Any]) -> None:
    manifest, receipt = checked_export(root)
    ids = [row["proposal_id"] for row in selection["selected"]]
    changed = {(row["recipient"], row["proposal_id"]) for row in manifest["changed"]}
    if receipt["identity"] != export_identity(selection, ids, manifest["complete_directory"]) or changed != {
        (row["recipient"], row["proposal_id"]) for row in selection["selected"]
    }:
        raise ValueError("Final export does not match the reconciled selection")


def _summary(bins: list[dict[str, Any]]) -> dict[str, int | None]:
    recipients = [row for row in bins if row["recipient"]]
    pairs = [row for row in recipients if row["hq_transition"] != "unknown"]
    gains = sum(row["hq_transition"] == "gain" for row in pairs)
    losses = sum(row["hq_transition"] == "loss" for row in pairs)
    return {
        "original_bins": len(bins),
        "recipients": len(recipients),
        "modified_bins": sum(row["selected_proposal_id"] is not None for row in bins),
        "comparable_pairs": len(pairs),
        "unassessed_pairs": len(recipients) - len(pairs),
        "original_hq": sum(row["original"]["hq_status"] == "pass" for row in pairs) if pairs else None,
        "final_hq": sum(row["final"]["hq_status"] == "pass" for row in pairs) if pairs else None,
        "hq_gains": gains if pairs else None,
        "hq_losses": losses if pairs else None,
        "net_hq_change": gains - losses if pairs else None,
    }


def _rna_summary(bins: list[dict[str, Any]]) -> dict[str, int]:
    recipients = [row for row in bins if row["recipient"]]
    pairs = [
        (row["original"]["rna"]["status"], row["final"]["rna"]["status"])
        for row in recipients
        if not RNA_FIELDS.intersection(row["assessment_mismatches"])
    ]
    incomplete = sum("unknown" in pair for pair in pairs)
    return {
        "comparable_pairs": len(pairs) - incomplete,
        "incomplete_pairs": len(recipients) - len(pairs) + incomplete,
        "gains": pairs.count(("fail", "pass")),
        "losses": pairs.count(("pass", "fail")),
    }


def _assemble(config: dict[str, Any], base: Path) -> dict[str, Any]:
    candidates_path = base / text_field(config.get("candidates"), "candidates path")
    selection_path = base / text_field(config.get("selection"), "selection path")
    approvals_path = base / text_field(config.get("approved_ids"), "approved IDs path")
    paths = _artifact_paths(base, config.get("proposals"), "proposals")
    manifests = {path: ProposalArtifact(path).manifest for path in paths}
    population = manifests[paths[0]]
    dataset_path = Path(population["dataset"]).resolve()
    dataset_identity = population["dataset_identity"]
    if any(m["dataset_identity"] != dataset_identity for m in manifests.values()):
        raise ValueError("Proposal artifacts must belong to one prepared dataset")
    prepared = Dataset(dataset_path)
    dataset = prepared.verify(inputs=False)
    if dataset["identity"] != dataset_identity:
        raise ValueError("Prepared dataset does not match the proposal artifacts")
    recipients = set().union(*(manifest["coverage"] for manifest in manifests.values()))
    if not recipients <= set(dataset["bins"]):
        raise ValueError("Proposal coverage includes an unknown original bin")
    proposals, inventory = _candidates(candidates_path, manifests)
    attach_donors(prepared, manifests, proposals)
    approvals = [
        line.strip() for line in approvals_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    approved = set(approvals)
    if len(approved) != len(approvals) or not approved <= proposals.keys():
        raise ValueError("Approval list must contain unique saved proposal IDs")
    selection_paths = (
        _artifact_paths(base, config["selection_proposals"], "selection_proposals")
        if "selection_proposals" in config
        else paths
    )
    if not set(selection_paths) <= manifests.keys():
        raise ValueError("selection_proposals must reference artifacts in proposals")
    time_limit = number(config.get("selection_time_limit", 60.0), "selection time limit")
    if time_limit <= 0:
        raise ValueError("Selection time limit must be positive")
    selection = _selection(
        selection_path,
        [manifests[path] for path in selection_paths],
        approvals,
        time_limit,
        dataset_identity,
        proposals,
    )
    selected = {row["recipient"]: row["proposal_id"] for row in selection["selected"]}
    final_export = None
    if "final_export" in config:
        final_export = base / text_field(config["final_export"], "final export path")
        _verify_final_export(final_export, selection)
    hq_policy = policy(config.get("hq_policy"))
    evidence, assessment_sources, recipes = read_assessments(
        config.get("assessments", []), base, dataset_path, set(dataset["bins"]), proposals
    )
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals.values():
        key, recipient = proposal["id"], proposal["recipient"]
        observed = evidence.get(("proposal", proposal["sequence_set_id"]))
        is_selected = selected.get(recipient) == key
        proposal.update(
            quality=quality_for(observed, hq_policy),
            assessed=observed is not None and has_evidence(observed),
            approved=key in approved,
            selected=is_selected,
            exported=is_selected if final_export is not None else None,
        )
        by_bin[recipient].append(proposal)

    biological, rna_source = (
        contig_evidence(
            prepared,
            dataset["bins"],
            {name: proposals[key] for name, key in selected.items() if proposals[key]["donors"]},
            config["contig_evidence"],
            base,
        )
        if "contig_evidence" in config
        else ({}, None)
    )
    bins = []
    for name in sorted(dataset["bins"]):
        original = quality_for(evidence.get(("original", name)), hq_policy)
        selected_id = selected.get(name)
        final = proposals[selected_id]["quality"] if selected_id else original
        mismatches = set()
        if selected_id:
            before = recipes.get(("original", name), {})
            after = recipes.get(("proposal", proposals[selected_id]["sequence_set_id"]), {})
            mismatches = {field for field in before.keys() & after.keys() if before[field] != after[field]}
        comparable_hq = not mismatches.intersection(
            HQ_FIELDS | RNA_FIELDS if hq_policy.get("standard") == "mimag" else HQ_FIELDS
        )
        transition = (
            {
                ("fail", "pass"): "gain",
                ("pass", "fail"): "loss",
                ("pass", "pass"): "retained",
                ("fail", "fail"): "non_hq",
            }.get((original["hq_status"], final["hq_status"]), "unknown")
            if comparable_hq
            else "unknown"
        )
        deltas = {
            metric: after - before
            if before is not None and after is not None and metric not in mismatches
            else None
            for metric, before, after in (
                ("completeness", original["completeness"], final["completeness"]),
                ("contamination", original["contamination"], final["contamination"]),
            )
        }
        bins.append(
            {
                "id": name,
                "recipient": name in recipients,
                "original": original,
                "final": final,
                "delta": deltas,
                "hq_transition": transition,
                "assessment_mismatches": sorted(mismatches),
                "selected_proposal_id": selected_id,
                "proposals": by_bin[name],
                **({"contig_evidence": biological[name]} if name in biological else {}),
            }
        )
    return {
        "schema_version": 2,
        "format": "bincomplete-report",
        "hq_policy": hq_policy,
        "provenance": {
            "bincomplete_version": __version__,
            "dataset_identity": dataset_identity,
            "candidate_inventory_sha256": inventory["sha256"],
            "selection_receipt_sha256": sha(selection_path / "complete.json"),
            "final_export_receipt_sha256": sha(final_export / "complete.json")
            if final_export is not None
            else None,
            "assessment_sources": assessment_sources,
            "selection_optimal": selection["optimal"],
            "selection_upper_bound": selection["upper_bound"],
            "selection_canonical_tie_break": selection["canonical_tie_break"],
            "approved_recipient_union": selection["approved_recipient_union"],
            "approval_list_sha256": sha(approvals_path),
            "priority": inventory["priority"],
            **({"contig_rna_source": rna_source} if rna_source is not None else {}),
        },
        "summary": _summary(bins),
        "rna_summary": _rna_summary(bins),
        "comparison": method_comparison(
            manifests,
            base / text_field(config["benchmark_results"], "benchmark results path")
            if "benchmark_results" in config
            else None,
        ),
        "bins": bins,
    }


def write_report(config: str | Path, output: str | Path) -> Path:
    """Write an inspector report from completed artifacts and supplied quality evidence."""
    config = Path(config).resolve()
    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError(f"Report already exists; choose a new output path: {target}")
    if any((parent / "complete.json").is_file() for parent in target.resolve().parents):
        raise ValueError("Write the report outside completed artifact directories")
    raw = read(config)
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Report configuration must be a JSON object with schema_version 1")
    try:
        report = _assemble(raw, config.parent)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid report input or artifact metadata: {error}") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=target.parent, suffix=".partial", delete=False, mode="w", encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            if os.name == "nt":
                temporary.rename(target)
            else:
                os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target
