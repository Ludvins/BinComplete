"""Read existing quality evidence for a BinCompleteInspector report."""

from __future__ import annotations

import csv
import math
import operator
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath
from typing import Any, NotRequired, TypedDict, cast

from .artifacts import read, sha, verify_outputs

COMPARISONS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
HQ_STATUSES = {"pass", "fail", "unknown"}
RNA_MARKERS = ("rrna_5s", "rrna_16s", "rrna_23s")
CHECKM2_FIELDS = {
    "Completeness_General": ("completeness_general", float),
    "Completeness_Specific": ("completeness_specific", float),
    "Completeness_Model_Used": ("completeness_model", str),
    "Translation_Table_Used": ("translation_table", int),
    "Coding_Density": ("coding_density", float),
    "Contig_N50": ("contig_n50", int),
    "Average_Gene_Length": ("average_gene_length", float),
    "Genome_Size": ("genome_size", int),
    "GC_Content": ("gc_content", float),
    "Total_Coding_Sequences": ("coding_sequences", int),
    "Total_Contigs": ("contig_count", int),
    "Max_Contig_Length": ("max_contig_length", int),
    "Cosine_Similarity": ("cosine_similarity", float),
    "Additional_Notes": ("notes", str),
}


class RNAEvidence(TypedDict):
    """Marker calls, distinct amino acid types and the resulting RNA verdict."""

    rrna_5s: bool | None
    rrna_16s: bool | None
    rrna_23s: bool | None
    trna_types: int | None
    status: str


class Quality(TypedDict):
    """Optional external metrics and an explicit three-way HQ status."""

    completeness: float | None
    contamination: float | None
    hq_status: str
    rna: RNAEvidence
    checkm2: dict[str, str | int | float]
    reported_hq_status: NotRequired[str]
    hq_conflicts: NotRequired[list[str]]


AssessmentKey = tuple[str, str]
AssessmentRecipe = tuple[str, str, str, str, str]
AssessmentRecipes = dict[AssessmentKey, dict[str, set[AssessmentRecipe]]]


def table(path: Path, required: set[str]) -> Iterator[dict[str, str]]:
    """Read a UTF-8 TSV with explicit required columns and well-formed rows."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = reader.fieldnames or []
        if not required <= set(columns) or len(columns) != len(set(columns)):
            raise ValueError(f"Invalid TSV columns in {path}; expected {', '.join(sorted(required))}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Invalid TSV row in {path} at line {reader.line_num}")
            yield row


def text_field(value: Any, label: str) -> str:
    """Require a nonempty text value from a report configuration."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def number(value: Any, label: str) -> float:
    """Read a finite number without accepting booleans as numbers."""
    try:
        result = float(value) if not isinstance(value, bool) else float("nan")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid number for {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number for {label}")
    return result


def _percent(value: Any, label: str) -> float:
    result = number(value, label)
    if result < 0 or (label == "completeness" and result > 100):
        raise ValueError(f"Out-of-range {label}: {result}")
    return result


def policy(raw: Any) -> dict[str, Any]:
    """Validate a report's externally declared or explicit numeric HQ policy."""
    if not isinstance(raw, dict) or raw.get("mode") not in ("external", "thresholds"):
        raise ValueError("hq_policy.mode must be external or thresholds")
    if raw["mode"] == "external":
        external = {
            "mode": "external",
            "description": text_field(raw.get("description"), "HQ policy description"),
        }
        if "standard" in raw:
            if raw["standard"] != "mimag":
                raise ValueError("The supported standard for external verdicts is mimag")
            external["standard"] = "mimag"
        return external
    result: dict[str, Any] = {"mode": "thresholds"}
    descriptions = []
    for metric, operators in (("completeness", (">", ">=")), ("contamination", ("<", "<="))):
        rule = raw.get(metric)
        if not isinstance(rule, dict) or rule.get("operator") not in operators:
            raise ValueError(f"hq_policy.{metric} requires an operator from {operators}")
        threshold = _percent(rule.get("value"), metric)
        result[metric] = {"operator": rule["operator"], "value": threshold}
        descriptions.append(f"{metric.title()} {rule['operator']} {threshold:g}%")
    result["description"] = " and ".join(descriptions)
    return result


def quality_for(evidence: Quality | None, hq_policy: dict[str, Any]) -> Quality:
    """Resolve HQ locally under the fixed report policy, retaining unknowns."""
    quality = evidence.copy() if evidence is not None else _metrics("", "", "")
    rna = quality["rna"].copy()
    markers = (rna["rrna_5s"], rna["rrna_16s"], rna["rrna_23s"])
    counts = rna["trna_types"]
    if None not in markers and counts is not None:
        rna["status"] = "pass" if all(markers) and counts >= 18 else "fail"
    quality["rna"] = rna
    completeness, contamination = quality["completeness"], quality["contamination"]
    if hq_policy["mode"] == "thresholds":
        if completeness is not None and contamination is not None:
            rules = (hq_policy["completeness"], hq_policy["contamination"])
            passes = all(
                COMPARISONS[rule["operator"]](value, rule["value"])
                for rule, value in zip(rules, (completeness, contamination))
            )
            quality["hq_status"] = "pass" if passes else "fail"
        else:
            quality["hq_status"] = "unknown"
    elif hq_policy.get("standard") == "mimag" and quality["hq_status"] == "pass":
        conflicts = [
            label for label, value in zip(("5S rRNA", "16S rRNA", "23S rRNA"), markers) if value is False
        ]
        if counts is not None and counts < 18:
            conflicts.append("tRNA amino acid types")
        if completeness is not None and completeness <= 90:
            conflicts.append("completeness")
        if contamination is not None and contamination >= 5:
            conflicts.append("contamination")
        if conflicts:
            quality["hq_status"] = "unknown"
            quality["reported_hq_status"] = "pass"
            quality["hq_conflicts"] = conflicts
    return quality


def _metrics(completeness: str, contamination: str, status: str) -> Quality:
    status = status.strip().lower() or "unknown"
    if status not in HQ_STATUSES:
        raise ValueError(f"Invalid HQ status: {status!r}")
    return {
        "completeness": _percent(completeness, "completeness") if completeness.strip() else None,
        "contamination": _percent(contamination, "contamination") if contamination.strip() else None,
        "hq_status": status,
        "rna": {"rrna_5s": None, "rrna_16s": None, "rrna_23s": None, "trna_types": None, "status": "unknown"},
        "checkm2": {},
    }


def _rna_call(value: str) -> bool | None:
    calls = {"detected": True, "not_detected": False, "not_assessed": None, "": None}
    if value.strip().lower() not in calls:
        raise ValueError(f"Invalid rRNA call: {value!r}; use detected, not_detected or not_assessed")
    return calls[value.strip().lower()]


def _tabular_metrics(row: dict[str, str]) -> Quality:
    quality = _metrics(row.get("completeness", ""), row.get("contamination", ""), row.get("hq_status", ""))
    count = row.get("trna_types", "").strip()
    types = int(count) if count else None
    if types is not None and not 0 <= types <= 20:
        raise ValueError("trna_types must count 0 to 20 distinct amino acid types")
    quality["rna"] = {
        "rrna_5s": _rna_call(row.get("rrna_5s", "")),
        "rrna_16s": _rna_call(row.get("rrna_16s", "")),
        "rrna_23s": _rna_call(row.get("rrna_23s", "")),
        "trna_types": types,
        "status": "unknown",
    }
    return quality


def checked_export(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify an export and the two records required for reporting."""
    receipt = read(root / "complete.json")
    if receipt.get("kind") != "export" or not {"export.json", "bins.tsv"} <= set(receipt["outputs_sha256"]):
        raise ValueError(f"Expected a completed export artifact: {root}")
    verify_outputs(root, receipt)
    return read(root / "export.json"), receipt


def _checkm2_rows(
    path: Path,
    root: Path,
    dataset_path: Path,
    bins: set[str],
    proposals: dict[str, dict[str, Any]],
    completeness_column: str,
) -> Iterator[tuple[str, str, Quality]]:
    export, receipt = checked_export(root)
    if Path(export["dataset"]).resolve() != dataset_path:
        raise ValueError(f"CheckM2 export belongs to a different prepared dataset: {root}")
    changed = {row["proposal_id"]: row for row in export["changed"]}
    unchanged = {row["recipient"]: row for row in export["unchanged"]}
    aliases: dict[str, tuple[str, str]] = {}
    conflicts = set()
    ids_by_file: dict[tuple[str, str], set[str]] = {}
    file_by_id: dict[tuple[str, str], tuple[str, str]] = {}
    for row in table(root / "bins.tsv", {"recipient", "proposal_id", "fasta", "sha256", "changed"}):
        if not row["fasta"]:
            continue
        recipient, proposal = row["recipient"], row["proposal_id"]
        original = unchanged.get(recipient)
        if (
            not proposal
            and original is not None
            and row["changed"] == "false"
            and "final_bin" not in original
        ):
            continue
        fasta = (root / row["fasta"]).resolve()
        if (
            not fasta.is_relative_to(root)
            or receipt["outputs_sha256"].get(fasta.relative_to(root).as_posix()) != row["sha256"]
        ):
            raise ValueError(f"CheckM2 input FASTA must be covered by the export receipt: {row['fasta']!r}")
        if proposal:
            saved = changed.get(proposal)
            known = proposals.get(proposal)
            if (
                row["changed"] != "true"
                or saved is None
                or known is None
                or known["recipient"] != recipient
                or saved["recipient"] != recipient
                or saved["sha256"] != row["sha256"]
                or saved.get("final_bin", {}).get("path", saved["path"]) != row["fasta"]
            ):
                raise ValueError(f"Invalid proposal mapping in {root}: {proposal}")
            kind, identity = "proposal", proposal
        else:
            if (
                row["changed"] != "false"
                or original is None
                or recipient not in bins
                or original.get("final_bin", {}).get("path") != row["fasta"]
            ):
                raise ValueError(f"Invalid original mapping in {root}: {recipient}")
            kind, identity = "original", recipient
        name = PureWindowsPath(row["fasta"]).name
        group = kind, row["fasta"]
        if (kind, identity) in file_by_id:
            raise ValueError(f"Repeated bin mapping in {root}: {identity}")
        file_by_id[kind, identity] = group
        ids_by_file.setdefault(group, set()).add(identity)
        for alias in {name, Path(name).stem}:
            if aliases.setdefault(alias, group) != group:
                conflicts.add(alias)

    seen = set()
    for row in table(path, {"Name", completeness_column, "Contamination"}):
        name = row["Name"]
        if name in conflicts or name not in aliases:
            raise ValueError(f"CheckM2 Name is unknown or ambiguous in {root}: {name!r}")
        group = aliases[name]
        if group in seen:
            raise ValueError(f"Repeated CheckM2 row for FASTA {name!r} in {path}")
        seen.add(group)
        metrics = _metrics(row[completeness_column], row["Contamination"], "")
        for column, (field, datatype) in CHECKM2_FIELDS.items():
            value = row.get(column, "").strip()
            if value and value.lower() not in ("na", "nan", "n/a"):
                metrics["checkm2"][field] = number(value, column) if datatype is float else datatype(value)
        for identity in sorted(ids_by_file[group]):
            yield group[0], identity, metrics


def has_evidence(quality: Quality) -> bool:
    """Return whether at least one quality or RNA measurement was supplied."""
    return (
        quality["completeness"] is not None
        or quality["contamination"] is not None
        or quality["hq_status"] != "unknown"
        or bool(quality["checkm2"])
        or any(value is not None for key, value in quality["rna"].items() if key != "status")
    )


def _completeness_column(path: Path, selected: Any) -> str:
    options = ("Completeness", "Completeness_General", "Completeness_Specific")
    if selected is not None:
        if selected not in options:
            raise ValueError("Invalid CheckM2 completeness_column")
        return str(selected)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"), [])
    columns = set(header).intersection(options)
    if len(columns) != 1:
        raise ValueError(f"Select one CheckM2 completeness column explicitly for {path}")
    return columns.pop()


def _record_recipes(
    fields: dict[str, set[AssessmentRecipe]], quality: Quality, source: dict[str, str]
) -> None:
    recipe: AssessmentRecipe = (
        source["format"],
        source["tool"],
        source["version"],
        source["settings"],
        source.get("completeness_column", ""),
    )
    supplied = {
        "completeness": quality["completeness"],
        "contamination": quality["contamination"],
        "hq_status": quality["hq_status"],
        **{field: value for field, value in quality["rna"].items() if field != "status"},
    }
    for field, value in supplied.items():
        if value is not None and value != "unknown":
            fields.setdefault(field, set()).add(recipe)


def _merge(previous: Quality, current: Quality, label: str) -> None:
    for target, source in (
        (previous, current),
        (previous["rna"], current["rna"]),
        (previous["checkm2"], current["checkm2"]),
    ):
        fields = cast(dict[str, Any], target)
        for field, value in source.items():
            if field in ("rna", "checkm2") or value is None or value == "unknown":
                continue
            if fields.get(field) not in (None, "unknown", value):
                raise ValueError(f"Conflicting {field} evidence in {label}")
            fields[field] = value


def read_assessments(
    sources: Any,
    base: Path,
    dataset_path: Path,
    bins: set[str],
    proposals: dict[str, dict[str, Any]],
) -> tuple[dict[AssessmentKey, Quality], list[dict[str, str]], AssessmentRecipes]:
    """Combine compatible fields under original IDs or exact proposal sequence identities."""
    if not isinstance(sources, list):
        raise ValueError("assessments must be a list")
    evidence: dict[AssessmentKey, Quality] = {}
    recipes: AssessmentRecipes = {}
    provenance = []
    for source in sources:
        if not isinstance(source, dict) or source.get("format") not in ("checkm2", "tsv"):
            raise ValueError("Each assessment format must be checkm2 or tsv")
        path = base / text_field(source.get("report"), "assessment report path")
        details = {
            name: text_field(source.get(name), f"assessment {name}")
            for name in ("tool", "version", "settings")
        }
        details.update(format=source["format"], report_sha256=sha(path))
        if source["format"] == "checkm2":
            root = (base / text_field(source.get("export"), "CheckM2 assessment export path")).resolve()
            details["export_receipt_sha256"] = sha(root / "complete.json")
            column = _completeness_column(path, source.get("completeness_column"))
            details["completeness_column"] = column
            rows = _checkm2_rows(path, root, dataset_path, bins, proposals, column)
        else:
            rows = ((row["kind"], row["id"], _tabular_metrics(row)) for row in table(path, {"kind", "id"}))
        seen = set()
        for kind, identity, metrics in rows:
            if (kind, identity) in seen:
                raise ValueError(f"Repeated assessment ID in {path}: {kind} {identity!r}")
            seen.add((kind, identity))
            if kind == "original" and identity in bins:
                key = kind, identity
            elif kind == "proposal" and identity in proposals:
                key = kind, proposals[identity]["sequence_set_id"]
            else:
                raise ValueError(f"Unknown assessment ID in {path}: {kind} {identity!r}")
            previous = evidence.setdefault(key, _metrics("", "", ""))
            _merge(previous, metrics, f"{path}: {kind} {identity!r}")
            _record_recipes(recipes.setdefault(key, {}), metrics, details)
        details["scope"] = ", ".join(sorted({kind for kind, _ in seen}))
        provenance.append(details)
    return evidence, provenance, recipes
