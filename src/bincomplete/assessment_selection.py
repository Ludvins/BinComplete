"""Schedule one alternative per bin or cover the complete eligible catalog."""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import fingerprint, save
from .assessment_inputs import Catalog
from .assessment_tools import POLICY, full_hq
from .reconcile import reconcile
from .resources import ResourceLimits


def _priority(name: str, row: dict[str, Any]) -> tuple[int, float, float, str]:
    c, x = row["completeness"], row["contamination"]
    if row["status"] != "SCORED":
        return (3, 0, 0, name)
    if x >= 5:
        return (2, x, -c, name)
    return (0 if 85 <= c <= 90 and row["rna"]["pass"] else 1, -c, x, name)


@dataclass
class Assessment:
    catalog: Catalog
    baseline: dict[str, Any]
    checked: dict[str, Any] = field(default_factory=dict)
    blocked: dict[str, str] = field(init=False)
    eligible: dict[str, list[str]] = field(init=False)
    selected: dict[str, str] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)
    cursor: Counter[str] = field(default_factory=Counter)
    unscored: int = field(default=0, init=False)
    imported: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.blocked = {
            name: "ORIGINAL_HQ" if full_hq(row) else "BASELINE_UNSCORED"
            for name, row in self.baseline.items()
            if full_hq(row) or row["status"] != "SCORED"
        }
        groups: dict[str, list[str]] = defaultdict(list)
        for key, row in sorted(self.catalog.rows.items(), key=lambda item: item[1]["priority"]):
            if row["recipient"] not in self.blocked:
                groups[row["recipient"]].append(key)
        self.eligible = {
            name: groups[name] for name in sorted(groups, key=lambda n: _priority(n, self.baseline[n]))
        }

    def add(self, manifest: dict[str, Any], results: dict[str, Any], context: str, imported: bool) -> None:
        controls = {
            row["recipient"]: results[name] | {"rna": self.baseline[row["recipient"]]["rna"]}
            for name, row in manifest.items()
            if name.startswith("o") and row["sha256"] == self.baseline[row["recipient"]]["sha256"]
        }
        for name, row in manifest.items():
            if not name.startswith("c"):
                continue
            recipient = row["recipient"]
            key = row["construction_id"]
            if key in self.checked or recipient not in self.eligible or recipient not in controls:
                raise ValueError(f"Repeated, ineligible or unmatched construction: {key}")
            control = controls[recipient]
            if full_hq(control):
                self.blocked[recipient] = "CONTROL_DISCORDANT"
            elif control["status"] != "SCORED":
                self.blocked.setdefault(recipient, "CONTROL_UNSCORED")
            self.checked[key] = {
                "candidate": results[name],
                "control": control,
                "context": context,
                "fasta_sha256": row["sha256"],
                "imported": imported,
            }
            self.unscored += results[name]["status"] != "SCORED"
            self.imported += imported

    def next_round(self) -> list[str]:
        keys = []
        for recipient, candidates in self.eligible.items():
            if recipient in self.blocked or recipient in self.selected:
                continue
            index = self.cursor[recipient]
            while index < len(candidates):
                key = candidates[index]
                if key not in self.checked and self.used.isdisjoint(self.catalog.rows[key]["members"]):
                    keys.append(key)
                    break
                index += 1
            self.cursor[recipient] = index
        return keys

    def accept(self, keys: list[str]) -> None:
        for key in keys:
            row = self.catalog.rows[key]
            name = row["recipient"]
            if (
                name not in self.blocked
                and full_hq(self.checked[key]["candidate"])
                and self.used.isdisjoint(row["members"])
            ):
                self.selected[name] = key
                self.used.update(row["members"])

    def full_order(self) -> list[str]:
        order = {name: index for index, name in enumerate(self.eligible)}
        return sorted(
            (key for keys in self.eligible.values() for key in keys if key not in self.checked),
            key=lambda key: (
                self.catalog.rows[key]["priority"][0],
                order[self.catalog.rows[key]["recipient"]],
                *self.catalog.rows[key]["priority"][1:],
            ),
        )

    def verdict(self, key: str, mode: str) -> str:
        row = self.catalog.rows[key]
        recipient = row["recipient"]
        if key in self.checked:
            result = self.checked[key]["candidate"]
            if result["status"] != "SCORED":
                return "UNSCORED"
            if recipient in self.blocked:
                return self.blocked[recipient]
            return "HQ_PASS" if full_hq(result) else "HQ_FAIL"
        if recipient in self.blocked and (
            mode == "greedy" or self.blocked[recipient] in {"ORIGINAL_HQ", "BASELINE_UNSCORED"}
        ):
            return f"SKIPPED_{self.blocked[recipient]}"
        if mode == "greedy" and recipient in self.selected:
            return "SKIPPED_REPLACED"
        if mode == "greedy" and not self.used.isdisjoint(row["members"]):
            return "SKIPPED_CONTIG_CONFLICT"
        return "UNASSESSED"

    def publish_selection(
        self,
        root: Path,
        mode: str,
        seconds: int,
        limits: ResourceLimits,
    ) -> dict[str, Any]:
        keys = (
            self.selected.values()
            if mode == "greedy"
            else [key for key in self.checked if self.verdict(key, mode) == "HQ_PASS"]
        )
        approved = sorted(min(self.catalog.rows[key]["proposals"]) for key in keys)
        protected = sorted(self.blocked)
        target = (
            root
            / "selection"
            / fingerprint(
                {
                    "approved": approved,
                    "protected": protected,
                    "solver_seconds": seconds,
                    "mode": mode,
                }
            )
        )
        target.mkdir(parents=True, exist_ok=True)
        completed = sorted(target.glob("attempt-*/complete.json"))
        path = completed[0].parent if completed else Path(tempfile.mkdtemp(prefix="attempt-", dir=target))
        result = reconcile(
            self.catalog.artifacts,
            approved,
            path,
            protected_bins=protected,
            time_limit=seconds,
            limits=limits,
            fixed=mode == "greedy",
        ).manifest
        by_proposal = {min(self.catalog.rows[key]["proposals"]): key for key in keys}
        self.selected = {row["recipient"]: by_proposal[row["proposal_id"]] for row in result["selected"]}
        return {
            "path": path.relative_to(root).as_posix(),
            "optimal": result["optimal"],
            "upper_bound": result["upper_bound"],
            "canonical_tie_break": result["canonical_tie_break"],
            "scope": "fixed greedy approvals" if mode == "greedy" else "eligible approved constructions",
        }


def _tsv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> None:
    partial = path.with_suffix(".partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    partial.replace(path)


def _measures(row: dict[str, Any]) -> list[Any]:
    rna = row["rna"]
    return [
        row["completeness"],
        row["contamination"],
        json.dumps(rna, separators=(",", ":")) if rna else "",
        row["status"],
        int(full_hq(row)),
    ]


MEASURES = ["completeness", "contamination", "rna_checks", "checkm2_status", "hq"]


def write_baseline(root: Path, baseline: dict[str, Any]) -> None:
    _tsv(
        root / "baseline.tsv",
        ["bin_id", *MEASURES, "fasta_sha256"],
        ([name, *_measures(row), row["sha256"]] for name, row in sorted(baseline.items())),
    )


def write_reports(root: Path, state: Assessment, mode: str) -> None:
    catalog = state.catalog.rows
    baseline = state.baseline

    def measurements(prefix: str) -> list[str]:
        return [prefix + column for column in MEASURES]

    def rows() -> Iterable[list[Any]]:
        for key, row in catalog.items():
            name = row["recipient"]
            item = state.checked.get(key)
            yield [
                name,
                key,
                row["sequence_set_id"],
                json.dumps(row["proposals"], sort_keys=True),
                *row["priority"][:2],
                state.verdict(key, mode),
                state.blocked.get(name, ""),
                int(bool(item and item["imported"])),
                item["context"] if item else "",
                item["fasta_sha256"] if item else "",
                *_measures(baseline[name]),
                *(_measures(item["control"]) if item else [""] * len(MEASURES)),
                *(_measures(item["candidate"]) if item else [""] * len(MEASURES)),
            ]

    _tsv(
        root / "assessment.tsv",
        [
            "bin_id",
            "construction_id",
            "sequence_set_id",
            "source_proposals",
            "assessment_tier",
            "priority_rank",
            "status",
            "control_status",
            "imported",
            "matched_batch",
            "fasta_sha256",
            *measurements("baseline_"),
            *measurements("matched_"),
            *measurements("candidate_"),
        ],
        rows(),
    )

    def before_after() -> Iterable[list[Any]]:
        for name, original in sorted(baseline.items()):
            key = state.selected.get(name)
            item = state.checked[key] if key else None
            pid = min(catalog[key]["proposals"]) if key else ""
            yield [
                name,
                *_measures(original),
                key or "",
                pid,
                catalog[key]["proposals"][pid] if key else "",
                *(_measures(item["candidate"]) if item else _measures(original)),
                item["context"] if item else "",
                int(key is not None),
                state.blocked.get(name, ""),
            ]

    _tsv(
        root / "before-after.tsv",
        [
            "bin_id",
            *MEASURES,
            "construction_id",
            "proposal_id",
            "method",
            *measurements("new_"),
            "matched_batch",
            "hq_gain",
            "control_status",
        ],
        before_after(),
    )


def write_summary(
    root: Path,
    state: Assessment,
    mode: str,
    status: str,
    rounds: int,
    batches: int,
    selection: dict[str, Any] | None = None,
    reason: str | None = None,
    *,
    last_selection: dict[str, Any] | None = None,
) -> None:
    baseline = sum(map(full_hq, state.baseline.values()))
    eligible = sum(map(len, state.eligible.values()))
    counts = (
        None if status == "IN_PROGRESS" else Counter(state.verdict(key, mode) for key in state.catalog.rows)
    )
    save(
        root / "summary.json",
        {
            "state": status,
            "stop_reason": reason,
            "mode": mode,
            "policy": POLICY,
            "rounds_completed": rounds,
            "batches_completed": batches,
            "original_bins": len(state.baseline),
            "baseline_hq": baseline,
            "compatible_hq_gains": len(state.selected),
            "panel_plus_gains": baseline + len(state.selected),
            "catalog_constructions": len(state.catalog.rows),
            "eligible_constructions": eligible,
            "checked_constructions": len(state.checked),
            "unchecked_eligible_constructions": eligible - len(state.checked),
            "unscored_constructions": state.unscored,
            "imported_constructions": state.imported,
            "status_counts": dict(counts) if counts is not None else None,
            "control_discordant_bins": sorted(
                name for name, reason in state.blocked.items() if reason == "CONTROL_DISCORDANT"
            ),
            "selection": selection,
            "last_published_selection": last_selection,
            "quality_status": "Computational HQ with at least 18 canonical tRNA types",
        },
    )
