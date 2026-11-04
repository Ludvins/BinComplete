"""Structural selection of externally approved proposals, without quality policies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from .artifacts import complete, fingerprint, read, save, verify_outputs
from .engine import ProposalArtifact, _new_or_same
from .resources import FileLock, ResourceLimits


@dataclass(frozen=True)
class SelectionArtifact:
    """Reference to structurally compatible externally approved proposals.

    Parameters
    ----------
    path : Path
        Completed selection artifact directory.
    """

    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the verified selection and optimization certificate."""
        verify_outputs(self.path, read(self.path / "complete.json"))
        return read(self.path / "selection.json")


def load_proposals(artifacts: list[ProposalArtifact]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read verified proposal memberships from one common prepared population."""
    proposals = {}
    population = None
    for artifact in artifacts:
        manifest = artifact.manifest
        if population is not None and population["dataset_identity"] != manifest["dataset_identity"]:
            raise ValueError("Reconcile requires the same prepared recipient population")
        population = manifest
        with (artifact.path / "memberships.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = row["proposal_id"]
                if key in proposals and proposals[key] != row:
                    raise ValueError("Conflicting proposal identity")
                proposals[key] = row
    if population is None:
        raise ValueError("Supply at least one proposal artifact")
    return proposals, population


def reconcile(
    artifacts: list[ProposalArtifact],
    approved_ids: list[str],
    output: str | Path,
    *,
    protected_bins: list[str] | None = None,
    time_limit: float = 60.0,
    limits: ResourceLimits | None = None,
) -> SelectionArtifact:
    """Maximize compatible original recipients among externally approved proposals.

    Parameters
    ----------
    artifacts : list of ProposalArtifact
        Saved proposal artifacts for a common prepared dataset.
    approved_ids : list of str
        Proposal IDs approved by external evaluation. No metrics are interpreted.
    output : path-like
        Selection artifact directory.
    protected_bins : list of str, optional
        Original bins that must remain unchanged.
    time_limit : float
        Whole optimization budget in seconds, including deterministic tie breaking.
    limits : ResourceLimits, optional
        Allocation and solver thread limits.

    Returns
    -------
    SelectionArtifact
        Compatible membership witnesses with an explicit optimality status.

    Notes
    -----
    Primary objective: maximum number of distinct original recipients. Among
    proven optima, prefer including the lexically first proposal ID, then the
    next. An expired tie-break budget is disclosed rather than called canonical.
    """
    limits = limits or ResourceLimits()
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    all_rows, population = load_proposals(artifacts)
    if len(approved_ids) != len(set(approved_ids)) or not set(approved_ids) <= set(all_rows):
        raise ValueError("Approved IDs must be unique saved proposal IDs")
    protected = set(protected_bins or [])
    rows = [all_rows[key] for key in sorted(approved_ids) if all_rows[key]["recipient"] not in protected]
    limits.require(
        sum(len(row["members"]) for row in rows) * 128 + len(rows) * 1024, "Compatibility constraint matrices"
    )
    identity = fingerprint(
        {
            "artifacts": [artifact.manifest for artifact in artifacts],
            "approved_ids": sorted(approved_ids),
            "protected": sorted(protected),
            "time_limit": time_limit,
            "selection_version": 1,
        }
    )
    root = Path(output).resolve()
    with FileLock(root.with_name(root.name + ".selection.lock")):
        if not _new_or_same(root, identity):
            return SelectionArtifact(root)
        keys: dict[tuple[str, str], int] = {}
        rr, cc = [], []
        for column, row in enumerate(rows):
            for key in [("recipient", row["recipient"])] + [("member", name) for name in row["members"]]:
                if key not in keys:
                    keys[key] = len(keys)
                rr.append(keys[key])
                cc.append(column)
        chosen = np.zeros(len(rows), bool)
        optimal = canonical = True
        upper_bound = 0.0
        started = time.monotonic()
        if rows:
            matrix = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(keys), len(rows))).tocsc()
            constraints = LinearConstraint(matrix, -np.inf, 1)

            def solve(cost: np.ndarray, lower: np.ndarray, upper: np.ndarray, constraint):
                import warnings

                remaining = max(0.001, time_limit - (time.monotonic() - started))
                # SciPy passes this supported HiGHS option through to its solver.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
                    return milp(
                        cost,
                        integrality=np.ones(len(rows)),
                        bounds=Bounds(lower, upper),
                        constraints=constraint,
                        options={"time_limit": remaining, "threads": 1, "mip_rel_gap": 0.0},
                    )

            lower, upper = np.zeros(len(rows)), np.ones(len(rows))
            result = solve(-np.ones(len(rows)), lower, upper, constraints)
            if result.status not in (0, 1):
                raise RuntimeError(f"Compatibility solver failed: {result.message}")
            if result.x is not None:
                chosen = result.x > 0.5
            else:
                # A zero allocation remains a valid witness even if admission times out.
                chosen = np.zeros(len(rows), bool)
            optimal = result.status == 0
            upper_bound = (
                float(-result.mip_dual_bound)
                if getattr(result, "mip_dual_bound", None) is not None
                else float(len({row["recipient"] for row in rows}))
            )
            canonical = optimal
            if optimal:
                cardinality = int(chosen.sum())
                equality = coo_matrix(np.ones((1, len(rows)))).tocsc()
                tied_matrix = vstack([matrix, equality], format="csc")
                tied = LinearConstraint(
                    tied_matrix,
                    np.r_[np.full(len(keys), -np.inf), cardinality],
                    np.r_[np.ones(len(keys)), cardinality],
                )
                for column in range(len(rows)):
                    if time.monotonic() - started >= time_limit:
                        canonical = False
                        break
                    if chosen[column]:
                        lower[column] = upper[column] = 1
                        continue
                    trial = lower.copy()
                    trial[column] = 1
                    attempt = solve(np.zeros(len(rows)), trial, upper, tied)
                    if attempt.x is not None:
                        chosen = attempt.x > 0.5
                        lower[column] = upper[column] = 1
                    elif attempt.status == 2:
                        upper[column] = 0
                    else:
                        canonical = False
                        break
        selected = [row for row, include in zip(rows, chosen) if include]
        used_members: set[str] = set()
        recipients: set[str] = set()
        for row in selected:
            if row["recipient"] in recipients or used_members.intersection(row["members"]):
                raise RuntimeError("Solver returned an incompatible witness")
            recipients.add(row["recipient"])
            used_members.update(row["members"])
        save(
            root / "selection.json",
            {
                "schema_version": 1,
                "dataset": population["dataset"],
                "dataset_identity": population["dataset_identity"],
                "approved_recipient_union": len({all_rows[key]["recipient"] for key in approved_ids}),
                "compatible_recipients": len(selected),
                "optimal": optimal,
                "canonical_tie_break": canonical,
                "upper_bound": upper_bound,
                "protected_bins": sorted(protected),
                "selected": selected,
                "approval_semantics": "External verdicts; structural compatibility only",
            },
        )
        complete(root, {"kind": "selection", "identity": identity})
        return SelectionArtifact(root)
