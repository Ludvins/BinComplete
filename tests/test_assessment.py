"""Drive the public assessment command with synthetic external executables."""

import csv
import json
import os
import sys

import numpy as np
import pytest

from bincomplete.artifacts import complete, fingerprint, read, save, sha
from bincomplete.cli import main
from bincomplete.engine import ProposalArtifact
from bincomplete.inventory import candidate_table
from bincomplete.resources import ResourceLimits

FAKE_TOOLS = """
import sys
from pathlib import Path

kind, events, switch, *args = sys.argv[1:]
events, switch = Path(events), Path(switch)
if args[0] in ("--version", "--help"):
    print({"checkm2": "1.1.0", "barrnap": "0.9", "trnascan": "2.0.12"}[kind])
    raise SystemExit(0)

def members(path):
    return sorted(line[1:] for line in path.read_text().splitlines() if line.startswith(">"))

with events.open("a") as stream:
    stream.write(kind + "\\n")

if kind == "checkm2":
    inputs = Path(args[args.index("--input") + 1])
    output = Path(args[args.index("--output-directory") + 1])
    output.mkdir()
    rows = ["Name\\tCompleteness\\tContamination"]
    logs = []
    candidate_batch = any(inputs.glob("c*.fa"))
    for fasta in sorted(inputs.glob("*.fa")):
        ids = members(fasta)
        if "c6" in ids:
            protein = output / "protein_files" / (fasta.stem + ".faa")
            protein.parent.mkdir(exist_ok=True)
            protein.touch()
            logs.append(f"Skipping protein file {protein} as it was empty.")
            continue
        value = 95 if "c4" in ids or len(ids) > 2 else 89
        if candidate_batch and "c2" in ids and len(ids) == 2 and switch.read_text() == "discordant":
            value = 95
        if "c11" in ids:
            value = 90
        rows.append(f"{fasta.stem}\\t{value}\\t1")
    (output / "checkm2.log").write_text("\\n".join(logs))
    if len(rows) == 1:
        print("in __calculate_metadata\\nself.total_threads / len(faa_files)\\n"
              "ZeroDivisionError: division by zero", file=sys.stderr)
        raise SystemExit(1)
    if candidate_batch and switch.read_text() == "mutate":
        next(inputs.glob("c*.fa")).write_text(">changed\\nACGT\\n")
    (output / "quality_report.tsv").write_text("\\n".join(rows) + "\\n")
elif kind == "barrnap":
    if Path(args[-1]).name.startswith("c") and switch.read_text() == "fail":
        raise SystemExit(7)
    print("##gff-version 3")
    if args[args.index("--kingdom") + 1] == "bac":
        for name in ("5S", "16S", "23S"):
            print(f"contig\\tbarrnap\\trRNA\\t1\\t100\\t0\\t+\\t.\\tName={name}_rRNA")
else:
    output = Path(args[args.index("-o") + 1])
    if "-A" in args and switch.read_text() == "zero":
        output.touch()
        print("No tRNAs found.", file=sys.stderr)
        raise SystemExit(0)
    if "-A" in args and switch.read_text() == "partial":
        print("Error: Fail to run infernal for contig")
    header = "Sequence\\t\\ttRNA\\tBounds\\ttRNA\\tAnti\\tIntron Bounds\\tInf\\n"
    header += "Name\\ttRNA #\\tBegin\\tEnd\\tType\\tCodon\\tBegin\\tEnd\\tScore\\tNote\\n"
    header += "--------\\t------\\t-----\\t-----\\t----\\t-----\\t-----\\t----\\t------\\n"
    offset = 0 if "-B" in args else 9
    stop = 18 if switch.read_text() == "zero" else offset + 9
    rows = [f"contig\\t{i + 1}\\t1\\t40\\tLabel{i}\\tTAG\\t0\\t0\\t50.0\\n" for i in range(offset, stop)]
    output.write_text(header + "".join(rows))
"""


@pytest.fixture
def assessment_run(dataset, tmp_path, limits):
    root = tmp_path / "experiment"
    root.mkdir()
    model = root / "model"
    save(model / "model.json", {"kind": "synthetic"})
    complete(model, {"kind": "model"})
    # A takes shared c8 first. B needs c8. A's c9 alternative lets both pass.
    proposal_sets = [
        (
            "cosine",
            [
                ("a-first", "b0", "c8"),
                ("a-second", "b0", "c9"),
                ("b-first", "b1", "c8"),
                ("b-fail", "b1", "c11"),
                ("protected", "b2", "c10"),
                ("unscored", "b3", "c10"),
            ],
        ),
        ("kmers", [("duplicate", "b0", "c8")]),
    ]
    population = dataset.manifest
    methods = {}
    artifacts = []
    assembly_ids = list(np.load(dataset.array_path("ids.npy"), allow_pickle=False))
    for method, proposals in proposal_sets:
        path = root / method
        path.mkdir()
        rankings = {}
        rows = []
        for pid, recipient, donor in proposals:
            rows.append(
                {
                    "proposal_id": pid,
                    "recipient": recipient,
                    "method": method,
                    "added": [donor],
                    "members": sorted([*population["bins"][recipient], donor]),
                }
            )
            if recipient not in rankings:
                folder = path / recipient
                folder.mkdir()
                np.save(
                    folder / "rows.npy", [assembly_ids.index(name) for name in ("c8", "c9", "c10", "c11")]
                )
                np.save(folder / "scores.npy", [0.99, 0.98, 0.97, 0.96])
                rankings[recipient] = {"ranking": recipient}
        (path / "memberships.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        save(
            path / "proposals.json",
            {
                "method": method,
                "dataset": str(dataset.path),
                "dataset_identity": population["identity"],
                "coverage": rankings,
            },
        )
        complete(path, {"kind": "proposals"})
        artifacts.append(ProposalArtifact(path))
        methods[method] = {
            "status": "SUCCEEDED",
            "model": str(model),
            "proposals": str(path),
            "model_receipt_sha256": sha(model / "complete.json"),
            "proposals_receipt_sha256": sha(path / "complete.json"),
        }
    candidate_table(artifacts, root / "candidates.tsv", limits=limits)
    save(
        root / "summary.json",
        {
            "status": "SUCCEEDED",
            "methods": methods,
            "datasets": {
                "sample": {
                    "path": str(dataset.path),
                    "identity": population["identity"],
                    "receipt_sha256": sha(dataset.path / "complete.json"),
                }
            },
            "assessment_priority": {"receipt_sha256": sha(root / "candidates.tsv.json")},
        },
    )
    complete(root, {"kind": "experiment"})

    script = tmp_path / "fake_tools.py"
    events = tmp_path / "calls"
    switch = tmp_path / "switch"
    switch.write_text("normal", encoding="utf-8")
    script.write_text(FAKE_TOOLS, encoding="utf-8")
    (tmp_path / "database.dmnd").write_bytes(b"test database")
    config = tmp_path / "quality.toml"
    config.write_text(
        'checkm2_database = "database.dmnd"\nruntime_files = ["fake_tools.py"]\nbatch_size = 1\n'
        "checkm2_threads = 1\nrna_threads = 1\n[commands]\n"
        + "".join(
            f"{name} = {json.dumps([sys.executable, str(script), name, str(events), str(switch)])}\n"
            for name in ("checkm2", "barrnap", "trnascan")
        ),
        encoding="utf-8",
    )
    return root, config, events, switch


def assess(run, output, mode, *options):
    experiment, config, _, _ = run
    return main(
        [
            "assess",
            "--experiment",
            str(experiment),
            "--config",
            str(config),
            "--output",
            str(output),
            "--mode",
            mode,
            *options,
        ]
    )


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_greedy_round_spans_batches_and_full_reuses_evidence(assessment_run, tmp_path):
    greedy, full = tmp_path / "greedy", tmp_path / "full"
    assert assess(assessment_run, greedy, "greedy") == 2
    assert assess(assessment_run, greedy, "greedy", "--max-rounds", "1") == 0
    report = read(greedy / "summary.json")
    assert (report["baseline_hq"], report["compatible_hq_gains"], report["checked_constructions"]) == (
        1,
        1,
        2,
    )
    assert report["state"] == "STOPPED"
    rows = table(greedy / "assessment.tsv")
    duplicated = next(row for row in rows if "duplicate" in row["source_proposals"])
    assert json.loads(duplicated["source_proposals"]) == {"a-first": "cosine", "duplicate": "kmers"}
    assert len(list((greedy / "batches/round-000001").glob("batch-*"))) == 2
    assert not list(greedy.rglob("*.fa"))
    calls = assessment_run[2].read_text().splitlines().count("checkm2")
    assert calls == 6

    assert assess(assessment_run, full, "full", "--reuse-from", str(greedy)) == 0
    report = read(full / "summary.json")
    assert (
        report["compatible_hq_gains"],
        report["imported_constructions"],
        report["checked_constructions"],
    ) == (2, 2, 4)
    assert report["state"] == "COMPLETE" and report["selection"]["optimal"]
    assert assessment_run[2].read_text().splitlines().count("checkm2") == calls + 2
    rows = table(full / "before-after.tsv")
    assert {row["proposal_id"] for row in rows if row["hq_gain"] == "1"} == {"a-second", "b-first"}
    assert any(
        row["status"] == "HQ_FAIL" and not row["candidate_rna_checks"]
        for row in table(full / "assessment.tsv")
    )
    assert (
        main(
            [
                "export",
                "--selection",
                str(full / report["selection"]["path"]),
                "--complete-directory",
                "--output",
                str(tmp_path / "export"),
            ]
        )
        == 0
    )
    assert len(table(tmp_path / "export/bins.tsv")) == 4
    assert assess(assessment_run, greedy, "greedy", "--max-rounds", "2") == 0
    assert read(greedy / "summary.json")["rounds_completed"] == 2
    assert assess(assessment_run, full, "full", "--reuse-from", str(greedy)) == 0
    assert read(full / "summary.json")["imported_constructions"] == 2


def test_tool_failure_resumes_numeric_checkpoint_and_rejects_changes(assessment_run, tmp_path):
    output = tmp_path / "greedy"
    _, config, events, switch = assessment_run
    switch.write_text("fail", encoding="utf-8")
    options = ("--max-rounds", "1")
    assert assess(assessment_run, output, "greedy", *options) == 2
    assert read(output / "summary.json")["state"] == "FAILED"
    first = output / "batches/round-000001/batch-000001"
    assert (first / "checkm2.json").is_file()
    assert not (first / "complete.json").exists()
    before = events.read_text().splitlines().count("checkm2")
    switch.write_text("normal", encoding="utf-8")
    assert assess(assessment_run, output, "greedy", *options) == 0
    assert events.read_text().splitlines().count("checkm2") == before + 1
    assert len(list(first.glob("c000001.rna-*"))) == 2
    (config.parent / "database.dmnd").write_bytes(b"changed")
    assert assess(assessment_run, output, "greedy", *options) == 2
    assert assess(assessment_run, tmp_path / "full", "full", "--reuse-from", str(output)) == 2


def test_full_excludes_discordant_control_and_pins_raw_evidence(assessment_run, tmp_path):
    output = tmp_path / "full"
    assessment_run[3].write_text("discordant", encoding="utf-8")
    assert assess(assessment_run, output, "full") == 0
    summary = read(output / "summary.json")
    assert summary["control_discordant_bins"] == ["b1"]
    assert summary["checked_constructions"] == 4
    assert summary["compatible_hq_gains"] == 1
    raw = next(output.glob("batches/*/checkm2-*/results/quality_report.tsv"))
    raw.write_text("changed", encoding="utf-8")
    assert assess(assessment_run, output, "full") == 2


def test_pause_after_checkm2_resumes_same_round(assessment_run, tmp_path, monkeypatch):
    from bincomplete.assessment_evidence import Evaluator

    output = tmp_path / "greedy"
    original = Evaluator.checkm2

    def check_and_pause(self, inputs, attempt, names):
        result = original(self, inputs, attempt, names)
        if any(name.startswith("c") for name in names):
            import signal

            signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Evaluator, "checkm2", check_and_pause)
        assert assess(assessment_run, output, "greedy", "--max-rounds", "2") == 75
    assert read(output / "summary.json")["state"] == "PAUSED"
    before = assessment_run[2].read_text().splitlines().count("checkm2")
    assert assess(assessment_run, output, "greedy", "--max-rounds", "1") == 0
    assert assessment_run[2].read_text().splitlines().count("checkm2") == before + 1
    assert read(output / "summary.json")["rounds_completed"] == 1


def test_no_hits_and_all_no_proteins_have_reusable_evidence(assessment_run, tmp_path):
    assessment_run[3].write_text("zero", encoding="utf-8")
    output = tmp_path / "assessment"
    assert assess(assessment_run, output, "greedy", "--max-rounds", "1") == 0
    assert read(output / "summary.json")["compatible_hq_gains"] == 1
    baseline = table(output / "baseline.tsv")
    assert {row["checkm2_status"] for row in baseline} == {"SCORED", "NO_PREDICTED_PROTEINS"}
    rna = json.loads(baseline[0]["rna_checks"])
    assert rna["domains"]["arc"]["trna_labels"] == [] and rna["pass"]
    assert assess(assessment_run, tmp_path / "full", "full", "--reuse-from", str(output)) == 0


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("partial", "tRNAscan-SE reported a failed scan"),
        ("mutate", "Assessment input FASTA changed"),
    ],
)
def test_rejects_invalid_tool_evidence(assessment_run, tmp_path, scenario, message):
    assessment_run[3].write_text(scenario, encoding="utf-8")
    output = tmp_path / "assessment"
    assert assess(assessment_run, output, "greedy", "--max-rounds", "1") == 2
    assert message in read(output / "summary.json")["error"]
    if scenario == "mutate":
        assert not list((output / "batches").rglob("checkm2.json"))


def test_environment_identity_and_solver_budget(assessment_run, tmp_path, monkeypatch):
    config = assessment_run[1]
    config.write_text('runtime_env = ["ASSESS_TEST_VARIANT"]\n' + config.read_text(), encoding="utf-8")
    output = tmp_path / "full"
    monkeypatch.setenv("ASSESS_TEST_VARIANT", "original")
    assert assess(assessment_run, output, "full") == 0
    calls = assessment_run[2].read_text()
    original = read(output / "summary.json")["selection"]["path"]
    config.write_text("solver_seconds = 601\n" + config.read_text(), encoding="utf-8")
    assert assess(assessment_run, output, "full") == 0
    assert assessment_run[2].read_text() == calls
    assert read(output / "summary.json")["selection"]["path"] != original
    monkeypatch.setenv("ASSESS_TEST_VARIANT", "changed")
    assert assess(assessment_run, output, "full") == 2


def test_sqlite_sequence_path_with_fragment(tmp_path):
    from conftest import population

    prepared = population(tmp_path / "input#test")
    sequence = next(prepared.sequences([0], ResourceLimits()))
    assert len(sequence) == 2400
    assert next(prepared.sequence_windows([0], ResourceLimits(), window_bp=12, count=1))


@pytest.mark.skipif(os.name != "posix", reason="POSIX child inherits the assessment lock")
def test_resume_reclaims_owned_scratch(assessment_run, tmp_path):
    output = tmp_path / "assessment"
    assessment_run[3].write_text("fail", encoding="utf-8")
    assert assess(assessment_run, output, "greedy", "--max-rounds", "1") == 2
    batch = output / "batches/round-000001/batch-000001"
    owned = batch / "fasta-work"
    owned.mkdir()
    save(owned / ".owner.json", {"plan": fingerprint(read(batch / "plan.json"))})
    (owned / "stale.fa").write_text(">stale\nACGT\n", encoding="utf-8")
    other = batch / "unrelated"
    other.mkdir()
    assessment_run[3].write_text("normal", encoding="utf-8")
    assert assess(assessment_run, output, "greedy", "--max-rounds", "1") == 0
    assert not owned.exists() and other.is_dir()


def test_unverified_empty_protein_attempt_is_rejected(tmp_path):
    from bincomplete.assessment_tools import read_checkm2_attempt

    results = tmp_path / "results"
    proteins = results / "protein_files"
    proteins.mkdir(parents=True)
    (proteins / "one.faa").touch()
    (results / "checkm2.log").write_text(
        "Skipping protein file /tmp/one.faa as it was empty.\n", encoding="utf-8"
    )
    (tmp_path / "stderr.log").write_text(
        "in __calculate_metadata\nself.total_threads / len(faa_files)\nZeroDivisionError: division by zero\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verified report"):
        read_checkm2_attempt(tmp_path, {"one", "two"})


@pytest.mark.parametrize(
    ("completeness", "contamination", "passes"), [(90, 1, False), (91, 5, False), (90.01, 4.99, True)]
)
def test_thresholds(completeness, contamination, passes):
    from bincomplete.assessment_tools import full_hq

    assert (
        full_hq(
            {
                "status": "SCORED",
                "completeness": completeness,
                "contamination": contamination,
                "rna": {"pass": True},
            }
        )
        == passes
    )
