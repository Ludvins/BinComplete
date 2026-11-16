"""Exercise the inspector export using completed synthetic BinComplete artifacts."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from bincomplete import ProposalArtifact, candidate_table, export, fit, propose, reconcile
from bincomplete.artifacts import read, save
from bincomplete.cli import main
from bincomplete.data import fasta_records
from bincomplete.report import write_report


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def assessment(path, header, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(records)


def source(path, kind="tsv", **extra):
    return {
        "format": kind,
        "report": path.name,
        "tool": "Synthetic QC",
        "version": "1",
        "settings": "Fixture",
    } | extra


@pytest.fixture
def saved_run(tmp_path, dataset, limits):
    cache = tmp_path / "cache"
    model = fit(dataset, tmp_path / "model", method="cosine", limits=limits, cache=cache)
    first = propose(
        dataset=dataset,
        model=model,
        output=tmp_path / "first",
        recipients=["b0", "b1", "b2"],
        limits=limits,
        cache=cache,
    )
    other_model = fit(dataset, tmp_path / "other-model", method="kmers", limits=limits, cache=cache)
    second = propose(
        dataset=dataset,
        model=other_model,
        output=tmp_path / "second",
        recipients=["b0"],
        limits=limits,
        cache=cache,
    )
    candidates = candidate_table([first, second], tmp_path / "candidates.tsv", limits=limits)
    inventory = rows(candidates)
    other = {row["sequence_set_id"]: row for row in inventory if row["method"] == "kmers"}
    gain = next(
        row
        for row in inventory
        if row["recipient"] == "b0" and row["method"] == "cosine" and row["sequence_set_id"] in other
    )
    duplicate = other[gain["sequence_set_id"]]
    members = set(json.loads(gain["members_json"]))
    loss = next(
        row
        for row in inventory
        if row["recipient"] == "b1" and not members.intersection(json.loads(row["members_json"]))
    )
    ids = [gain["proposal_id"], loss["proposal_id"]]
    (tmp_path / "approved.txt").write_text("\n".join(ids), encoding="utf-8")
    selection = reconcile([second, first], ids, tmp_path / "selection", time_limit=17, limits=limits)
    assert len(selection.manifest["selected"]) == 2
    export(selection, tmp_path / "final", limits=limits)
    original = reconcile([first], [], tmp_path / "original", limits=limits)
    export(original, tmp_path / "baseline", complete_directory=True, limits=limits)
    export(first, tmp_path / "batch", proposal_ids=ids, limits=limits)
    config = {
        "schema_version": 1,
        "candidates": "candidates.tsv",
        "proposals": ["first", "second"],
        "selection": "selection",
        "selection_proposals": ["second", "first"],
        "selection_time_limit": 17,
        "approved_ids": "approved.txt",
        "final_export": "final",
        "hq_policy": {"mode": "external", "description": "Synthetic HQ policy"},
    }
    return config, gain, loss, duplicate


def test_report_exports_selected_contig_evidence(tmp_path, saved_run, dataset, limits):
    config, gain, _, _ = saved_run
    original = set(dataset.manifest["bins"]["b0"])
    added = set(json.loads(gain["members_json"])) - original
    recipient = min(added)
    path = tmp_path / "rna-features.tsv"
    header = ["proposal_id", "contig_id", "feature", "start_bp", "end_bp", "strand"]
    feature = (gain["proposal_id"], recipient, "tRNA-Lys", 1, 80, "-")
    assessment(path, header, [feature])
    config["contig_evidence"] = {"rna_features": source(path)}
    config_path = tmp_path / "config.json"
    save(config_path, config)
    report = read(write_report(config_path, tmp_path / "evidence.json"))
    evidence = report["bins"][0]["contig_evidence"]
    sequences = {
        name: sequence for name, sequence, _, _ in fasta_records(dataset.manifest["assembly"], limits)
    }
    assert set(row["id"] for row in evidence["added"]) == added
    for row in evidence["added"]:
        sequence = sequences[row["id"]]
        assert row["sequence_sha256"] == hashlib.sha256(sequence).hexdigest()
        assert row["gc_fraction"] == (sequence.count(b"G") + sequence.count(b"C")) / len(sequence)
    assert evidence["original"] == [
        {
            "length_bp": len(sequences[name]),
            "mean_depth": 40 + int(name[1:]) % 3,
            "gc_fraction": (sequences[name].count(b"G") + sequences[name].count(b"C")) / len(sequences[name]),
        }
        for name in sorted(original)
    ]
    assert evidence["rna_features"] == [
        {"contig_id": recipient, "feature": "tRNA-Lys", "start_bp": 1, "end_bp": 80, "strand": "-"}
    ]
    assert (
        report["provenance"]["contig_rna_source"]["report_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert "contig_evidence" not in report["bins"][2]

    for name, row in [
        ("original", (gain["proposal_id"], min(original), "tRNA-Lys", 1, 80, "-")),
        ("past-end", (gain["proposal_id"], recipient, "tRNA-Lys", 1, len(sequences[recipient]) + 1, "-")),
    ]:
        assessment(path, header, [row])
        with pytest.raises(ValueError, match="Invalid or repeated RNA feature"):
            write_report(config_path, tmp_path / f"{name}.json")
    assessment(path, header, [feature])
    assembly = Path(dataset.manifest["assembly"])
    data = assembly.read_bytes()
    offset = data.index(b"\n") + 1
    alternate = b"C" if data[offset : offset + 1] == b"A" else b"A"
    assembly.write_bytes(data[:offset] + alternate + data[offset + 1 :])
    with pytest.raises(ValueError, match="Assembly sequence changed after preparation"):
        write_report(config_path, tmp_path / "changed-sequence.json")


def test_report_includes_benchmark_results_and_saved_training(tmp_path, saved_run):
    config, _, _, _ = saved_run
    identity = read(tmp_path / "second" / "complete.json")["identity"]
    config_path = tmp_path / "config.json"
    metrics_path = tmp_path / "metrics.tsv"
    header = [
        "artifact_id",
        "benchmark_id",
        "protocol",
        "precision_at_5",
        "recall_at_5",
        "run_id",
        "evaluated_groups",
    ]
    assessment(metrics_path, header, [(identity, "holdout-v1", "group-disjoint test", 0, 0.35, "run-1", 12)])
    config["benchmark_results"] = metrics_path.name
    save(config_path, config)
    report = read(write_report(config_path, tmp_path / "report.json"))
    rows_by_artifact = report["comparison"]["methods"]
    row = rows_by_artifact[identity]
    assert row["training"]["learned"]
    assert row["training"]["positives"] > 0
    assert row["training"]["rows"] >= row["training"]["positives"]
    assert row["search"]["max_checks"] > 0
    assert row["benchmark"] == {
        "benchmark_id": "holdout-v1",
        "protocol": "group-disjoint test",
        "precision_at_5": 0,
        "recall_at_5": 0.35,
        "run_id": "run-1",
        "evaluated_groups": 12,
        "report_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }
    assert "attention_validation" not in row
    assert all("benchmark" not in item for key, item in rows_by_artifact.items() if key != identity)
    assessment(metrics_path, header, [(identity, "holdout-v1", "group-disjoint test", 48, "", "", "")])
    with pytest.raises(ValueError, match="fraction from 0 to 1"):
        write_report(config_path, tmp_path / "invalid.json")


def test_report_imports_checkm2_and_verdicts_and_preserves_proposal_states(tmp_path, saved_run):
    config, gain, loss, duplicate = saved_run
    before, after, verdicts = (tmp_path / name for name in ("before.tsv", "after.tsv", "verdicts.tsv"))
    baseline = {row["recipient"]: Path(row["fasta"]).stem for row in rows(tmp_path / "baseline/bins.tsv")}
    exported = {row["proposal_id"]: Path(row["fasta"]).name for row in rows(tmp_path / "batch/bins.tsv")}
    assessment(
        before,
        ["Name", "Completeness_Specific", "Contamination"],
        [(baseline["b0"], 89, 2), (baseline["b1"], 96, 3)],
    )
    assessment(
        after,
        [
            "Name",
            "Completeness_General",
            "Completeness_Specific",
            "Contamination",
            "Contig_N50",
            "GC_Content",
            "Coding_Density",
            "Average_Gene_Length",
        ],
        [
            (exported[gain["proposal_id"]], 92, 93, 3, 2400, 0.52, 0.891, 287.3),
            (exported[loss["proposal_id"]], 96, 97, 6, 2400, 0.48, 0.812, 291.7),
        ],
    )
    assessment(
        verdicts,
        ["kind", "id", "hq_status", "rrna_5s", "rrna_16s", "rrna_23s", "trna_types"],
        [
            ("original", "b0", "fail", "detected", "not_detected", "detected", 17),
            ("original", "b1", "pass", "detected", "detected", "detected", 20),
            ("proposal", duplicate["proposal_id"], "pass", "detected", "detected", "detected", 19),
            ("proposal", loss["proposal_id"], "fail", "detected", "not_assessed", "detected", 20),
        ],
    )
    config["assessments"] = [
        source(before, "checkm2", export="baseline"),
        source(after, "checkm2", export="batch", completeness_column="Completeness_Specific"),
        source(verdicts),
    ]
    config_path, target = (tmp_path / name for name in ("config.json", "bincomplete-report.json"))
    save(config_path, config)
    assert main(["report", "--config", str(config_path), "--output", str(target)]) == 0
    report = read(target)
    assert report["schema_version"] == 2
    assert report["rna_summary"] == {"comparable_pairs": 1, "incomplete_pairs": 2, "gains": 1, "losses": 0}
    assert report["summary"] == {
        "original_bins": 4,
        "recipients": 3,
        "modified_bins": 2,
        "comparable_pairs": 2,
        "unassessed_pairs": 1,
        "original_hq": 1,
        "final_hq": 1,
        "hq_gains": 1,
        "hq_losses": 1,
        "net_hq_change": 0,
    }
    bin0, bin1, bin2, bin3 = report["bins"]
    assert bin0["delta"] == {"completeness": 4, "contamination": 1}
    assert bin1["hq_transition"] == "loss" and bin2["hq_transition"] == "unknown" and not bin3["recipient"]
    proposals = {row["id"]: row for row in bin0["proposals"]}
    selected, equivalent = proposals[gain["proposal_id"]], proposals[duplicate["proposal_id"]]
    assert selected["quality"] == equivalent["quality"]
    assert selected["quality"]["rna"]["trna_types"] == 19
    assert selected["quality"]["checkm2"]["contig_n50"] == 2400
    assert selected["quality"]["checkm2"]["completeness_general"] == 92
    assert selected["quality"]["checkm2"]["gc_content"] == 0.52
    assert selected["quality"]["checkm2"]["coding_density"] == 0.891
    assert selected["quality"]["checkm2"]["average_gene_length"] == 287.3
    sources = report["provenance"]["assessment_sources"]
    assert [item["scope"] for item in sources] == ["original", "proposal", "original, proposal"]
    assert sources[0]["completeness_column"] == sources[1]["completeness_column"] == "Completeness_Specific"
    assert bin0["original"]["rna"]["rrna_16s"] is False and bin0["final"]["rna"]["status"] == "pass"
    assert bin1["final"]["rna"]["rrna_16s"] is None and bin1["final"]["rna"]["status"] == "unknown"
    assert sum(donor["length_bp"] for donor in selected["donors"]) == selected["added_bp"]
    assert min(donor["score"] for donor in selected["donors"]) == selected["min_added_score"]
    assert report["comparison"]["pairwise"][0]["identical_constructions"] > 0
    assert selected["assessed"] and selected["approved"] and selected["selected"] and selected["exported"]
    assert (
        equivalent["assessed"]
        and not equivalent["approved"]
        and not equivalent["selected"]
        and not equivalent["exported"]
    )
    assert str(tmp_path) not in target.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        write_report(config_path, target)
    with pytest.raises(ValueError, match="outside completed"):
        write_report(config_path, tmp_path / "selection/report.json")

    config["assessments"][1]["completeness_column"] = "Completeness_General"
    save(config_path, config)
    mixed = read(write_report(config_path, tmp_path / "mixed.json"))
    assert mixed["bins"][0]["assessment_mismatches"] == ["completeness", "contamination"]
    assert mixed["bins"][0]["delta"] == {"completeness": None, "contamination": None}
    assert mixed["summary"]["comparable_pairs"] == 0
    assert mixed["rna_summary"] == report["rna_summary"]

    config["assessments"][1]["completeness_column"] = "Completeness_Specific"
    config["hq_policy"].update(standard="mimag")
    conflicting = tmp_path / "contradictory-verdicts.tsv"
    conflicting.write_text(
        verdicts.read_text(encoding="utf-8").replace(
            "pass\tdetected\tdetected\tdetected\t19", "pass\tdetected\tnot_detected\t\t17"
        ),
        encoding="utf-8",
    )
    config["assessments"][2]["report"] = conflicting.name
    save(config_path, config)
    conflict = read(write_report(config_path, tmp_path / "conflict.json"))
    final = conflict["bins"][0]["final"]
    assert final["reported_hq_status"] == "pass" and final["hq_status"] == "unknown"
    assert final["hq_conflicts"] == ["16S rRNA", "tRNA amino acid types"]
    assert final["rna"]["status"] == "unknown"
    assert conflict["bins"][0]["hq_transition"] == "unknown" and conflict["summary"]["hq_gains"] == 0


def test_thresholds_and_empty_selection(tmp_path, saved_run, limits):
    config, *_ = saved_run
    reconcile(
        [ProposalArtifact(tmp_path / name) for name in config["proposals"]],
        [],
        tmp_path / "empty_selection",
        limits=limits,
    )
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    for key in ("final_export", "selection_proposals", "selection_time_limit"):
        del config[key]
    config.update(
        selection="empty_selection",
        approved_ids="empty.txt",
        hq_policy={
            "mode": "thresholds",
            "completeness": {"operator": ">=", "value": 90},
            "contamination": {"operator": "<", "value": 5},
        },
    )
    tsv = tmp_path / "quality.tsv"
    assessment(
        tsv,
        ["kind", "id", "completeness", "contamination"],
        [
            ("original", "b0", 90, 4.9),
            ("original", "b1", 99, 5),
            ("original", "b2", 99, ""),
        ],
    )
    config["assessments"] = [source(tsv)]
    path = tmp_path / "config.json"
    save(path, config)
    report = read(write_report(path, tmp_path / "report.json"))
    assert [row["hq_transition"] for row in report["bins"]] == ["retained", "non_hq", "unknown", "unknown"]
    assert report["summary"]["modified_bins"] == report["summary"]["net_hq_change"] == 0
    assert all(row["exported"] is None for row in report["bins"][0]["proposals"])
    del config["assessments"]
    save(path, config)
    unavailable = read(write_report(path, tmp_path / "unavailable.json"))["summary"]
    assert unavailable["comparable_pairs"] == 0 and unavailable["hq_gains"] is None


def test_rejects_mismatched_approvals_and_inventory(tmp_path, saved_run):
    config, gain, _, _ = saved_run
    config_path = tmp_path / "config.json"
    save(config_path, config)
    (tmp_path / "approved.txt").write_text(gain["proposal_id"], encoding="utf-8")
    with pytest.raises(ValueError, match="Selection receipt"):
        write_report(config_path, tmp_path / "report.json")
    with (tmp_path / "candidates.tsv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Candidate inventory receipt"):
        write_report(config_path, tmp_path / "report.json")


def test_rejects_conflicts_and_duplicate_checkm2_aliases(tmp_path, saved_run):
    config, gain, _, duplicate = saved_run
    tsv = tmp_path / "quality.tsv"
    assessment(
        tsv,
        ["kind", "id", "completeness", "contamination"],
        [
            ("proposal", gain["proposal_id"], 93, 2),
            ("proposal", duplicate["proposal_id"], 91, 2),
        ],
    )
    config["assessments"] = [source(tsv)]
    path = tmp_path / "config.json"
    save(path, config)
    with pytest.raises(ValueError, match="Conflicting completeness"):
        write_report(path, tmp_path / "report.json")
    fasta = Path(rows(tmp_path / "batch/bins.tsv")[0]["fasta"])
    assessment(tsv, ["Name", "Completeness", "Contamination"], [(fasta.name, 92, 2), (fasta.stem, 92, 2)])
    config["assessments"] = [source(tsv, "checkm2", export="batch")]
    save(path, config)
    with pytest.raises(ValueError, match="Repeated CheckM2 row"):
        write_report(path, tmp_path / "report.json")
