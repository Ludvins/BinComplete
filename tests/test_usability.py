"""Real adapter layouts, complete bin protection and read-only user preflight."""

import json

import numpy as np
import pytest

from bincomplete import prepare_dataset
from bincomplete.artifacts import read, save
from bincomplete.cli import main
from bincomplete.data import mmlong2_inputs
from bincomplete.defaults import expand_experiment
from bincomplete.experiments.config import load_plan
from bincomplete.preflight import preflight


def raw_configuration(tmp_path, dataset, *, methods=None):
    raw = dataset.path.parent
    global_path = tmp_path / "global.json"
    save(
        global_path,
        {
            "device": "cpu",
            "encoder_path": "absent_weights",
            "cache": str(tmp_path / "cache"),
            "resources": {"threads": 1, "memory_gib": 0.5, "cache_gib": 0.01, "chunk_size": 3},
        },
    )
    path = tmp_path / "experiment.json"
    save(
        path,
        {
            "schema_version": 2,
            "global_config": str(global_path),
            "output": "not_created/results",
            "methods": methods or ["kmers"],
            "datasets": {
                "sample": {
                    "assembly": str(raw / "assembly.fa"),
                    "memberships": str(raw / "members.tsv"),
                    "coverage": str(raw / "coverage.tsv"),
                }
            },
        },
    )
    return path


def mmlong_layout(tmp_path):
    root = tmp_path / "sample"
    (root / "results/bins").mkdir(parents=True)
    (root / "results/sample_assembly.fasta").write_text(">a\nACGT\n>free\nAAAA\n")
    (root / "results/bins/one.fa").write_text(">a\nACGT\n")
    mapping = root / "tmp/binning/mapping"
    mapping.mkdir(parents=True)
    content = "contigName\tcontigLen\ttotalAvgDepth\tnp.bam\tnp.bam-var\na\t4\t8\t8\t2\nfree\t4\t7\t7\t3\n"
    for name in ("1-NP.tsv", "1-NP_tr1.tsv", "1-NP_tr2.tsv"):
        (mapping / name).write_text(content)
    return root, mapping, content


def test_real_mmlong_primary_and_trimmed_depth_views(tmp_path):
    root, mapping, content = mmlong_layout(tmp_path)
    assert mmlong2_inputs(root)["coverage"] == mapping / "1-NP.tsv"
    prepared = prepare_dataset(tmp_path / "prepared", input_format="mmlong2", input_root=root)
    np.testing.assert_array_equal(np.load(prepared.path / "coverage.npy"), [[8, 2], [7, 3]])
    (mapping / "1-PB.tsv").write_text(content)
    with pytest.raises(ValueError, match="1-NP.tsv.*1-PB.tsv.*init --coverage"):
        mmlong2_inputs(root)
    assert mmlong2_inputs(root, coverage=mapping / "1-NP.tsv")["coverage"] == mapping / "1-NP.tsv"


def test_mmlong_missing_or_mean_only_depth_is_actionable(tmp_path):
    root, mapping, _ = mmlong_layout(tmp_path)
    (mapping / "1-NP.tsv").write_text("contig_id\tmean\na\t8\nfree\t7\n")
    with pytest.raises(ValueError, match="measured variance.*Retain"):
        mmlong2_inputs(root)


@pytest.mark.parametrize("suffix", [".fa.gz", ".fna.bz2", ".fasta.xz", ".fasta.bgz", ".fsa"])
def test_mixed_bins_never_silently_omit_original_members(tmp_path, suffix):
    bins = tmp_path / "bins"
    bins.mkdir()
    (bins / "one.fa").write_text(">a\nACGT\n")
    (bins / ("two" + suffix)).write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="Unsupported bin FASTAs.*two.*Decompress"):
        prepare_dataset(
            tmp_path / "prepared",
            bins=bins,
            assembly=tmp_path / "assembly.fa",
            coverage=tmp_path / "coverage.tsv",
        )
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "prepared.prepare.lock").exists()


def test_composition_only_compact_needs_no_weights_including_file_settings(tmp_path, dataset):
    config = raw_configuration(tmp_path, dataset)
    value = read(config)
    save(tmp_path / "settings.json", {"classifier": {"params": {"max_iter": 3}}})
    value["methods"] = [{"method": "kmers", "settings": "settings.json"}]
    save(config, value)
    plan = load_plan(config)
    assert "encoder_path" not in plan.document["datasets"]["sample"]
    value["methods"] = [{"method": "cosine"}]
    expanded, _, _ = expand_experiment(value, config.parent)
    assert "encoder_path" in expanded["datasets"]["sample"]


def test_generic_init_flags_and_same_dataset_preflight_are_read_only(tmp_path, dataset, capsys):
    config = raw_configuration(tmp_path, dataset)
    spec = read(config)["datasets"]["sample"]
    output = tmp_path / "initialized"
    assert (
        main(
            [
                "init",
                "--assembly",
                spec["assembly"],
                "--memberships",
                spec["memberships"],
                "--coverage",
                spec["coverage"],
                "--global-config",
                str(tmp_path / "global.json"),
                "--methods",
                "kmers",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    files_before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    assert main(["preflight", "--config", str(output / "experiment.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] and report["datasets"]["sample"]["candidate_contigs"] == 6
    assert report["datasets"]["sample"]["protected_contigs"] == 8
    assert report["datasets"]["sample"]["training_groups"] == 4
    assert "transductive" in " ".join(report["warnings"])
    assert files_before == {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    assert not (output / "results").exists()


def test_init_bins_and_mmlong_coverage_overrides(tmp_path, capsys):
    root, mapping, _ = mmlong_layout(tmp_path)
    output = tmp_path / "initialized"
    assert (
        main(
            [
                "init",
                "--input-root",
                str(root),
                "--coverage",
                str(mapping / "1-NP.tsv"),
                "--coverage-profile",
                "np.bam",
                "--donor-ids",
                str(tmp_path / "donors.txt"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    spec = read(output / "experiment.json")["datasets"]["sample"]
    assert spec["coverage"] == str(mapping / "1-NP.tsv")
    assert spec["coverage_profile"] == "np.bam" and "donor_ids" in spec
    assert (
        main(
            [
                "init",
                "--assembly",
                str(root / "results/sample_assembly.fasta"),
                "--bins",
                str(root / "results/bins"),
                "--coverage",
                str(mapping / "1-NP.tsv"),
                "--output",
                str(tmp_path / "generic"),
            ]
        )
        == 0
    )
    spec = read(tmp_path / "generic/experiment.json")["datasets"]["sample"]
    assert "bins" in spec and "memberships" not in spec
    assert main(["init", "--assembly", "assembly.fa", "--output", str(tmp_path / "incomplete")]) == 2
    assert not (tmp_path / "incomplete").exists()
    assert "Generic init needs" in capsys.readouterr().err


def test_preflight_reports_variance_weights_groups_and_unknown_recipients(tmp_path, dataset):
    config = raw_configuration(tmp_path, dataset)
    raw = dataset.path.parent
    lines = (raw / "coverage.tsv").read_text().splitlines()
    (raw / "coverage.tsv").write_text("\n".join("\t".join(line.split("\t")[:2]) for line in lines) + "\n")
    result = preflight(config)
    assert not result["ready"] and "variance" in " ".join(result["errors"])
    value = read(config)
    value["methods"] = ["cosine"]
    save(config, value)
    assert "absent_weights" in " ".join(preflight(config)["errors"])
    value["datasets"]["sample"].update(
        embeddings=str(raw / "embedding.npy"),
        embedding_ids=str(raw / "embedding_ids.txt"),
        embedding_space=str(raw / "space.json"),
    )
    value["recipients"] = ["missing"]
    save(config, value)
    assert "Unknown recipient" in " ".join(preflight(config)["errors"])
    value.pop("recipients")
    value["methods"] = ["single"]
    groups = tmp_path / "groups.tsv"
    groups.write_text("bin_id\tgroup_id\n" + "".join(f"b{i}\tsame\n" for i in range(4)))
    value["datasets"]["sample"]["groups"] = str(groups)
    save(config, value)
    assert "four training groups" in " ".join(preflight(config)["errors"])
    assert not (tmp_path / "not_created").exists()


def test_preflight_checks_prepared_inputs_and_supplied_vector_norms(tmp_path, dataset):
    config = raw_configuration(tmp_path, dataset, methods=["cosine"])
    value = read(config)
    value["datasets"]["sample"] = {"prepared": str(dataset.path)}
    save(config, value)
    assert preflight(config)["ready"]
    raw = dataset.path.parent
    value["datasets"]["sample"] = {
        "assembly": str(raw / "assembly.fa"),
        "memberships": str(raw / "members.tsv"),
        "coverage": str(raw / "coverage.tsv"),
        "embeddings": str(raw / "embedding.npy"),
        "embedding_ids": str(raw / "embedding_ids.txt"),
        "embedding_space": str(raw / "space.json"),
    }
    save(config, value)
    vectors = np.load(raw / "embedding.npy")
    vectors[0] *= 2
    np.save(raw / "embedding.npy", vectors)
    assert "L2 normalized" in " ".join(preflight(config)["errors"])


def test_cli_input_errors_are_concise_with_optional_debug(tmp_path, capsys):
    args = ["experiment", "--config", str(tmp_path / "missing.json")]
    assert main(args) == 2
    message = capsys.readouterr().err
    assert "bincomplete: error:" in message and "Traceback" not in message
    with pytest.raises(FileNotFoundError):
        main(["--debug", *args])
