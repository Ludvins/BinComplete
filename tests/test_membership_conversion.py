"""Exercise explicit upstream column conversion without guessing cluster IDs."""

import importlib.util
from pathlib import Path

import pytest


def converter():
    spec = importlib.util.spec_from_file_location(
        "convert_memberships", Path(__file__).resolve().parents[1] / "tools/convert_memberships.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def test_numeric_cluster_zero_and_csv_headers(tmp_path):
    source = tmp_path / "vamb.tsv"
    source.write_text("0\tc1\n1\tc2\n1\tc3\n")
    output = tmp_path / "members.tsv"
    convert = converter()
    proof = convert(source, output, contig_column="2", bin_column="1", headerless=True, bin_prefix="bin_")
    assert proof["bins"] == 2
    assert output.read_text().splitlines() == ["contig_id\tbin_id", "c1\tbin_0", "c2\tbin_1", "c3\tbin_1"]
    csv = tmp_path / "upstream.csv"
    csv.write_text("sequence,cluster\nc1,0\nc2,unassigned\n")
    proof = convert(
        csv,
        tmp_path / "converted.tsv",
        contig_column="sequence",
        bin_column="cluster",
        delimiter=",",
        bin_prefix="bin_",
        unassigned=("unassigned",),
    )
    assert proof["assigned_contigs"] == 1 and proof["unassigned_rows_omitted"] == 1
    assert csv.read_text() == "sequence,cluster\nc1,0\nc2,unassigned\n"
    with pytest.raises(ValueError, match="Output exists"):
        convert(source, output, contig_column="2", bin_column="1", headerless=True)


def test_reject_ambiguous_assignment_and_unprefixed_reserved_label(tmp_path):
    source = tmp_path / "clusters.tsv"
    source.write_text("0\tc1\n")
    convert = converter()
    with pytest.raises(ValueError, match="Reserved"):
        convert(source, tmp_path / "a.tsv", contig_column="2", bin_column="1", headerless=True)
    source.write_text("1\tc1\n2\tc1\n")
    with pytest.raises(ValueError, match="multiple bins"):
        convert(source, tmp_path / "b.tsv", contig_column="2", bin_column="1", headerless=True)
    assert not (tmp_path / "b.tsv").exists()
