"""Convert an explicitly described upstream table to BinComplete memberships.

Column order and unassigned labels are never guessed. Source files are not
modified. Numeric cluster 0 is preserved with --bin-prefix bin_, rather than
being confused with the membership TSV's reserved unassigned label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def convert(
    source: Path,
    output: Path,
    *,
    contig_column: str,
    bin_column: str,
    delimiter: str = "\t",
    headerless: bool = False,
    bin_prefix: str = "",
    unassigned: tuple[str, ...] = (),
) -> dict:
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise ValueError("Output exists; preserve it and choose another filename")
    labels: dict[str, str] = {}
    omitted = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream, delimiter=delimiter)
        if headerless:
            columns = [int(contig_column) - 1, int(bin_column) - 1]
            if min(columns) < 0:
                raise ValueError("Headerless columns are numbered from 1")
        else:
            header = next(rows)
            columns = [header.index(contig_column), header.index(bin_column)]
        if columns[0] == columns[1]:
            raise ValueError("Contig and bin columns must differ")
        for number, row in enumerate(rows, start=1 if headerless else 2):
            if not row:
                continue
            if len(row) <= max(columns):
                raise ValueError(f"Row {number} has too few columns")
            contig, label = [row[i].strip() for i in columns]
            if not contig:
                raise ValueError(f"Missing contig ID on row {number}")
            if label in unassigned:
                omitted += 1
                continue
            value = bin_prefix + label
            if not label or value in {"", "0", "NA", "unassigned", "unbinned", "-"}:
                raise ValueError(
                    f"Reserved/empty bin label on row {number}; set --bin-prefix or --unassigned-label"
                )
            if contig in labels and labels[contig] != value:
                raise ValueError(f"Contig {contig!r} belongs to multiple bins")
            labels[contig] = value
    if not labels:
        raise ValueError("No assigned contigs were retained")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["contig_id", "bin_id"])
        writer.writerows(labels.items())
    receipt = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "columns": {"contig": contig_column, "bin": bin_column},
        "headerless": headerless,
        "delimiter": delimiter,
        "bin_prefix": bin_prefix,
        "explicit_unassigned_labels": list(unassigned),
        "assigned_contigs": len(labels),
        "bins": len(set(labels.values())),
        "unassigned_rows_omitted": omitted,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contig-column", required=True, help="Header name, or 1-based number with --headerless"
    )
    parser.add_argument("--bin-column", required=True)
    parser.add_argument("--headerless", action="store_true")
    parser.add_argument("--delimiter", choices=["tab", "comma"], default="tab")
    parser.add_argument("--bin-prefix", default="")
    parser.add_argument("--unassigned-label", action="append", default=[])
    args = parser.parse_args()
    try:
        result = convert(
            args.input,
            args.output,
            contig_column=args.contig_column,
            bin_column=args.bin_column,
            delimiter="\t" if args.delimiter == "tab" else ",",
            headerless=args.headerless,
            bin_prefix=args.bin_prefix,
            unassigned=tuple(args.unassigned_label),
        )
    except (ValueError, OSError, StopIteration) as exc:
        parser.error(str(exc) or "Input table is empty")
    print(json.dumps(result, indent=2))
