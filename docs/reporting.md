# Export a report for BinCompleteInspector

[BinCompleteInspector](inspector.md) opens one JSON report from a completed
selection. Generate it after evaluating proposals and running `reconcile`. This
step reads saved outputs. It does not run CheckM2, rescore proposals, change the
selection, or create FASTAs.

For a run from [`bincomplete assess`](evaluation.md#run-checkm2-and-rna-checks-with-bincomplete),
use `baseline.tsv` and `assessment.tsv` as sources when preparing the common
assessment TSV described below. Leave unscored quality verdicts unknown.
The common TSV's `trna_types` field requires a count of distinct amino acid
types. Leave it blank if your assessment policy supplies only a reported-label
count. In `hq_policy`, describe the criterion used for the supplied verdicts.

A greedy assessment uses a fixed selection.
To prepare a report for that selection, put its selected proposal IDs in the
approval file and run `bincomplete reconcile` in a new directory with the
same proposal artifacts and protected bins. Use that reconciliation in the report.
For a full assessment, the standard reconciliation receipt can be used with
its original solver time limit, protected bins, and proposal artifact order.
Use one approved ID per `HQ_PASS` construction, choosing the first ID
lexically from its `source_proposals` object, as the assessment does.

## Generate the report

| What you want to review | Saved inputs to include |
| --- | --- |
| Proposals, rankings and final selections | Candidate inventory and its receipt, proposal artifacts, exact approval list and reconciliation artifact. Include the final export receipt to show export status. |
| CheckM2 before and after, including evaluated alternatives | Assessment outputs for originals and evaluated proposals with their export maps. Include completeness model, assembly statistics and other fields when present in the saved output. |
| HQ and RNA transitions | The run's declared HQ policy and any externally supplied verdicts, plus original and proposal rRNA calls and tRNA amino acid type counts. |
| Added contig evidence | Enable `contig_evidence` and keep the prepared assembly accessible. Supply saved per-contig RNA features for the annotation panel. |
| Sequence viewer | Open the matching local uncompressed FASTA in the browser after loading the report. |

Save the following example as `report.json` and adjust its paths. Input paths
are relative to this configuration file. The CLI output path is relative to the
current directory and must be new and outside completed artifact directories.

```json
{
  "schema_version": 1,
  "candidates": "review/hgb_candidates.tsv",
  "proposals": ["results/methods/hgb_full_pool/propose/0001"],
  "approved_ids": "review/approved.txt",
  "selection": "review/selected",
  "final_export": "review/refined_bins",
  "hq_policy": {
    "mode": "thresholds",
    "completeness": {"operator": ">", "value": 90},
    "contamination": {"operator": "<", "value": 5}
  },
  "assessments": [
    {
      "format": "checkm2",
      "report": "review/original_qc/quality_report.tsv",
      "export": "review/original_controls",
      "tool": "CheckM2",
      "version": "YOUR_VERSION",
      "settings": "YOUR_DATABASE_AND_PREDICTION_SETTINGS"
    },
    {
      "format": "checkm2",
      "report": "review/candidate_qc/quality_report.tsv",
      "export": "review/candidates_hgb",
      "tool": "CheckM2",
      "version": "YOUR_VERSION",
      "settings": "YOUR_DATABASE_AND_PREDICTION_SETTINGS"
    }
  ]
}
```

```shell
bincomplete report --config report.json --output bincomplete-report.json
```

- Use a version 2 candidate inventory with its adjacent `candidates.tsv.json`
  receipt. List the same proposal artifact directories, in the order passed to
  `candidates`. For a standard experiment, use the order in which the methods
  ran. The single-method example assumes an inventory generated for that method.
  If `reconcile` used a different order or subset, set `selection_proposals`
  to the ordered list used there. If it used a nondefault `--time-limit`, set
  `selection_time_limit` to that value. Supply the exact approval list used for
  reconciliation, including an empty file for a zero-approval selection.
- Omit `final_export` if no final export receipt is available. The report still
  shows the selected constructions and displays their export status as unknown.
- List each CheckM2 batch with the completed [export and FASTA map](evaluation.md)
  from which it was assessed. The importer matches each `Name` against the
  mapped FASTA filename or its stem. If multiple completeness columns are
  present, declare `completeness_column` as `Completeness`,
  `Completeness_General`, or `Completeness_Specific`, as appropriate. Use the
  common TSV below for original FASTAs evaluated without a BinComplete export
  map. References to original FASTAs outside an export are not assessed through
  its map. Use the same evaluator settings for paired baseline and proposal results.
  The report records each source's scope, effective completeness column, and
  export receipt. The inspector displays GC content and coding density as
  percentages and average gene length in amino acids, when CheckM2 supplies them.

## Import a common assessment TSV

Add an assessment entry with `format` set to `tsv`, the `report` path, and
`tool`, `version`, and `settings` describing the source. The TSV has required
columns `kind` and `id`. Add `completeness`, `contamination`, `hq_status`, or
any of the RNA columns shown below. Use percentages on the 0 to 100 scale for completeness and
nonnegative percentages for contamination. Leave missing measurements blank.
Use `original` with an original bin ID and `proposal` with a `proposal_id`.

```tsv
kind	id	completeness	contamination	hq_status	rrna_5s	rrna_16s	rrna_23s	trna_types
original	sample-bin	86.4	2.1	fail	detected	not_detected	detected	17
proposal	YOUR_PROPOSAL_ID	93.2	2.8	pass	detected	detected	detected	19
```

Use `hq_status` values `pass`, `fail` or `unknown`. To use those verdicts,
set `hq_policy` to an object with `mode` set to `external` and a `description`
of the criterion used. This is the mode for MIMAG verdicts based on completeness,
contamination and the required RNA evidence. Set `standard` to `mimag` in that
policy to label the report's HQ counts accordingly. The TSV can also supply verdicts
alongside CheckM2 measurements. In threshold mode, the CLI derives HQ from the
two stated comparisons. Blanks remain unknown. Conflicting measurements or
verdicts for the same original or exact proposal construction stop the export.

For each rRNA column, use `detected`, `not_detected`, or `not_assessed`. A blank
is equivalent to `not_assessed`. `trna_types` counts distinct represented amino
acid types from 0 to 20. The RNA verdict compares complete observations against
the three rRNA markers and the 18-type cutoff. Incomplete RNA observations have
an unknown RNA verdict. RNA-only TSVs can supplement existing CheckM2 or verdict
tables. When supplied measurements contradict an external MIMAG HQ pass, the
report retains the supplied pass, lists the conflicting fields, and treats the
HQ transition as unresolved until the evidence is reconciled.

## Include added contig evidence

Add `"contig_evidence": {}` to the report configuration to include a GC and
mean-depth plot on each modified bin page. The exporter uses the prepared
assembly and saved coverage. GC is calculated over A, C, G, and T bases.
The original-bin points include length, GC and depth. The added-contig
records also include the sequence digest used to verify a local FASTA.

To show saved RNA features, add the following fields to the report
configuration and supply the source information.

```json
{
  "contig_evidence": {
    "rna_features": {
      "report": "review/rna-features.tsv",
      "tool": "YOUR_RNA_ANNOTATOR",
      "version": "YOUR_VERSION",
      "settings": "YOUR_ANNOTATION_SETTINGS"
    }
  }
}
```

```tsv
proposal_id	contig_id	feature	start_bp	end_bp	strand
YOUR_SELECTED_PROPOSAL_ID	YOUR_ADDED_CONTIG_ID	16S rRNA	2401	3900	+
```

Use the selected proposal ID from the candidate inventory and an added contig
ID from that construction. Coordinates are one-based and inclusive, with
`+` or `-` strand. The panel displays the supplied calls alongside the
existing bin-level checks. To view or copy sequence, open **View sequence** and
select a matching uncompressed FASTA, such as the final bin FASTA. Feature
sequences are shown in the annotated strand orientation. The browser checks
the contig ID, length, and sequence digest.

## Interpret the inspector

The summary compares each recipient with its final selected construction or
with its original if unchanged. The two HQ totals use the same set of comparable
pairs. Missing HQ evidence and original bins outside the recipient set are
visible in the report. For each paired measurement, use matching source format,
tool, version, settings, and, for CheckM2, the same completeness column. When
these differ, the inspector shows the observations and withholds affected changes.
Regenerate older reports before interpreting paired changes from different CheckM2
configurations. Read the [counting rules](validation.md#counting-quality-changes)
for how gains and losses are defined. The Overview lists all HQ gains and losses
with paired measurements, signed changes in percentage points where available,
before-and-after RNA checks, and links to the selected proposals by method.
Its plot includes HQ transitions with paired completeness and contamination
measurements. The CheckM2 and RNA panels summarize modified recipient bins.
The CheckM2 median uses the comparable pairs for each metric, and the RNA panel
counts missing calls and differing settings as incomplete for the affected
criterion. The method panel counts saved approved and selected proposals by
method, pooling artifacts with the same method name.

In Bins, use **Final quality → HQ and RNA pass** to require both verdicts for
the final bin. Each bin page shows paired CheckM2 measurements, RNA evidence,
and the selected proposal. Expand **CheckM2 details** and **Assembly metrics**
for additional recorded fields. Open **Proposal ranking and added contigs**
to plot the minimum and mean membership scores within one method artifact and
inspect the lengths and mean depths of the selected contigs when available.
The main proposal table shows original and evaluated completeness and
contamination side by side when measurements are available. The selected
proposal stays at the top across sorting and pagination. Scores and ranks
describe proposal membership and priority within a method artifact and
recipient. Assessment, approval, selection, and final export are shown
separately. The final export column
refers to the reconciled output, even if other proposals were exported earlier
for assessment.

The generated JSON includes bin and proposal identifiers, observed CheckM2
fields, optional RNA evidence, donor-level lengths, coverage and ranking scores,
method comparisons, and declared evaluator metadata. Older version 1 reports
open with the fields they contain. Sequences, full memberships, model weights,
and artifact paths are omitted. Use descriptive labels without filesystem
paths or private notes in the evaluator fields. Opening a file in
BinCompleteInspector keeps it in the browser.
