# Assess proposals and export compatible bins

An experiment fits membership models and generates alternative additions. Its
`evaluation` role means applying the models to that dataset and comparing full
membership constructions. `report.md` counts proposals, **not HQ gains**. On the
same training/evaluation dataset, learned models disclose transductive overlap.
This does not establish independent generalization.

Terms such as recipient, donor, construction and HQ are defined in the
[glossary](glossary.md).

For plain-language explanations of model scores, external completeness and
contamination, compatible counts and HQ gains, see
[reported quantities](quantities.md#retrieval-proposal-budgets-and-reported-outcomes).

## Set the working directory

This walkthrough continues an experiment initialized with `--output runs/sampleA`.
From the directory where you ran that initialization, enter its project directory:

```shell
cd runs/sampleA
```

All commands below run beside `experiment.json`. Its default `output: "results"`
therefore means `runs/sampleA/results` relative to your earlier working directory.
For the materialized recipe, enter `my_experiment` instead. If your configuration
uses another output path, substitute that resolved path for `results` below.
The relative `--output` paths on these CLI commands use the current working directory.

## Export candidates for external assessment

`results/report.md` lists each completed proposal artifact. For one method the
usual path is `results/methods/METHOD/propose/0001`; use the actual path from the
report, because resumed attempts may have another number. Create a candidate
table, select `proposal_id` values and save them one per line in `candidates.txt`.
The table records full members, recipient, method, construction identity and
sequence-set identity. It includes scores and a default **assessment priority**;
every proposal remains in the table and starts `UNASSESSED`. Model scores are rankings,
not proof of genome quality. `candidates` accepts repeated `--proposals` flags to
inventory more methods together; a candidate export takes one proposal artifact.

A successful `bincomplete experiment` also writes the complete, priority-ordered
`results/candidates.tsv` and its JSON receipt automatically. The command below
creates an additional inventory for selected artifacts, such as a single method;
use a new filename if one already exists for a different artifact set.

```shell
bincomplete candidates --proposals results/methods/hgb_full_pool/propose/0001 --output candidates.tsv
bincomplete export --proposals results/methods/hgb_full_pool/propose/0001 --proposal-ids candidates.txt --output candidates_hgb
```

### Score-based assessment order

Candidate inventories include these additional columns:

| Column | Meaning |
|---|---|
| `added_contig_count` | Number of contigs added to the intact original bin. |
| `mean_added_score` | Arithmetic mean of those donors' saved membership scores against the original bin. |
| `min_added_score` | Minimum of those membership scores. |
| `within_size_rank` | One-based position among proposals adding the same number of contigs, sorted by descending mean score and then proposal ID. |
| `priority_rank` | One-based position after cycling through addition counts in ascending order, taking one proposal from each count per round. |
| `priority_strategy` | `size_balanced_mean`. |
| `assessment_tier` | 1 for the first 40 priority positions in a group; 2 for every remaining position. |

For an added-contig set $A$, with saved donor score $s_i$,
$\mathrm{mean\_added\_score}=|A|^{-1}\sum_{i\in A}s_i$ and
$\mathrm{min\_added\_score}=\min_{i\in A}s_i$. The balanced strategy creates an
ordering, not an additional numerical confidence score. For example, if addition
counts 1, 2 and 3 are present, take the best one-contig proposal, best two-contig
proposal, best three-contig proposal, then the second-best of each, continuing
until **every** proposal has a position. Exhausted groups are skipped.

Ranks are computed separately for each method artifact and recipient. Raw scores
from different methods or bins are not on a common calibrated scale, and none
are probabilities of an HQ genome. The TSV is ordered by assessment tier,
priority rank, recipient, method and proposal ID. Its JSON receipt records the
priority strategy/version, tier boundary, input identity and table hash.

The default 40 is a first-tier scheduling boundary, **not a shortlist that deletes
other rows**. `--priority-limit N` changes that boundary; all proposals, IDs,
memberships and sequence keys remain present. Evaluate controls, then tier 1,
then tier 2. Deduplicate identical constructions and sequences across methods
before external assessment, preserving every source proposal ID. A tool's native
batch context and matched controls still need an explicit assessment plan.

This command only orders an inventory; it launches no quality tools and does not
change fitted models or proposal generation. Quality decisions and compatible
reconciliation remain external. A promising early rank is not permission to
approve a bin or to assume that the deferred pool has no further successes.
Use a new output filename when upgrading a saved version-1 inventory; old tables
and receipts are preserved rather than overwritten.

Repeat the export step for each method being assessed, using only IDs from that
method's artifact and a new output directory. Keep the generated inventory and
export receipts unchanged; put review decisions in separate files.

`candidates_hgb/bins.tsv` maps each exported FASTA filename to `proposal_id` and
recipient, with a file hash and quality status. Join the genome-name column from
your quality tool to its `fasta` column (or the stem if that tool drops extensions),
then keep `proposal_id` as the approval key. A single deduplicated FASTA can map
to several proposals; preserve all mapping rows. The candidate table's
`sequence_set_id` permits exact-content joins across artifacts. Neither a filename
nor a matching recipient ID alone is sufficient to reuse old quality evidence.

### Export original controls

For `--bins` input, the supplied original bin FASTAs are the controls; their
filename stems are the original bin IDs. For assembly-plus-membership-TSV input,
materialize controls from the saved original memberships and assembly with the
commands below. The same procedure also gives bin-FASTA users a mapped control
directory. It needs one completed proposal artifact from the dataset, but selects
none of its proposed additions.

Run beside `experiment.json`, using the actual completed proposal path from your
report. Create a new empty UTF-8 text file, then reconcile that empty approval set
and export it to distinct baseline output directories:

```shell
python -c "open('control_approvals_empty.txt', 'x', encoding='utf-8').close()"
bincomplete reconcile --proposals results/methods/hgb_full_pool/propose/0001 --approved-ids control_approvals_empty.txt --output baseline_control_selection
bincomplete export --selection baseline_control_selection --complete-directory --output baseline_original_controls
```

The empty file means zero approved proposal IDs; it makes no quality decision
about any bin. Keep these new outputs outside completed proposal artifacts and
separate from the later approval selection and refined-bin export. If a name
already exists, preserve it and choose a new name consistently in the commands.

`baseline_original_controls/bins/` contains **all original bins** in the prepared
dataset, including originals outside a restricted recipient list, with their
original members and no candidate additions. Its `bins.tsv` maps original bin IDs
in `recipient` to control FASTA paths in `fasta`, relative to
`baseline_original_controls`. FASTA filenames are generated hashes: use this mapping
rather than guessing a bin ID from a filename. Control rows have an empty
`proposal_id`, `changed=false` and `quality_verdict=UNCHANGED_CONTROL`. That label
describes unchanged membership, not assessed genome quality. For paired assessment,
use the mapping to find each proposed recipient's original control; controls for
other original bins remain available without enlarging the assessed recipient set.
Preserve the mapping and export receipts alongside external assessment results.

## Run CheckM2 and RNA checks with BinComplete

`bincomplete assess` takes the results directory produced by a completed
`bincomplete experiment`. It uses the saved proposal inventory, its receipt,
and the prepared dataset. Write to a separate assessment directory.

Configure CheckM2 1.1.0, Barrnap 0.9, tRNAscan-SE 2.0.12 and the CheckM2
database in `assess.toml`. Use `configs/assess.example.toml` from the checkout
as a template. Set the database path and tool commands for your installation.
BinComplete appends tool options to the configured command arrays. Use the
actual immutable images or the native executables, models, dependencies, and
configuration files in `runtime_files`. BinComplete hashes the declared files
and uses a fixed environment for the assessment. For wrapper-specific environment
variables, list their names in `runtime_env` so their values are passed and hashed.
The runtime file paths and database path are relative to the TOML file. Use
absolute paths for file arguments in command arrays. Keep prepared inputs and
proposal artifacts accessible at the paths saved in the experiment.

For Apptainer, a wrapper can bind the assessment directory and database at
the same absolute paths inside the image and pass the tool arguments through.
This example assumes the two variables name dedicated directories and the
image contains CheckM2. Use analogous commands for Barrnap and tRNAscan-SE.

```sh
#!/bin/sh
set -eu
exec apptainer exec --cleanenv \
    --bind "$ASSESSMENT_DIR:$ASSESSMENT_DIR" \
    --bind "$QUALITY_DATABASE_DIR:$QUALITY_DATABASE_DIR:ro" \
    --env "TMPDIR=$TMPDIR,OMP_NUM_THREADS=$OMP_NUM_THREADS,OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS,MKL_NUM_THREADS=$MKL_NUM_THREADS,TF_NUM_INTRAOP_THREADS=$TF_NUM_INTRAOP_THREADS,TF_NUM_INTEROP_THREADS=$TF_NUM_INTEROP_THREADS,CUDA_VISIBLE_DEVICES=-1" \
    /tools/quality-runtime.sif checkm2 "$@"
```

List the wrapper and image in `runtime_files`, and list the two directory
variables in `runtime_env`. Give the database and assessment output paths only
letters, digits, slashes, underscores, dots, plus signs, or hyphens on Linux.
The pinned tools pass some paths through their own shell commands.

```shell
bincomplete assess --experiment results --config assess.toml --output quality-greedy --mode greedy --max-rounds 1
bincomplete assess --experiment results --config assess.toml --output quality-full --mode full --reuse-from quality-greedy
```

Run in your existing CPU allocation with sufficient memory and disk for CheckM2.
The example uses four CheckM2 threads and two RNA threads, with one tool active
at a time. Assessment results, logs, and temporary directories go under the
assessment output. Configure external wrappers to keep their writes there too.
The `--memory-gib` option controls BinComplete's own allocation budget.

- **Greedy** tests one next-ranked compatible construction per unresolved original
  per round. Selection waits for the entire round, even across multiple batches.
  A compatible HQ pass reserves its contigs and retires its recipient. Increase
  the cumulative `--max-rounds` and rerun the command to continue.
- **Full** tests the whole eligible catalog and uses `reconcile` to maximize
  compatible gains. `--reuse-from` verifies and reuses the baseline and complete
  batches from an assessment created by this command. Full mode keeps their
  original matched controls and considers all qualifying alternatives. Keep the
  source assessment directory available for verification on resume.

The `canonical-trna18-v1` policy uses completeness above 90%, contamination below 5%,
detected 5S, 16S and 23S rRNA, and at least 18 directly reported canonical amino-acid
tRNA types across bacterial and archaeal scans. The RNA results record the
qualifying types as `canonical_trna_types`. Partial rRNA detections count.
Originals meeting this rule are protected. A matched original that passes HQ
or lacks a score blocks gain selection for its recipient. These computational
verdicts still require biological review. Use a new assessment directory when
switching from the earlier reported-label policy.

Read `summary.json` for coverage, gain counts, and stop reason. `activity.json`
identifies the current tool, batch, and log path. Read `baseline.tsv`
for original scores, `assessment.tsv` for individual construction results, and
`before-after.tsv` for one row per original. Solver optimality and the solver bound
apply to the reported selection scope. To retry a full-mode selection with more
time, raise `solver_seconds` and rerun the same command. Verified quality checks
are reused. The TSVs are refreshed when an assessment pauses, reaches its round
limit, or finishes. At the round limit or completion,
`selection.path` in `summary.json` gives the relative path to pass to
`bincomplete export --selection`.

```sh
selection=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["path"])' quality-full/summary.json)
bincomplete export --selection "quality-full/$selection" --complete-directory --output refined-bins
```

Rerun the command after interruption or tool failure. The workflow verifies
completed CheckM2 and RNA checkpoints, reuses valid results, and preserves logs.
Use a new assessment directory after changing inputs, tool identity, or the
batch plan. On POSIX, `SIGUSR1` requests a pause at the next tool boundary.
For Slurm, use a controller-only early signal such as `--signal=B:USR1@300`
with `exec bincomplete assess …` in the batch script. Allow enough time for the
active tool to finish. A pause returns exit code 75, and
`last_published_selection` retains any earlier completed selection. A hard
allocation timeout may require repeating the interrupted tool stage.

For browser review, follow the [Inspector report guide](reporting.md). Prepare
the report's approved IDs and quality evidence in the formats described there.

### Assess full constructions against controls manually

Assess the exported full constructions and original controls using independently
installed quality tools and an explicitly chosen acceptance policy. Keep the quality-tool
versions, settings, results and sequence-to-proposal mapping. Genome-quality
assessment and biological/human review remain separate from membership scoring.

Save the externally approved `proposal_id` values, one per line, in `approved.txt`.
Do not approve a construction solely because it has a high membership score.
Reusing old quality evidence requires exact full-membership **and sequence** matches.
Unknown constructions remain unassessed. No count of gains can be inferred from
the proposal union alone.

## Reconcile approvals across methods

Raw proposals can conflict by recruiting the same contig to different recipients.
Select a compatible set before final export:

```shell
bincomplete reconcile --proposals results/methods/hgb_full_pool/propose/0001 --proposals results/methods/logistic_full_pool/propose/0001 --approved-ids approved.txt --output selected
bincomplete export --selection selected --complete-directory --output refined_bins
```

Repeat `--proposals` for every artifact containing approved IDs. Reconciliation
enforces at most one construction per recipient and disjoint full memberships;
inspect its compatible count and optimization status. Approvals and the raw
recipient union can exceed the compatible selected total.

The solver maximizes the number of distinct original recipients represented by
approved, compatible constructions. Among proven maximum-count selections, it
prefers including proposal IDs in lexical order. It does not optimize membership
scores, completeness, contamination or another quality metric. If you require a
particular biological preference between approved alternatives, resolve that
preference in the approval set before reconciliation. All supplied proposal
artifacts must refer to the same prepared recipient population.

Inspect `selected/selection.json`: `optimal` says whether the maximum count was
proved, `upper_bound` records its bound, and `canonical_tie_break` says whether
the lexical tie break completed. A feasible selection can be exported without
being a proven maximum. Neither status is a biological quality verdict.

Default export writes changed FASTAs and a manifest referencing unchanged bins.
`--complete-directory` exposes all final bins and prefers hard links where
supported. Treat inputs and completed exports as immutable: a hard-linked FASTA
shares file content with another path, potentially an original bin FASTA. Make
independent copies outside the artifact before any downstream sequence edits.
Preserve referenced inputs and artifact receipts. For an incomplete
assembly snapshot, obtain the missing original sequences rather than silently
truncating bin cores. Report any bins necessarily kept unchanged outside the run.

Count quality changes only after choosing the final compatible selection. Compare
each recipient's original bin with its selected final construction under the same
declared external policy. An approved construction for a bin that was already HQ
is not automatically another HQ genome gained. See the
[counting rules](validation.md#counting-quality-changes) for gains, losses,
denominators and unassessed pairs.

To explore a completed selection alongside its original bins, proposals, and
external quality evidence, [export a report for BinCompleteInspector](reporting.md).
