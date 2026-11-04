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

A successful `bincomplete run` also writes the complete, priority-ordered
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

### Assess full constructions against controls

Assess the exported full constructions and original controls using independently
installed quality tools and an explicitly chosen acceptance policy. BinComplete
does not launch CheckM2/RNA or interpret their thresholds. Keep the quality-tool
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
