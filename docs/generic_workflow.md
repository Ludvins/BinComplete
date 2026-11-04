# Train and propose with another binner

Use Python 3.12 for the tested package versions. From the BinComplete source
directory, install into your activated environment:

```shell
python -m pip install -c requirements.txt ".[sets,embeddings]"
bincomplete config init --encoder-path /models/dnabert_s --device cuda:0
```

Use `--device cpu` on a CPU host. If weights are absent, create the configuration
without `--encoder-path` and explicitly run `bincomplete models install dnabert_s`.
Runs do not download weights automatically. See [global configuration](global_configuration.md).
For environment creation, the CPU Torch wheel and installation checks, start with
[installation and a first run](from_scratch.md). Replace the input paths below
with your files and quote paths containing spaces. After installation, change to
an existing external project directory for initialization and execution:

```shell
cd /path/to/project
```

Keep that working directory through each `init`/`preflight`/`experiment` sequence.
The optional table-conversion commands below also stay here and invoke the helper
through an explicit source-checkout path.

## What you need from the upstream analysis

- The **complete original assembly**, including unbinned candidate contigs.
- Original bin FASTAs or a complete contig-to-bin membership table.
- Measured coverage for **every assembly contig**. The standard six need mean
  and variance; do not fabricate variance from a mean or a quality score. Retain
  the full JGI depth table from read alignments or obtain it from the upstream
  analysis. Final per-bin quality summaries do not replace contig coverage.
- Local DNABERT-S weights or compatible supplied vectors, row IDs and their true
  encoder-space manifest for the embedding methods.

Bin FASTAs alone cannot provide missing unbinned donors or measured coverage.
Plain `.fa`, `.fna`, `.fasta` files are accepted, including uppercase extensions.
Decompress compressed bin files first. Mixed plain/compressed folders fail
explicitly so no original bins silently lose protection. Do not merely rename a
compressed file. FASTA IDs are the first header token and must match the assembly.

## Coverage handoff

This walkthrough starts with a measured contig-depth table. If you only have bin
FASTAs, per-bin quality results or a mean-only abundance matrix, obtain the full
depth table from the upstream read-mapping analysis before running all six methods.
The mapping must refer to the same assembly and contig IDs, including unbinned
contigs. Ask for per-contig means and measured depth variances for the selected
sample, either as an accepted full JGI table or the TSV in the [input contract](inputs.md).

If that table was not retained, producing it requires the original reads or
appropriate alignment files and independently installed read-mapping/depth tools.
Perform that upstream step outside BinComplete, retaining the tools, settings and
sample-column choice with your run evidence. BinComplete does not map reads or
reconstruct variance from mean depth, other bins, or quality scores.

## Bin FASTA directory

With the shared configuration in place:

```shell
bincomplete init --assembly /data/sample/assembly.fasta --bins /data/sample/bins --coverage /data/sample/depth.tsv --output runs/sample
bincomplete preflight --config runs/sample/experiment.json
bincomplete experiment --config runs/sample/experiment.json
```

This trains fresh membership models and proposes back into the same dataset.
All supplied bin members stay protected. With no recipient list, all supplied
bins receive proposals. Add `--recipients incomplete_bins.txt` to restrict targets.

## Membership table instead of bin FASTAs

```shell
bincomplete init --assembly /data/sample/assembly.fasta --memberships /data/sample/memberships.tsv --coverage /data/sample/depth.tsv --output runs/sample
bincomplete preflight --config runs/sample/experiment.json
bincomplete experiment --config runs/sample/experiment.json
```

Supply exactly one of `--bins` or `--memberships`. Membership tables use tab-separated
`contig_id`/`bin_id` headers or two headerless columns. Coverage uses tab-separated
`contig_id`, `mean`, `variance`, or a full JGI table. For multiple JGI samples add
`--coverage-profile EXACT_MEAN_COLUMN` to `init`. IDs must match exactly; arbitrary
CSV membership/coverage formats need conversion. See [input contracts](inputs.md).

After proposals complete, follow [export original controls](evaluation.md#export-original-controls)
to materialize the original membership-table bins from the assembly for paired
external quality assessment. That procedure exports all original bins, including
those outside a restricted recipient list, and provides a bin-ID-to-FASTA mapping.
With bin-FASTA input, the original supplied FASTAs already serve as controls.

### Explicit conversion from common upstream tables

If your binner already exports bin FASTAs (for example MetaBAT2, MaxBin2 or
CONCOCT's FASTA extraction), pass that directory directly. Bin names are literal
FASTA filename stems, including `0.fa` or `NA.fa`; all members stay protected.
Check the resulting preflight bin/candidate counts against the upstream output.

Stay in the external project directory. Replace `/path/to/BinComplete` below with
your source checkout and `/path/to/project` with the external project containing
the upstream table. Choose one conversion matching your input; the examples are
alternatives and both write `memberships.tsv`, so do not run both to that output.

For a VAMB-style **headerless cluster-to-contig TSV**, with cluster in column 1
and contig in column 2:

```shell
python /path/to/BinComplete/tools/convert_memberships.py --input /path/to/project/clusters.tsv --headerless --bin-column 1 --contig-column 2 --bin-prefix bin_ --output /path/to/project/memberships.tsv
```

Inspect the upstream file first: formats and headers vary by software version.
The explicit prefix preserves a numeric cluster `0` as `bin_0`; a raw membership
TSV label `0` means unassigned. Match recipient/group files to the converted bin
IDs. Do not prefix contig IDs or change FASTA header tokens.

For a CSV with `sequence` and `cluster` headers:

```shell
python /path/to/BinComplete/tools/convert_memberships.py --input /path/to/project/clusters.csv --delimiter comma --contig-column sequence --bin-column cluster --bin-prefix bin_ --unassigned-label unassigned --output /path/to/project/memberships.tsv
```

Unassigned labels are omitted only when explicitly supplied to this helper.
It rejects conflicting assignments and records source/output hashes and conversion
settings beside the table. It does not infer biological groups, repair missing
contigs or synthesize coverage. CONCOCT-style comma-separated tables can be
converted the same way using their actual column names, or explicit numbered
columns with `--headerless`. A plain abundance/co-abundance matrix is not a JGI
depth table and cannot supply missing variance; obtain measured means and variance
from the original read mapping. Keep the original files unchanged.

Still in the external project directory, use the converted output with the
matching assembly and measured coverage:

```shell
bincomplete init --assembly /path/to/project/assembly.fasta --memberships /path/to/project/memberships.tsv --coverage /path/to/project/depth.tsv --output runs/converted_sample
bincomplete preflight --config runs/converted_sample/experiment.json
bincomplete experiment --config runs/converted_sample/experiment.json
```

## Useful optional controls

Add `--groups related_bins.tsv` when related bins are known. The table maps every
bin to `group_id`; related bins must share a group. Learned methods require at
least two unrelated multi-contig source groups; default attention epoch selection
requires at least four. With no table, bins are assumed unrelated, which does not
establish biological independence.

By default all unassigned assembly contigs form the candidate pool, including
contigs upstream software may have excluded. Add `--donor-ids candidates.txt` for
an explicit subset of unassigned contigs, one ID per line. Preflight reports that
choice and protected/candidate counts; per-recipient length/coverage eligibility
and ranking are applied later. No quality-based exclusion is inferred.

For supplied representations, add all three flags: `--embeddings vectors.npy
--embedding-ids row_ids.txt --embedding-space space.json`. Do not invent a
representation identity for arbitrary vectors. Optional `--composition` and
`--source-ids` retain existing numerical and source-identity contracts.

For the composition-and-coverage method, no Torch or encoder is needed. You may
choose the [core-only installation](from_scratch.md) while still in the source
checkout, before changing to the project directory. Its coverage input still
requires measured variance. From the project directory, run:

```shell
bincomplete init --assembly assembly.fasta --memberships memberships.tsv --coverage depth.tsv --methods hgb_composition_shortlist --output runs/composition
bincomplete preflight --config runs/composition/experiment.json
bincomplete experiment --config runs/composition/experiment.json
```

## Results and a second dataset

Re-run the unchanged experiment command to verify and reuse completed stages.
Use a new `--output` directory for another dataset. Shared installation settings
stay in the global file. For a project-specific copy, pass `--global-config
shared.toml` to `init`, not to `experiment`; see the
[complete project flow](global_configuration.md#project-specific-configuration).
Dataset paths passed at initialization are resolved explicitly, so the
generated configuration can be invoked from another working directory. Update
external paths when transferring it to another machine.

The examples using `--output runs/sample` create a dataset declaration at
`runs/sample/experiment.json`; the run writes its report to
`runs/sample/results/report.md`. Read that report for proposal counts and actual
proposal artifact paths, such as
`runs/sample/results/methods/hgb_full_pool/propose/0001`. The composition-only
example instead writes `runs/composition/results/report.md`. Use the paths from
your report when following the [evaluation walkthrough](evaluation.md).

Training and application on the same dataset is recorded as **transductive** use. It does not
measure performance on an independent dataset. The named evaluation dataset is
where proposals are generated; HQ evaluation is external.
