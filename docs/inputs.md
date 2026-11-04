# Input contracts

See the [glossary](glossary.md) for bin, recipient, donor, core and construction.
The small tables below illustrate formats; they are not a complete trainable
dataset. Fields inside TSV examples are separated by tabs.

## Assembly and original memberships

Supply a plain assembly FASTA, exactly one bin FASTA directory or membership TSV,
and raw coverage. FASTA IDs are the first header token. Duplicate IDs, multiple
original-bin assignments and bin sequences differing from the assembly are rejected.
Compressed random-access input is rejected instead of silently copied/decompressed.
Bin directories accept plain `.fa`, `.fna`, `.fasta` files. A mixed directory
containing compressed bin FASTAs is rejected in full; decompress first so no
original bin loses protection. Arbitrary membership/coverage CSV formats need
explicit conversion to the TSV contracts below.

Memberships accept `contig_id`/`bin_id`, `contig`/`bin`, or two headerless columns.
Empty, `unbinned`, `unassigned`, `-`, `NA` and `0` labels mean unassigned. All assigned
bins protect their members. These unassigned labels apply only to membership
tables: FASTA filename stems are literal bin names, so `0.fa` and `NA.fa` remain
protected bins. Empty bin FASTAs are rejected. Optional donor IDs restrict
unbinned candidates. See [explicit table conversions](generic_workflow.md).

For example, `memberships.tsv` can contain:

```text
contig_id	bin_id
contig_1	bin_A
contig_2	bin_A
contig_3	bin_B
contig_4	bin_B
contig_5	unassigned
```

An assembly contig absent from the table is also unassigned. Keep all original
assigned memberships; omitting a bin or its members changes the protected
population and training labels. Memberships are proxy labels supplied by the
upstream analysis: BinComplete does not establish their biological correctness.

## Coverage

Read depth counts mapped reads covering a base. The [DNA illustration and
measurement explanation](quantities.md#read-count-length-and-depth) distinguish
read count, read length, mean depth and variance before describing table columns.

Coverage uses `contig_id`, `mean` and optional `variance`. Means must be finite,
nonnegative and present for all contigs. The four feature-classifier profiles
(`kmers`, `dnabert_s`, `hgb_full_pool`, `logistic_full_pool`) need measured variance.
Single/cosine only need means. JGI tables are accepted; multiple samples require an explicit
mean-column name via `--coverage-profile`. Quality columns never enter features.

When a variance column is supplied, every assembly contig needs a finite,
nonnegative variance as well. Use measured, untransformed coverage from the
declared read-mapping profile and retain the upstream measurement policy. Do not
substitute a standard deviation, across-sample abundance variation or a per-bin
quality metric for that profile's contig coverage variance.

```text
contig_id	mean	variance
contig_1	12.0	4.0
contig_2	10.5	3.2
contig_3	21.0	8.0
contig_4	19.0	7.1
contig_5	5.0	1.5
```

A two-sample JGI header and illustrative row look like:

```text
contigName	contigLen	totalAvgDepth	sampleA.bam	sampleA.bam-var	sampleB.bam	sampleB.bam-var
contig_1	5000	16.0	12.0	4.0	20.0	9.0
```

For `--coverage-profile sampleA.bam`, the loader selects `sampleA.bam` and its
adjacent `sampleA.bam-var` column. It does not use `totalAvgDepth` as the selected
sample mean. Match the mean-column name exactly, and retain the sample's variance
column when exporting a table. Coverage is still required for unassigned contigs.
See the [per-base moment definitions and upstream-convention caveat](features.md#what-exactly-are-mean-depth-and-depth-variance)
for the precise meanings of mean depth and within-contig variance.

## Related-bin groups

Optional `--groups` maps every bin to a related group using `bin_id`/`group_id`
headers or two headerless columns. Negative sampling excludes related groups.
Neural epoch selection splits groups before generating selection-training and
validation masks/negatives, then refits all sources. Without groups, bins are
assumed distinct; biological independence is not established by this assumption.

For a dataset with the five bin IDs below, `groups.tsv` could contain:

```text
bin_id	group_id
bin_A1	related_A
bin_A2	related_A
bin_B	related_B
bin_C	related_C
bin_D	related_D
```

Here `bin_A1` and `bin_A2` share a declared relatedness group. Choose and record a
grouping criterion appropriate to the study using available biological evidence;
bin names or a desire for more training groups are not evidence of independence.
There is no built-in biological relatedness threshold. Map every supplied bin
exactly once, including single-contig bins and bins outside the recipient list.

A training group counts toward the minimum only if it contains at least one
bin with two or more contigs. Learned profiles need at least two such distinct
groups; default attention epoch selection needs at least four. The example has
four eligible groups only if each named group has a multi-contig source bin.
Grouping does not validate the original membership labels or turn same-dataset
application into an independent test.

## Precomputed DNABERT-S vectors

Precomputed DNABERT-S inputs require a finite, already L2-normalized float16/float32 NPY
matrix, a row-ID text file, and encoder-space JSON. Supply all three together as
`--embeddings vectors.npy --embedding-ids row_ids.txt --embedding-space space.json`.

`row_ids.txt` is UTF-8 text with one unique, exact contig ID per line: no header,
blank rows or extra columns. Do not add spaces around IDs; the loader does not
trim them. Its number of lines must equal the NPY matrix's number of rows. The
first line names matrix row 0, the second names row 1, and so on. For example:

```text
contig_3
contig_1
contig_2
```

Here `vectors.npy[0]` must contain the vector for `contig_3`, regardless of that
contig's position in the assembly FASTA. Include vectors and IDs for **every
assembly contig**, both assigned and unassigned. Donor or recipient restrictions
do not reduce this requirement; vectors only for binned contigs or selected donors
are insufficient. Preparation rejects a missing assembly ID with
`Embedding population omits assembly contigs`.

The matrix may contain a superset of the assembly IDs and may use a different row
order from the assembly. The loader maps exact IDs to rows through indirection
rather than copying or reordering the matrix. All supplied rows, including extras,
must satisfy the vector contract.
Float16 storage stays in place; validation and scoring use bounded float32 blocks.
The storage dtype and observed norm error are recorded, without renormalizing the
saved vectors. Norm validation allows 5e-4 absolute error for float16 quantization
and 1e-4 for float32. Neither setting permits zero or nonfinite vectors.

The JSON declares `encoder` (`dnabert_s`), `encoder_sha256`, `tokenizer_sha256`,
`pooling`, `window_bp`, `windows`, `max_tokens`, `normalization` (`l2`) and
`dimension`, describing the actual vector production. Compatibility compares
feature formulas and encoder meaning separately from matrix hashes and row IDs.
Equal dimensions alone are insufficient.

### Inspecting a supplied-vector contract

Use the exact checkpoint and options used to produce your vectors. This command
hashes local files and writes their encoder-space description; it never loads
model code, downloads weights or computes embeddings:

```bash
bincomplete models inspect dnabert_s --path /models/dnabert_s --encoder-options encoder-options.json --output embedding-space.json
```

For example, a standard experiment's options file is:

```json
{
  "pooling": "non_special_bp_weighted_windows",
  "window_bp": 6000,
  "windows": 4,
  "max_tokens": 2048,
  "reverse_complement": true
}
```

The resulting manifest has this complete shape (hash values below are explanatory
placeholders, not valid provenance for an existing matrix):

```json
{
  "encoder": "dnabert_s",
  "encoder_sha256": "<aggregate SHA256 of checkpoint files>",
  "tokenizer_sha256": "<aggregate SHA256 of tokenizer files>",
  "pooling": "non_special_bp_weighted_windows",
  "window_bp": 6000,
  "windows": 4,
  "max_tokens": 2048,
  "reverse_complement": true,
  "normalization": "l2",
  "dimension": 768
}
```

`dimension` comes from the actual checkpoint's `hidden_size`; it is not always
768. The two supported local pooling protocols are
`masked_mean_uniform_windows` (the low-level CPU default, without a
`reverse_complement` key) and `non_special_bp_weighted_windows` (with an explicit
boolean `reverse_complement`). Every key in the space dictionary participates
in compatibility, including extra keys; omitted and false are not interchangeable.

Each aggregate hash is SHA256 of a canonical JSON mapping from checkpoint-relative,
forward-slash filenames to the SHA256 of each file's bytes. The encoder mapping
includes all `.bin`, `.safetensors` and `.py` files plus `config.json`; the tokenizer
mapping includes files whose basename contains `token`, plus `vocab.txt` and
`merges.txt`. Canonical JSON uses sorted keys, separators `(',', ':')`, default
ASCII escaping and UTF-8 encoding, as implemented by `bincomplete.artifacts.fingerprint`.
The array's file hash and contig row IDs are recorded separately during preparation.

Inspection describes the requested protocol; it cannot establish how arbitrary
existing vectors were produced. Recover their original production metadata rather
than inventing or editing identities to make a model accept them. An existing
output manifest can be reused only if it is identical. A space exported with the
low-level defaults is not the standard experiment's space.

Alternatively, `--encoder-path` references local frozen weights/tokenizer. The
encoder policy depends on the entry point:

| CPU encoder policy | Token limit | Strands and pooling |
| --- | --- | --- |
| Low-level `prepare --encoder-path` without encoder options | 512 | Forward strand, masked token means and mean window pooling |
| Standard experiment | 2,048 | Both strands, non-special-token pooling and base-pair-weighted window pooling |

Both use up to four uniformly spaced 6,000-bp windows and L2 normalization.
These are distinct encoder identities and are not assumed equivalent to each
other or arbitrary cached embeddings. Explicit encoder options are recorded in
the contract; see [shared encoder settings](global_configuration.md#shared-controls).
Dependencies are lazy.

## MMLong2 handoff

The mmlong2 adapter discovers `results/*_assembly.fasta`, `results/bins/`,
and the full primary JGI table under `tmp/binning/mapping/1-*.tsv`, excluding
the derived `_tr1.tsv` and `_tr2.tsv` views and checking primary headers.
Without bin FASTAs it can use
`results/*_contigs.tsv`. Missing/ambiguous paths require overrides. It never reads
quality tables to select sources or imports mmlong2 modules.
Retain the primary mean/variance table alongside `results/` when archiving inputs;
the final mean-only summary cannot reconstruct variance. Use `init --coverage`
and `--coverage-profile` for explicit selection. The final assembly includes
upstream-excluded sequences; all unassigned contigs are candidates unless an
explicit `donor_ids` restriction is supplied. `preflight --config FILE` checks
the handoff and reports that candidate population without creating a run.

Pass the sample directory containing both `results/` and the retained depth
table as `--input-root`, rather than passing `results/` itself:

```text
sampleA/
  results/
    sampleA_assembly.fasta
    bins/
      bin_A.fa
      bin_B.fa
  tmp/
    binning/
      mapping/
        1-depth.tsv
```

The names here are examples of the discovery patterns, not required sample IDs.
If the original layout is unavailable, map generic input paths explicitly. When
several primary tables or sample columns exist, select the actual file and column:

```shell
bincomplete init --input-root /data/mmlong2/sampleA --coverage /data/mmlong2/sampleA/tmp/binning/mapping/1-depth.tsv --coverage-profile sampleA.bam --output runs/sampleA
bincomplete preflight --config runs/sampleA/experiment.json
```

This creates `runs/sampleA/experiment.json`; its default report will be
`runs/sampleA/results/report.md` after the experiment completes. Inspect preflight
bin, protected-contig and candidate counts against the upstream handoff before
running it.
