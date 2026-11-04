# One-command experiments


For the recommended compact six-method workflow, use
[shared configuration](global_configuration.md). Initialize each dataset with
`bincomplete init --input-root ... --output ...` or explicit generic input flags.
Alternatively, copy the compact schema-2 example
`configs/experiment.mmlong2.example.toml` to your project directory, set its input
and output paths, then run from the activated environment:

```shell
bincomplete preflight --config experiment.toml
bincomplete experiment --config experiment.toml
```

Compact schema-2 experiments are the recommended interface: `init --input-root`
or generic `init --assembly ... --bins ... --coverage ...` inherit global settings,
select all six by default and stop on errors. Run `preflight --config FILE` first.
See [generic examples](generic_workflow.md) and [global configuration](global_configuration.md).

For a self-contained **schema-1** recipe with explicit per-method settings, use
`bincomplete init --output my_experiment --device cpu` without dataset input
flags or `--compact`, then edit the generated inputs. See the
[materialized recipe guide](portable_recipe.md). Installed packages expose the same operation as
`bincomplete experiment --config comparison.toml` or the Python API
`run_experiment("comparison.toml")`. Both TOML and JSON support schema 1 and schema 2;
YAML is not required. Paths inside the recipe and settings files are relative to
the recipe's directory. `--output` may override the output path; a relative CLI
override is relative to the current working directory.

Schema 1 explicitly selects methods, settings, training/evaluation datasets,
optional recipients and the common resource policy; it ignores ambient globals.
Schema 2 inherits shared settings and the enabled global methods when omitted.
The shipped compact example therefore uses all six methods only when the shared
configuration enables all six. Its input/output paths are explicit, but its
method settings are not self-contained.
All resolved method settings and the input/configuration/code identities are
frozen in `experiment.json` and the individual model artifacts.

## Input roles

`training` and `evaluation` reference named entries under `datasets`. Use one
name for both roles to prepare once. Identical input definitions under two names
also share one prepared artifact. To transfer models between datasets, declare
two entries. Each may supply generic `assembly`, `memberships` or `bins`,
`coverage`, optional `groups`, donor IDs and embeddings/encoder inputs. The
`mmlong2` format accepts the same overrides as `prepare`.

A dataset can instead contain only `prepared = "/path/to/prepared"`. Prepared
inputs and external matrices are verified and referenced rather than copied.
An optional top-level `recipients` is either a unique list of evaluation bin IDs
or a text-file path containing one ID per line. The default is all supplied bins;
no quality-based source or recipient selection occurs.

Every method entry declares `method` and optional `settings`, as an inline object
or a TOML/JSON settings path. Add a unique `id` to compare two parameter variants
of the same method. Resources apply to all methods; model parameters may differ.
Methods run sequentially. Rebuildable representations share one bounded cache,
and training buffers are released before proposal generation. Input FASTAs and
precomputed embeddings stay in place. Individual `prepare`, `fit`, `propose`,
`reconcile` and `export` commands remain available.

## Outputs and resumption

- `experiment.json`: immutable resolved recipe, dependencies and code identity.
- `status.json`: durable dataset/method/stage status and attempt history.
- `datasets/<name>/0001/`: shared prepared metadata for raw inputs.
- `methods/<id>/fit/0001/`: separately verifiable model artifact.
- `methods/<id>/propose/0001/`: full scores and membership proposals.
- `comparison.json`, `summary.json` and `report.md`: structural comparison/status.
- `errors/`: failure details retained outside immutable child artifacts.

Invoke the same command to resume. Completed stages are hash-verified and reused.
If an unfinished stage lacks a completion receipt, its files are preserved and a
new numbered attempt is created for that stage. A completed fit is reused when
only proposal generation needs retrying. This does not claim to resume an
unfinished optimizer inside a partial fit. Corrupt completed artifacts fail
verification instead of being silently replaced.

In schema 1 the default `on_error = "continue"` allows other selected methods to finish after
a method fails; `"stop"` stops dispatching further methods. A run containing a
failed/unattempted method exits with code 1 and has no terminal completion receipt.
All-success runs exit with code 0. Compact schema 2 defaults to `on_error = "stop"`;
either schema can override it. A cross-process experiment lock prevents two
invocations from dispatching the same experiment simultaneously. Changed inputs,
settings, dependencies or package code require a new output directory.

## Meaning of comparison counts

The comparison reports evaluated targets, candidates scored, distinct membership
constructions, identical constructions shared between methods, unique
constructions and the union without duplicate memberships. Pairwise counts use
the common evaluated target set. A construction identity contains the original
recipient and its complete contig membership; run-specific proposal IDs are not
used to define overlap. Donors proposed for multiple recipients are flagged as
prospective assignment conflicts. Raw alternatives are not a compatible final
selection.

These are proposal counts, not HQ gains. Quality tools and acceptance rules remain
external. Externally approved proposal IDs can subsequently be reconciled and
exported through the existing commands. Saved quality evidence may be reused
only after exact full-membership and sequence matching. Reproducing a prior result
requires matching its input, preprocessing, numerical environment and settings.
No saved model or approved construction is an experiment input unless explicitly
supplied through a separate supported operation.

## Explicit numerical input contracts

A dataset may declare `source_ids` as a two-column `bin_id/source_id` TSV. This
keeps source-specific random sampling stable across renamed input files; it does
not select bins by quality. Bin iteration remains sorted by bin ID. `groups`
continues to map bin IDs to related-group IDs.

Method settings may explicitly declare `preprocessing.composition` with
`alphabet_order` (`ACGT` or `ATGC`), a nonnegative `pseudocount`, and
`renormalize_float32`. `preprocessing.coverage_dtype` selects `float32` or
`float64`. These options belong to the fitted numerical contract and cache
identity. Defaults preserve the original package profiles.

For a frozen local DNABERT-S encoder, dataset `encoder_options` may select
`pooling = "non_special_bp_weighted_windows"`, `window_bp = 6000`, `windows = 4`,
`max_tokens = 2048`, `reverse_complement = true`, `storage_dtype = "float16"`,
`device = "cuda"`, `precision = "bfloat16"`, and `batch_size = 8`. This profile
recursively splits oversized tokenizations, pools non-special tokens, combines
pieces by base-pair weight, and normalizes each contig. Token truncation is not
used. These are explicit encoder options, recorded in the numerical input contract.
The default CPU profile remains available.

Strand-aware inference checkpoints each completed group in the derived cache.
A matching interrupted cache entry can resume without re-encoding completed
contigs. The completed array is shared across training and evaluation datasets
using the same assembly and encoder contract; no assembly or pretrained weight
copies are made. The cache ceiling covers checkpoint and completed array files.
Set `HF_MODULES_CACHE` to a writable local directory when the tokenizer/model
uses Transformers dynamic modules. This cache contains encoder Python modules,
not another copy of the pretrained weights.

Original input composition can be supplied with dataset `composition` (or
`prepare --composition`). This is a CSV/TSV whose first column is the contig ID,
with canonical four-mer columns named `0` through `135`; unrelated columns are
ignored. Declare their alphabet order in method preprocessing. The values may be
counts or frequencies and must be finite and nonnegative with positive row mass.
The profile normalizes them without adding another pseudocount or replacing
zeros. In particular, an upstream MMLong table need not equal a later direct
FASTA recomputation. Preserve that table when reproducing an earlier experiment.
Supplied-table and sequence-derived models have distinct compatibility contracts.

## Shared input metadata and bounded sequence windows

Datasets referencing the same assembly path in one experiment share a verified
sequence index, IDs, lengths and per-sequence hashes. Cohort memberships, groups,
source IDs and donor eligibility remain independent. Identical raw coverage and
supplied composition inputs also reuse their immutable arrays; different sample
coverage profiles retain separate arrays. Preparation compares complete input
SHA-256 identities before sharing, and normal verification checks the referenced
artifact receipts and outputs.

For separately prepared contexts, use `prepare --assembly-index /path/to/first`
or dataset `assembly_index` pointing to the first completed prepared dataset.
No hard links, FASTA copies or new pretrained-weight copies are required. Keep
referenced prepared datasets while any later artifact depends on them. The
shared index is a persistent input dependency, separate from the evictable
representation cache. `Dataset.array_path(name)` resolves local and shared array
files for API callers; assuming that every array is physically in each dataset
folder is no longer sufficient.

The index validates FASTA line layout while computing sequence hashes. Frozen
encoder inference reads the requested uniform windows directly when wrapping is
regular, merges overlapping reads, and retains the original window order,
strands, token splitting, batches and floating-point pooling. Irregular FASTA and
older indexes use the previous full-sequence fallback. This reduces I/O and peak
sequence-buffer memory without changing encoder semantics. Changing GPU,
precision or batch size is a separate numerical experiment and is not part of
this I/O optimization.


## Logistic regression and saved source identities

`logistic_full_pool` is the sixth supported profile and is included in the
example recipe. It uses the existing composition, coverage and embedding features,
original-membership masking and source-balanced weights. The weighted feature
scaler is fitted only on training rows and is saved with the classifier.
Inference never refits it. The supported profile set is listed in the
[standard recipe](portable_recipe.md).

Existing model manifests remain immutable and the strict source-compatibility
guard remains active. Use the original frozen package for older model artifacts;
new fits record the installed implementation's source identity. Adding or
renaming implementation modules can change that identity even when formulas are
unchanged. Project results and quality evidence remain external; enabling a
profile does not establish HQ gains for a new dataset.
