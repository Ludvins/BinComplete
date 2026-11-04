# Configure once, reuse across datasets

BinComplete separates **installation settings** from **dataset inputs**. Configure
the encoder, device, resource limits, proposal search and method parameters once:

```shell
bincomplete config init --encoder-path /path/to/dnabert_s --device cuda:0
bincomplete config show --resolved
```

Reference an existing local checkpoint in place; no download is needed then.
Use `--device cpu` for an explicit CPU variant. The global
file is editable TOML, and `show --resolved` displays the complete effective method
recipes, including parameters supplied by the named preset.

The default file is `%APPDATA%/BinComplete/config.toml` on Windows or
`$XDG_CONFIG_HOME/bincomplete/config.toml` on Linux (default `~/.config`). Override
discovery with `BINCOMPLETE_CONFIG`, or reference a particular file through
`global_config` in an experiment. `config init --output shared.toml` creates a
project-specific global file without modifying the user-wide configuration.
Existing files are never overwritten by `config init`.

## Project-specific configuration

Run this sequence from your external project directory after installing
BinComplete. Replace the assembly/bin/depth paths with your inputs. This CPU
example creates a dedicated shared file and explicitly installs the official
checkpoint at its configured location:

```shell
bincomplete config init --output shared.toml --device cpu
bincomplete models install dnabert_s --global-config shared.toml
bincomplete config show --path shared.toml --resolved
bincomplete init --global-config shared.toml --assembly /data/sample/assembly.fasta --bins /data/sample/bins --coverage /data/sample/depth.tsv --output runs/sample
bincomplete preflight --config runs/sample/experiment.json
bincomplete experiment --config runs/sample/experiment.json
```

For existing weights, add `--encoder-path /models/dnabert_s` to the first command
and skip the installer. Replace `--bins` with `--memberships` for a membership TSV.
Use a fresh `shared.toml` and experiment directory, or inspect existing files
before editing them; initialization never overwrites them.

`init --global-config` records the shared file's absolute path in the dataset
configuration. Later `preflight` and `experiment` commands read that reference;
they do not take a `--global-config` option. To inspect a particular shared file,
`config show` uses `--path`. The installer uses `--global-config` to choose the
same file when selecting its model destination.

Relative `encoder_path` and `cache` values inside a shared file resolve from that
file's directory. A hand-written experiment's relative `global_config` value
resolves from the experiment file's directory. CLI paths resolve from the current
working directory. On another host, update those references and use a new output
directory when inputs, settings or numerical execution change.

## Per-dataset commands

For an original mmlong2 output directory:

```shell
bincomplete init --input-root /data/mmlong2/sampleA --output runs/sampleA
bincomplete preflight --config runs/sampleA/experiment.json
bincomplete experiment --config runs/sampleA/experiment.json
```

Initialization creates the experiment directory, `experiment.json` dataset
declaration and `README.txt`; the run creates its `results/` subdirectory. It
inherits enabled methods, encoder options, shared cache and method settings. It
does not download anything. Methods execute sequentially; embeddings are computed
once per dataset or reused from an exact compatible cache. Five models fit from
original memberships when all six methods are enabled; cosine is analytic.
If no recipient list is supplied, proposals cover all supplied bins.
Add `init --recipients incomplete_bins.txt` for
a fixed subset. No quality table is used as training supervision.

For the command above, the report is `runs/sampleA/results/report.md`, and the
frozen resolved run recipe is `runs/sampleA/results/experiment.json`. The outer
`runs/sampleA/experiment.json` remains the input declaration. Use actual proposal
artifact paths from the report when exporting candidates for assessment.

For generic input files, pass `init --assembly assembly.fasta --bins bins --coverage
depth.tsv --output runs/sampleA`, or replace `--bins` with `--memberships memberships.tsv`.
These input flags imply the compact workflow. Optional flags include `--groups`,
`--source-ids`, `--donor-ids`, `--recipients`, `--composition`, and supplied
`--embeddings`/`--embedding-ids`/`--embedding-space`. Select a subset with
`--methods hgb_composition_shortlist`; this composition-only subset does not require encoder weights.
`init --compact --output runs/sampleA` remains available for editable placeholders.
See the [generic walkthrough](generic_workflow.md) for copyable examples. Flat
server snapshots need explicit input mapping; they do not have the original
mmlong2 `results/` and `tmp/` layout. Coverage still needs measured mean/variance.
Global configuration does not reconstruct missing data or select a scientific
baseline for you.

The mmlong2 adapter skips `_tr1` and `_tr2` derived depth tables. For multiple
primary tables, select one with `init --coverage PATH --coverage-profile COLUMN`.
The complete handoff is the final assembly, final original bins or memberships,
and the primary full JGI table with measured variance. A copied `results/` alone
may lack the depth table. The default candidate pool is **all unassigned assembly
contigs**, including upstream-excluded contigs; use `--donor-ids PATH` to record
an explicit restriction. Preflight reports protected-bin and candidate counts.

`preflight --config FILE` reads and checks inputs without creating a run/cache,
loading an encoder model, fitting or assessing quality. It checks sequence/ID
joins, supplied vectors, measured variance, weights, dependencies, declared
devices, recipients and training-group requirements. It reports resource budgets,
metadata estimates and training-row caps, not an exact peak-memory/runtime forecast.
Large assemblies and supplied vectors must be read; this can take time. A ready
report does not guarantee optimizer convergence or HQ gains. Input errors are
concise; add `--debug` for a traceback. `config show --resolved` validates recipes,
whereas preflight checks a particular dataset and its required assets.

Supplied embeddings are also supported: declare `embeddings`, `embedding_ids`
and `embedding_space` in the dataset. Then the global encoder path is not injected.
The strict feature and encoder identity guards remain active.

## Shared controls

[Understanding the quantities](quantities.md#training-and-representation-settings)
explains what the numerical settings control, with separate examples for
[retrieval and search](quantities.md#retrieval-proposal-budgets-and-reported-outcomes).
For example, singleton search depth counts donor ranks; it is unrelated to read depth.

The generated global file exposes these sections:

| Section | Controls |
| --- | --- |
| `encoder_path`, `device`, `cache` | Local weights, numerical device, reusable representation cache |
| `enabled_methods` | Explicit subset of the supported six methods |
| `[resources]` | Threads, working-memory budget, cache ceiling, scoring block size |
| `[encoder]` | Windows, token limit, strands, precision, batch size and vector storage |
| `[retriever]` | Candidate shortlist size, minimum length and coverage/length eligibility |
| `[search]` | Proposal cap, singleton depth and prefix-boundary alternatives |
| `[methods.NAME...]` | Classifier, attention and other supported per-method overrides |

Standard defaults are two numerical threads, a 6 GiB allocation budget, a 10 GiB
rebuildable-cache ceiling and 4,096-row scoring blocks. The allocation budget is
not an OS RSS cap. Model weights, fitted models and preserved output artifacts do
not count toward the rebuildable-cache ceiling. Keep inputs and outputs outside
that cache. See [resource boundaries](architecture.md) for retained and temporary files.

For example:

```toml
[search]
max_proposals = 128
singleton_depth = 50
boundary_width = 2
max_added_fraction = 1.0

[methods.hgb_full_pool.classifier.params]
max_iter = 150
max_leaf_nodes = 7
learning_rate = 0.05
min_samples_leaf = 30
l2_regularization = 2.0
early_stopping = false
random_state = 73019

[methods.single.scorer.params]
layers = 2
heads = 4
width = 128
max_epochs = 40
```

These are the current preset values, not newly tuned parameters. Increasing a
search cap alone does not create additional search templates: singleton depth
and boundary width determine the alternatives available below that cap.
`max_proposals` is the preferred input name for the proposal cap. Saved recipes
retain `max_checks` as the equivalent compatibility key; supply only one spelling.
Per-method tables use the stored profile IDs, so `[methods.single...]` configures
the method also accepted as `attention_query`. See [method naming](migration.md#method-names).
`[encoder].precision = "auto"` resolves to FP32 on CPU and BF16 on CUDA, and the
resolved numerical profile is recorded. An explicit precision can be selected.
There is no silent fallback from an unavailable GPU.

Measured **coverage depth is input data** for each dataset. `coverage_fold` is a
candidate eligibility setting; it does not replace observed coverage means or
variances.

Resolution order is: built-in `standard_v1` preset, shared global settings,
global per-method settings, then explicit per-experiment method settings. A
compact experiment may override resource limits or its device. Every run saves
the fully resolved recipes, global file hash, inputs and source identities.
Changing the global file cannot change a running recipe; resuming with changed
settings is rejected and requires a new output directory. Keep the global file
with the experiment or use an explicit project configuration for archival use.

Only schema-2 compact experiments inherit global settings. Existing schema-1
configurations remain self-contained and ignore ambient global configuration.
The original standalone `init --output ... --device ...` remains available for
fully materialized settings, including the earlier `--embedding-mode` interface.

## Optional model installation

If no local checkpoint exists:

```shell
bincomplete config init --device cuda:0
bincomplete models install dnabert_s
```

Without `--encoder-path`, `config init` writes the default pinned checkpoint
location into `encoder_path`. For the default revision, the installer reads the
selected global configuration and installs there. It does **not** rewrite that
configuration or search for weights elsewhere. Its JSON result reports the model
`path`; `config show --resolved` displays the configured path, and dataset
`preflight` checks that the required local assets are present.

The built-in model location ends in
`bincomplete/models/dnabert_s/1cdf84d992ace6f3e75c7356774b4da088c8dc7c`, under:

- Windows: `%LOCALAPPDATA%` (normally `~/AppData/Local`).
- Linux: `$XDG_DATA_HOME` (default `~/.local/share`).

An explicit `encoder_path` in the selected shared file replaces that location.
For a project file, use `models install dnabert_s --global-config shared.toml`.
The model directory is separate from the evictable representation cache.

The installer uses the [official DNABERT-S checkpoint](https://huggingface.co/zhihan1996/DNABERT-S),
pinned by default to commit `1cdf84d992ace6f3e75c7356774b4da088c8dc7c`, an explicit
immutable upstream revision. It uses
[`snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download)
with that exact revision, saves file hashes and reuses verified installations
offline. It downloads the upstream Python model code and weights but executes no
model code, encoding or training during installation. Normal local inference
uses the checkpoint's custom code through the existing local loader.

Model assets live outside the repository. Their installation receipt is beside
the model folder, so it does not alter the encoder-space identity. A user-managed
valid checkpoint is referenced in place and never overwritten. Failed download
files and their attempt contract are preserved; retry only with the same revision
or select a new directory. `--revision` requires a full commit SHA; floating
`main` is rejected. `--output` installs to a chosen directory; point the global
`encoder_path` there when using a custom location.

Installation requires the `embeddings` extra and network access. Scientific runs
never install dependencies or download weights automatically. Shared configuration
sets execution defaults; it does not establish performance on a new dataset.
