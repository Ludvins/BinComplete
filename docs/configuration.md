# Current configuration reference

Use `bincomplete config init` for shared installation settings and `bincomplete
init` for each dataset. `preflight --config experiment.json` checks a resolved
dataset before any run. [Shared configuration](global_configuration.md) describes precedence.

## Compact experiment (schema 2)

```json
{
  "schema_version": 2,
  "output": "results",
  "datasets": {"sample": {
    "assembly": "data/assembly.fasta",
    "memberships": "data/memberships.tsv",
    "coverage": "data/coverage.tsv"
  }}
}
```

By default training and evaluation both select `sample`, all enabled six methods
run sequentially, and failures stop dispatch. Supply `global_config` for a project
file, `methods` for a subset, `recipients` for targets, and `on_error` to override
the policy. Replace `memberships` with `bins` for plain bin FASTAs. Do not supply both.
For mmlong2 use `input_format: mmlong2` and `input_root` instead of generic paths.

Dataset fields include assembly, bins/memberships, coverage, coverage_profile,
groups, source_ids, donor_ids, composition, local encoder_path/encoder_options,
or embeddings/embedding_ids/embedding_space. `prepared` instead references a
completed dataset and cannot be combined with raw fields. Paths are relative to
the experiment file. See [exact schemas](inputs.md) and [generic commands](generic_workflow.md).

## Separate training and application datasets

Save the following as `transfer/experiment.json`, replacing the input paths.
This example fits models once on `train` and proposes into `apply`. It inherits
the same encoder and numerical settings from a project global file located
beside the `transfer` directory:

```json
{
  "schema_version": 2,
  "global_config": "../shared.toml",
  "output": "results",
  "training": "train",
  "evaluation": "apply",
  "recipients": "apply_recipients.txt",
  "datasets": {
    "train": {
      "assembly": "/data/train/assembly.fasta",
      "memberships": "/data/train/memberships.tsv",
      "coverage": "/data/train/depth.tsv",
      "groups": "/data/train/groups.tsv"
    },
    "apply": {
      "assembly": "/data/apply/assembly.fasta",
      "memberships": "/data/apply/memberships.tsv",
      "coverage": "/data/apply/depth.tsv"
    }
  }
}
```

`transfer/apply_recipients.txt` contains bin IDs from the application dataset,
one per line. Omit the top-level `recipients` key to target all its bins. The
training groups table must describe the training bins; it does not redefine
application IDs. Each dataset needs complete original memberships and measured
coverage. If either JGI table contains multiple profiles, add that dataset's
`coverage_profile` with its exact mean-column name.

```shell
bincomplete config init --output shared.toml --encoder-path /models/dnabert_s --device cpu
bincomplete preflight --config transfer/experiment.json
bincomplete experiment --config transfer/experiment.json
```

The output is `transfer/results/report.md`. Input/configuration paths resolve
relative to the file containing them; adjust absolute example paths for your
host. Training needs at least four unrelated multi-contig groups when using
default attention selection. Both datasets must use compatible feature and
encoder contracts, including compatible supplied-vector provenance if used.
Cross-dataset application alone is not a claim of independent validation: report
any biological overlap and assess proposed genomes externally. See [training](training_protocol.md).

## Standalone experiment (schema 1)

`bincomplete init --output example --device cpu` creates explicit settings and
input placeholders. This schema requires named training/evaluation datasets and
method entries, ignores ambient global settings, and defaults to `on_error=continue`
unless overridden. The initializer explicitly uses `stop`. Preserve the materialized
files when archiving a run. Configurations do not support legacy `extends` or `study`.

## Supported method overrides

See [settings explained](quantities.md#training-and-representation-settings) for
the meanings of iterations, learning rate, regularization, epochs and batch size,
and [search quantities](quantities.md#retrieval-proposal-budgets-and-reported-outcomes)
for the proposal cap, rank depth and added-length fraction.

The six named profiles fix component/formula choices. Ordinary overrides use a
`params` object, such as `classifier.params.max_iter` for HGB,
`classifier.params.C` for logistic, or `scorer.params.max_epochs` for attention.
Search additionally accepts a registered `name` and `params`: the standard
`expanded_ranked_alternatives` or explicit legacy `ranked_alternatives`.
Do not copy arbitrary historical evaluator/representation/feature blocks.
Preflight validates supported constructor parameters. A changed recipe needs a
new output directory; it does not relabel or change previous models.

The standard search cap is 128, singleton depth 50, boundary width 2 and added-length
fraction 1. These are candidate-generation settings, not genome-quality thresholds.
Use `search.params.max_proposals` in new method overrides, or `max_proposals`
inside the shared `[search]` table. The equivalent stored name `max_checks`
remains supported; do not specify both. The standard settings retain their stored
keys and values.
Model scores are uncalibrated membership scores. Quality tools, HQ thresholds,
approvals and biological acceptance remain external.
