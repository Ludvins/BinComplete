# Python API and command index

The [first-run guide](from_scratch.md) uses experiment configurations and is the
recommended starting point. The lower-level API lets a script manage prepared
datasets, fitted models and saved proposals explicitly. It has different resource,
encoder and search defaults from the standard experiment; see [the entry-point
defaults](architecture.md#resource-and-artifact-boundaries) and [portable recipe](portable_recipe.md).

## Artifact flow

This example assumes a successful standard six-method run with the default
configured ID `hgb_full_pool`. With a method subset or custom IDs, inspect
`summary["methods"]` and choose a successful entry containing a `proposals` path;
summary keys are configured method IDs, not always profile names.

```python
from pathlib import Path
from bincomplete import (
    Dataset, ModelArtifact, ProposalArtifact, SelectionArtifact,
    ResourceLimits, prepare_dataset, fit, propose, candidate_table,
    reconcile, export, run_experiment,
)

# Recommended configuration-driven workflow; the config selects the methods.
experiment = run_experiment("runs/sample/experiment.json")
summary = experiment.manifest

# Open a saved artifact using the actual path recorded in that summary.
proposals = ProposalArtifact(Path(summary["methods"]["hgb_full_pool"]["proposals"]))
candidate_table([proposals], "review/candidates.tsv")
```

Constructing a path wrapper does not load a model into memory. Model, proposal
and selection `.manifest` properties check their recorded receipts. In contrast,
`Dataset.manifest` only reads its metadata: call `Dataset.verify()` to check the
dataset receipt, recorded outputs and external inputs. Use trusted model files only; deserialization
is not a safe way to inspect an unknown model. See [trust boundaries](security.md).

The signatures below show required arguments and the most common optional ones;
the linked definitions document all options. `output` is a path to an artifact
directory except where a table filename is specified. A parameter receiving a
`Dataset` or `ModelArtifact` needs that wrapper, not an unwrapped string path.

| Function | Inputs and important keyword arguments | Return value |
| --- | --- | --- |
| [`prepare_dataset`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/data.py) | `output`, then keyword `assembly`, one of `bins` / `memberships`, `coverage`; optional `groups`, `coverage_profile`, embeddings or encoder fields, `limits`; MMLong2 instead uses `input_format="mmlong2"`, `input_root` | `Dataset` |
| [`fit`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/engine.py) | `dataset: Dataset`, `output`, required keyword `method`; optional `settings` (dict or None), `limits`, `cache` (str, Path or None) | `ModelArtifact` |
| [`propose`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/engine.py) | `model: ModelArtifact`, `dataset: Dataset`, `output`; optional `recipients: list[str]`, `limits`, `cache` | `ProposalArtifact` |
| [`run`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/engine.py) | `training: Dataset`, `recipients: Dataset`, `output`, required keyword `method`; optional `settings` (dict or None), `limits`, `cache` (str, Path or None) | `ProposalArtifact`; fits then proposes for one method |
| [`candidate_table`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/inventory.py) | List of `ProposalArtifact` objects, output TSV filename; optional `priority_limit=40`, `limits` | `Path` to the TSV |
| [`reconcile`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/reconcile.py) | List of `ProposalArtifact` objects, list of externally approved proposal IDs, `output`; optional `protected_bins`, `time_limit=60.0`, `limits` | `SelectionArtifact` |
| [`export`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/export.py) | `SelectionArtifact` or `ProposalArtifact`, `output`; proposal exports require keyword `proposal_ids`; optional `complete_directory`, `limits` | `Path` to the export manifest |
| [`run_experiment`](https://github.com/Ludvins/BinComplete/blob/main/src/bincomplete/experiments/runner.py) | JSON/TOML configuration path; optional `output` override, `progress` callback receiving stage dictionaries | `ExperimentArtifact` |

For example, reopen prepared data and a fitted model with
`Dataset(Path("prepared"))` and `ModelArtifact(Path("model"))`. A saved model
can propose into another compatible prepared population; the feature and encoder
contract must pass unchanged. The `recipients` argument to `propose` is a list
of bin IDs; the `recipients` argument to `run` is a prepared dataset. They are
different types despite sharing that parameter name.

`ResourceLimits(threads=1, memory_gib=4, cache_gib=10)` explicitly requests one
numerical thread. Its defaults and buffer budgets are documented in
[resource boundaries](architecture.md). Limits are not an OS memory reservation.
The standard experiment obtains resources from its configuration, rather than
from a `limits` argument to `run_experiment`.

## Method settings in Python

For `fit` and `run`, `settings` is a Python `dict[str, Any]` or `None`. A JSON/TOML
filename is not accepted by these Python arguments: parse it first. CLI
`--settings` and experiment method entries handle file paths separately.
`None` selects the lower-level profile defaults, including legacy proposal
search; it does not automatically load the standard experiment recipe.

To start from a packaged standard recipe and change one parameter:

```python
import json
from importlib.resources import files
from pathlib import Path
from bincomplete import Dataset, ResourceLimits, fit

settings = json.loads(
    files("bincomplete")
    .joinpath("profiles/standard_v1/logistic_full_pool.json")
    .read_text(encoding="utf-8")
)
settings["classifier"]["params"]["C"] = 0.5
model = fit(
    Dataset(Path("prepared")),
    "models/logistic_variant",
    method="logistic_full_pool",
    settings=settings,
    limits=ResourceLimits(threads=1, memory_gib=4, cache_gib=10),
)
```

The prepared dataset must already contain compatible inputs. The example changes
the recipe and uses a new model output directory; it is not a recommended tuning
choice or a quality claim. For an external settings file, use
`json.loads(Path("settings.json").read_text(encoding="utf-8"))`, or read TOML
using `tomllib.load` on a binary stream, then pass the resulting dictionary.

Settings merge rules are deliberately limited:

- `classifier`, `scorer`, `masker`, `negative_sampler` and `retriever` accept
  a `params` dictionary. Its entries update that component's profile parameters;
  other parameters remain unchanged. These overrides do not select arbitrary
  implementations.
- `search` accepts `name` and `params`. Keeping the profile's search component
  updates its parameters. Selecting a different supported search component
  replaces the search block with that name and the supplied parameters.
- `seed` replaces the seed. `preprocessing` replaces that whole block rather
  than recursively merging it; supply the intended complete preprocessing policy.
- `training` and `validation` update their permitted top-level entries.
  In particular, a supplied `validation.masker` replaces that whole entry.
  Unsupported keys are rejected rather than silently ignored.

See [supported overrides](configuration.md#supported-method-overrides) and
[numerical contracts](experiments.md#explicit-numerical-input-contracts).
The `cache` argument to low-level operations is a string/`Path` directory or
`None`, which uses normal cache discovery. It is independent of the model's
artifact output directory.

`reconcile(..., protected_bins=["bin_A"])` leaves those original recipient bins
unchanged by excluding their proposed replacements from selection. It does not
redefine which contigs belong to a bin. Protection of all original assigned
members from recruitment as donors already applies independently.

## Reuse, errors and external approvals

An identical completed artifact can be verified and reused. Changing inputs,
code, dependencies or settings requires a new output directory. The experiment
runner preserves failed attempts and allocates a fresh attempt when resumed;
do not treat the existence of an output directory as completion. The report
lists the actual successful attempt path. See [resumption](experiments.md).

Malformed or incompatible inputs normally raise `ValueError`; file, resource
and stage failures can raise other standard exceptions. Scripts should retain
the exception and stage logs, rather than silently retry with changed settings.
The CLI reports setup errors and exits unsuccessfully; use `--debug` for a traceback.

`candidate_table` includes mean/minimum added-contig scores and addition-size-balanced
priority ranks by default. Its optional `priority_limit=40` marks the first
assessment tier; every remaining proposal is retained in tier 2. Ranks apply
within one method artifact and recipient. See the [assessment order](evaluation.md#score-based-assessment-order)
for fields, formulas and tie-breaking. Changing a tier boundary requires a new
inventory output path and never changes the original proposal artifact.

No function above launches genome-quality tools or infers HQ from a score.
`approved_ids` must come from an external decision with exact full-membership
and sequence joins. Reconciliation maximizes compatible approved recipients;
it does not rank approvals by CheckM2 or RNA metrics. See [evaluation](evaluation.md).

## CLI discovery

Run `bincomplete --help` for the command list and
`bincomplete COMMAND --help` for that command's exact arguments.

| Commands | Purpose |
| --- | --- |
| `config init`, `config show` | Create or inspect installation/project defaults |
| `models install`, `models inspect` | Explicit checkpoint download, or read-only local space inspection |
| `doctor`, `methods`, `demo` | Environment receipt, six-method catalogue, synthetic installation walkthrough |
| `init`, `preflight`, `experiment` | Initialize, check and run a configuration-driven experiment |
| `prepare`, `fit`, `propose`, `run` | Explicit lower-level artifact stages |
| `candidates`, `reconcile`, `export` | Review inventory, externally approved compatible selection and FASTAs |

`python -m bincomplete` exposes the same commands. Installation-time model
downloads are explicit; ordinary experiment commands do not fetch weights.
