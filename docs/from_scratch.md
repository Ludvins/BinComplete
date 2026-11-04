# Install and run from original binner outputs

Use Python 3.12 for the tested dependency profile. Obtain the
[BinComplete source checkout](https://github.com/Ludvins/BinComplete) and enter its
root directory, containing `pyproject.toml` and `requirements.txt`.

Before a real six-method run, have:

- The complete plain assembly FASTA, including unbinned contigs, original bin
  FASTAs or memberships, and measured coverage mean **and variance for every
  assembly contig**. See the [coverage handoff](generic_workflow.md#coverage-handoff)
  if the upstream analysis retained only means.
- Local DNABERT-S weights, or compatible precomputed vectors with their row IDs
  and original encoder-space manifest. The explicit installer below can obtain weights.
- At least four unrelated multi-contig source groups for default attention epoch
  selection. HGB/logistic require at least two. Supply related-bin groups when
  known; treating each bin as unrelated does not establish independence.
- Writable environment, model, cache and output locations. Clean dependencies need
  several GB of disk; the official checkpoint is about 468 MB. Standard runs use
  two numerical threads, a 6 GiB allocation budget and a 10 GiB reusable-cache
  ceiling. These are not an OS memory cap or a total output-disk limit. See
  [resources](architecture.md) and the [tested support matrix](validation.md).

Input paths below are placeholders to replace with your files. Quote paths that
contain spaces. Commands keep the same working directory unless stated otherwise.
Create the environment from the source root:

```shell
python -m venv .venv
```

Activate `.venv/Scripts/Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux. Then:

For a CPU-only installation, install the official CPU Torch wheel first. This
avoids downloading a CUDA runtime that the CPU workflow does not use:

```shell
python -m pip install --upgrade pip
python -m pip install -c requirements.txt setuptools wheel
python -m pip install -c requirements.txt torch --index-url https://download.pytorch.org/whl/cpu
```

On a GPU host, install the Torch build matching your CUDA setup instead, then
install BinComplete with the same constraints. Use the PyTorch version pinned in
`requirements.txt`; CPU and CUDA builds are separate numerical environments:

```shell
python -m pip install -c requirements.txt ".[sets,embeddings]"
bincomplete demo --output demo
bincomplete doctor --device cpu
```

The demo uses artificial vectors and deliberately tiny fits. It verifies six
methods, output tables and resume behavior; it does not encode DNA or establish HQ.
Run the same demo command to verify and reuse its completed artifacts.

For the composition-and-coverage method alone, install
`python -m pip install -c requirements.txt .` and select
`--methods hgb_composition_shortlist`; Torch and pretrained weights are not
required for that subset. Measured coverage variance is still required. The
six-method demo does require Torch.

After installation, change to an existing external project directory for the
remaining commands, keeping run evidence outside the source checkout:

```shell
cd /path/to/project
```

## Configure the real encoder once

If weights are absent, create the user-wide CPU configuration and explicitly
install the pinned official checkpoint:

```shell
bincomplete config init --device cpu
bincomplete models install dnabert_s
bincomplete config show --resolved
```

For existing weights, add `--encoder-path /models/dnabert_s` to `config init` and
skip installation. The installer uses the configured `encoder_path`; it does not
rewrite the configuration or discover arbitrary checkpoint folders. Normal runs
never download weights automatically. See [model locations and discovery](global_configuration.md#optional-model-installation).

Use `cuda:0` only on a supported CUDA host with the appropriate Torch build, and
use that device for `doctor` as well. Do not overwrite an existing shared config;
inspect and edit it or follow the complete
[project-specific configuration flow](global_configuration.md#project-specific-configuration).

## One original mmlong2 output

```shell
bincomplete init --input-root /data/mmlong2/sampleA --output runs/sampleA
bincomplete preflight --config runs/sampleA/experiment.json
bincomplete experiment --config runs/sampleA/experiment.json
```

Keep the final assembly/bin outputs and full primary JGI table under the sample's
tmp/binning/mapping directory. Trimmed depth views and mean-only final summaries
do not replace measured variance. Explicit `--coverage`/`--coverage-profile` flags
resolve genuine ambiguity. Flat research exports require explicit input mapping.

## Another binner

```shell
bincomplete init --assembly assembly.fasta --bins bins --coverage depth.tsv --output runs/sampleA
bincomplete preflight --config runs/sampleA/experiment.json
bincomplete experiment --config runs/sampleA/experiment.json
```

Replace `--bins` with `--memberships memberships.tsv` when appropriate.
[Generic inputs](generic_workflow.md) covers the coverage handoff, groups,
candidate restrictions and supported formats. The complete assembly must include
unbinned candidates. All supplied original bin members are protected.

For either example, initialization creates `runs/sampleA/experiment.json` and
`runs/sampleA/README.txt`. The experiment creates results below that directory:

```text
runs/sampleA/
  experiment.json                 # Dataset declaration
  results/
    experiment.json               # Frozen resolved run recipe
    status.json
    report.md
    methods/<method>/propose/<attempt>/
```

Read `runs/sampleA/results/report.md` for the completed proposal artifact paths;
resumed attempts can have different numbers. Use those paths in the
[external assessment walkthrough](evaluation.md) to approve, reconcile and export
constructions.

The same unchanged experiment command verifies and resumes completed stages.
Use a new output for another dataset. A run fits from original memberships; the
same training/evaluation dataset is transductive use, not independent validation.
Proposals alone are not HQ gains. Preserve referenced input files with the run;
the output directory does not contain copies of the assembly, pretrained weights
or supplied vectors.

The [portable materialized recipe](portable_recipe.md) is an alternative for
projects that need explicit per-method settings and separate evaluation cohorts.
