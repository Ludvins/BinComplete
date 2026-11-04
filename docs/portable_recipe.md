# Standard six-method recipe

For shared paths, method hyperparameters and compact dataset configurations,
start with [global configuration](global_configuration.md). The materialized
standalone recipe below remains supported.

The standard experiment is `standard_v1`: cosine, composition HGB, DNABERT-S
HGB, single-query attention, full-pool HGB and full-pool logistic regression.
Each profile has explicit preprocessing, sampling, classifier and search settings.

## Install and run

Use Python 3.12 for the tested dependency profile. From the source directory,
create and activate a virtual environment, then install:

```shell
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -c requirements.txt ".[sets,embeddings]"
python -m bincomplete init --output my_experiment --device cuda:0
python -m bincomplete doctor --device cuda:0 --output my_experiment/environment.json
# Edit my_experiment/experiment.json to point to your inputs.
python -m bincomplete preflight --config my_experiment/experiment.json
python -m bincomplete experiment --config my_experiment/experiment.json
```

The default initializer uses supplied embeddings. It does not copy or regenerate
them. Use `--embedding-mode encode` to request frozen local DNABERT-S inference
instead, and set `encoder_path` to your local weights. Run commands never download
weights automatically; the explicit `models install dnabert_s` command is available.
Install the appropriate Torch wheel for your hardware using
the [official installation selector](https://pytorch.org/get-started/locally/).
The constraints pin the public Torch version; they do not select a CUDA wheel or
GPU driver. CUDA encoder defaults use BF16, batches of eight and float16 storage.
`doctor` records the actual build, GPU identity and arithmetic backend. Preserve
that receipt with the experiment; hardware and precision affect numerical results.

For CPU use, pass `--device cpu` to both commands. CPU encoding uses FP32; attention
also uses CPU. This is a supported execution variant, not a claim of bitwise
equivalence to CUDA execution. Unavailable devices fail explicitly. Device and
precision changes produce distinct cache/model identities.

All generated paths are relative to the configuration file, so invoking the same
configuration from another working directory works. You may replace them with
external absolute paths instead of moving large data. The generated directory
contains the complete editable settings and can be transferred to another host.
Install the package on that host and update paths/devices as necessary.

## Inputs and dataset changes

Supply assembly FASTA, original memberships (`contig_id`, `bin_id`), measured
coverage (`contig_id`, `mean`, `variance`) and compatible encoder inputs. Related-bin
groups (`bin_id`, `group_id`) and a recipient list (one bin ID per line) are optional
inputs. This materialized initializer includes paths to `data/groups.tsv` and
`data/recipients.txt`: either supply those files or remove the dataset's `groups`
key and the top-level `recipients` key to use their omission defaults. Omitting
groups assumes each bin is unrelated; omitting recipients targets all supplied bins.
Those defaults do not establish biological independence or select incomplete bins.

TSV columns use tabs. A supplied group table must map every bin exactly once.
Use biologically meaningful groups and record the criterion used; do not put
related bins in different groups to increase the apparent sample size. The
[input examples](inputs.md#related-bin-groups) explain group counts and minimum
training requirements. Original memberships are proxy training labels, not
previously accepted HQ constructions or established biological truth. Existing
misassignments can affect training.

Supplied embeddings need a `.npy` array, row IDs and the real encoder-space JSON;
see [input contracts](inputs.md). Do not invent an identity for arbitrary vectors.
You may supply a composition table using `composition`. Omitting it computes
composition from FASTA and can produce a different numerical input condition. Composition uses the explicit ATGC basis, pseudocount `1e-5`,
float32 renormalization and float32 coverage in the standard settings.

For a second dataset, initialize a **new** directory and change only its inputs.
Fit fresh models from that dataset's original memberships, or declare separate
`training` and `evaluation` datasets in the configuration. The default uses the
same memberships for training and evaluation and records transductive overlap.
Supply training `source_ids` if stable source identities differ from bin names.
Use `donor_ids` to restrict eligible candidates; all original bin contigs remain
protected. Do not merge evaluation cohorts with different protected memberships
or donor pools and expect identical proposals. To reuse a model across contexts,
prepare each context and call `propose` separately with the saved model and its
recipient list; the strict feature/encoder compatibility guard still applies.

The experiment command runs methods sequentially, sharing preparation and caches.
Run the same command again to verify and reuse completed stages. Changed inputs,
source, dependencies or settings require a new results directory. Allocation
limits are two threads, 6 GiB working buffers and a 10 GiB reusable cache; they
are not an OS-level process memory cap. Checkpoint/model outputs are preserved
outside cache eviction.

## Search and compatibility

Each method uses the same addition-only search: retain inherited 20 alternatives,
singletons through rank 50, and one-donor substitutions within width two around
prefixes 10 and 20, capped at 128 distinct proposals per bin. Added sequence length
cannot exceed the original bin length. Proposal counts depend on eligible donors
and deduplication; 128 is a cap, not an acceptance threshold.

The standard experiment settings are installed inside the wheel. The lower-level
API without explicit settings retains the older 20-proposal profile to preserve
existing scripts. `configs/fresh` now contains the standard settings; the portable
`init` command is the recommended workflow. For an explicit legacy search, use:

```json
{"search":{"name":"ranked_alternatives","params":{"max_proposals":20,"max_added_fraction":1.0,"fallback_singletons":50}}}
```

Fitted model recipes are immutable. Upgrading the package does not relabel or
change old models. Feature, classifier, sampling, encoder and scoring source
formulas are unchanged by this release. Search selection is saved in model and
proposal provenance.

`max_proposals` is the preferred input spelling; saved settings retain the
equivalent `max_checks` key. Descriptive method aliases also resolve to the same
stored profile IDs. See [naming and compatibility](migration.md#method-names).

## Assessing results

The recipe generates alternative memberships, not quality verdicts. Compare
original bins and full proposed constructions with the same external quality
tools and thresholds. Reconcile mutually compatible approved constructions
before counting gains. Unknown proposals remain unassessed.

The default report is `my_experiment/results/report.md`, because `results` is
relative to `my_experiment/experiment.json`. For the assessment commands in the
[evaluation walkthrough](evaluation.md), first enter `my_experiment`; its relative
paths then start at the correct project directory. Count gains against each
original bin's assessed status, rather than counting approved alternatives; see
[quality-change counting](validation.md#counting-quality-changes).

Dataset-specific results belong with that project's configuration and evidence.
Performance across independent metagenomes is not established by software tests;
see [validation and benchmark guidance](validation.md).

## Environment and CI

`requirements.txt` constrains tested core/optional numerical packages
and development tools. It is not a complete cross-platform, hash-locked wheelhouse:
transitive packages and platform wheels still depend on the installer. Reference
direct dependency versions ship in
`profiles/standard_v1/tested_environment.json`; `doctor` reports version differences.
Each new experiment also records its environment alongside the frozen source/input
contract. Exact reproduction needs the same input bytes, composition table,
encoder/tokenizer, settings, dependency builds, device and arithmetic behavior.

GitHub Actions builds and installs a wheel, checks dependencies, runs tests and
executes the same two-dataset CLI smoke on Windows/Linux with Python 3.11/3.12 and
CPU attention. CI never runs scientific quality tools. These jobs run after the
repository is pushed; authored CI is not evidence that hosted Linux jobs passed.
