# BinComplete documentation

Start with [installation and a first run](from_scratch.md), then choose
[generic binner inputs](generic_workflow.md) or the [mmlong2 adapter](inputs.md).
Run `bincomplete demo --output demo` to test installation without weights/data.

| Guide | Purpose |
| --- | --- |
| [Shared configuration](global_configuration.md) | Paths, device, resources and six-method defaults |
| [Configuration reference](configuration.md) | Supported experiment schemas and overrides |
| [Understanding the quantities](quantities.md) | Reads, depth, sequence distances, model scores and settings explained with examples |
| [Glossary](glossary.md) | Core, source, recipient, construction and evidence terminology |
| [Python API and CLI index](api.md) | Public operations, argument types, return artifacts and command discovery |
| [Input contracts](inputs.md) | FASTA, membership, coverage, groups and encoder identities |
| [Model features and formulas](features.md) | All 16 statistics, model mapping, neural inputs and diagrams |
| [Training protocol](training_protocol.md) | Original-membership labels, grouping and overlap |
| [Experiments](experiments.md) | Artifacts, resumption and failures |
| [External evaluation](evaluation.md) | TSV inventory, quality assessment, approvals and export |
| [Validation and performance](validation.md) | Support matrix, evidence scope and benchmark protocol |
| [Architecture](architecture.md) | Resource/cache boundaries and provenance |
| [Migration](migration.md) | Immutable historical artifacts and version changes |
| [Documentation website](documentation.md) | Local preview, rendering checks and GitHub Pages publication |
| [Release preparation](releasing.md) | Public file selection, local checks and hosted release boundaries |
| [Trust and third-party assets](security.md) | Saved models and pinned encoder custom code |
| [Portable recipe](portable_recipe.md) | Materialized schema-1 alternative |

Dataset-specific studies and run evidence belong in external project directories.
This beta generates candidates; it does not establish HQ from scores.
