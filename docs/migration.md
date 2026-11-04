# Version changes and artifact compatibility

The package, CLI and Python namespace are `bincomplete`. Install from a source
checkout or a built wheel, then use `bincomplete --help` or `python -m bincomplete`.
The current beta includes six profiles and portable shared configuration.

## Method names

Run `bincomplete methods` to list supported profiles and their display names.
New commands may use descriptive aliases without changing the stored recipe:

| Preferred input name | Stored profile ID |
| --- | --- |
| `hgb_composition_shortlist` | `kmers` |
| `hgb_embedding_shortlist` | `dnabert_s` |
| `attention_query` | `single` |
| `hgb_full_pool` | `hgb_full_pool` |
| `logistic_full_pool` | `logistic_full_pool` |
| `cosine` | `cosine` |

The CLI and Python API accept the aliases. Global per-method configuration
tables, packaged recipe filenames and the recipe's `method` field use the stored
**profile ID**. An alias selects the same profile and does not add another model
family. Numerical settings and seeds are unchanged.

An experiment also has a **configured method ID** identifying one instance of a
profile. With no explicit `id`, it defaults to the stored profile ID. A custom
`id` names that instance's summary key and artifact directory; the fitted recipe
still records the underlying profile. For example:

```json
{"method":"hgb_full_pool","id":"hgb_more_rounds","settings":{"classifier":{"params":{"max_iter":200}}}}
```

This instance uses `methods/hgb_more_rounds/fit/0001/` and
`methods/hgb_more_rounds/propose/0001/` for its first attempts. It remains an
instance of `hgb_full_pool`. Use the actual attempt paths from the run report;
keep the profile/instance distinction when interpreting existing artifacts.
Input search settings accept `max_proposals`; `max_checks` remains its equivalent
stored compatibility name. Do not supply both names in the same parameter table.

## Component names

Current feature implementations live in `features/membership.py`. Use the
registered names `composition_distances`, `coverage_agreement` and
`embedding_similarities`. Dataset-specific aliases and old research configurations
are not part of the supported interface. Preserve the original installation and
configuration for projects that used an earlier interface. See [features](features.md).

Unused internal registrations and feature-column descriptors have been removed.
The maintained feature blocks and six method recipes remain the supported portfolio.
Direct internal registry calls are not a substitute for the documented public
profile interface; preserve an older installation for earlier custom components.

## Saved artifacts

Models, datasets and proposals are immutable. Updates do not rewrite their hashes
or fitted parameters. Resume requires the same code, dependencies and settings;
use a new output directory for changed conditions. Use the original frozen package
for exact replay of an older artifact. Strict numerical and encoder compatibility
guards remain active; never edit a manifest to make incompatible sources match.

Use the documented CLI and Python API for the installed package. Earlier command
interfaces and serialized class definitions require their original installation.
Load only trusted saved models; see [trust boundaries](security.md).

Current configuration examples are under `configs/`. Keep project-specific recipes,
completed outputs, historical source snapshots and evaluation evidence outside
the package repository. Upgrading does not resume those projects automatically.
