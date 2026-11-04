# Contributing

Discuss changes in [the repository](https://github.com/Ludvins/BinComplete) or contact
[Luis A. Ortega](mailto:laoa@cs.aau.dk). Contributions follow the MIT licence.
Do not submit restricted datasets, model weights or credentials.

Install the selected method dependencies and development tools:

```shell
python -m pip install -c requirements.txt ".[dev,sets,embeddings]"
python -m pytest -q
python -m ruff check src tests tools
python -m ruff format --check src tests tools
python -m mypy
bincomplete demo --output demo-run
```

Include a small test for a changed input contract, numerical operation or artifact
identity. Keep expensive scientific validation separate from the unit suite.
For a new method, document its training labels, related-group separation,
resource requirements, dependencies and reproducible settings. Quality scores
must not silently become training labels or model-selection criteria.

Change numerical profiles under a new explicit identity. Never overwrite frozen
sources, fitted models, completed results or historical evidence. A proposal
count is not a quality-gain count. State whether comparisons use fresh fitting,
fresh encoding, saved rankings or exact sequence-matched quality evidence.

The [release preparation guide](docs/releasing.md) lists public files, exclusions,
validation steps and staged-content review. Before a public release, run `python tools/check_release.py --root . --public`
from the maintained source checkout. This public signoff checks the declared
repository/contact and the current release's verification receipt. The receipt
describes that checkout, so the public signoff is not a check for an extracted
source archive. To check source structure from an extracted source distribution,
run `python tools/check_release.py --root .` without `--public`.
A local passing check is not evidence that hosted CI has run or that results
generalize to a new metagenome.

Keep campaign notes, dataset identifiers, expected outcomes and machine-specific
receipts outside the source repository and release archives. Supported examples
use placeholder paths and document input formats rather than private datasets.
Optional cross-version tests read `BINCOMPLETE_GOLDEN_ROOT`; override
`BINCOMPLETE_GOLDEN_FEATURE_MODULE` and `BINCOMPLETE_GOLDEN_SEARCH_MODULE` when
the preserved checkout uses different module paths. No old source is bundled.

For documentation changes, use the separate [website build and preview workflow](docs/documentation.md).
The strict documentation build checks page links and section anchors.
