# Prepare a public release

This guide separates the software repository from local installations, datasets
and run evidence. It does not publish anything automatically.

## Files to include

| Path | Purpose |
| --- | --- |
| `src/` | Maintained BinComplete package |
| `docs/` | Website source, user guides, formulas, source-based diagrams and page assets |
| `configs/` | Portable examples with placeholder paths |
| `tests/` | Small synthetic tests; no private datasets or fitted models |
| `tools/` | Conversion, package checks and synthetic installation smokes |
| `.github/workflows/` | Wheel, source-distribution, documentation and optional encoder CI |
| Root metadata and guides | `README.md`, `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SUPPORT.md`, `SECURITY.md`, `THIRD_PARTY.md` |
| Build and installation files | `pyproject.toml`, `requirements.txt`, `MANIFEST.in`, `run.py`, `.gitignore`, `requirements-docs.txt`, `zensical.toml` |

Keep local maintenance instructions and checkout verification receipts local.
Do not include virtual environments, caches, generated builds, run artifacts,
logs, private configurations, credentials, raw reads, assemblies, coverage
tables, pretrained weights or fitted models. Project-specific results belong
with the external project that produced them. Empty development run directories
are not needed in a release.

Retain licence and attribution notices for dependencies and external models.
Review binary assets and their metadata as well as source text; graphics should
be appropriate for public distribution. The README uses the workflow illustration;
detailed workflow and model diagrams use Mermaid source alongside the prose.

## Local validation

From an activated development environment at the source root:

```shell
python -m pip install -c requirements.txt ".[dev,sets,embeddings]"
python -m pytest -q
python -m ruff check src tests tools run.py
python -m ruff format --check src tests tools run.py
python -m mypy
python tools/check_release.py --root .
```

Build both distributions in a new build-output directory, then inspect the
source archive with `tools/check_distribution.py --root . --sdist PATH`.
The checkout's maintainer verification receipt records the actual completed
checks and their scope. Only a maintained checkout with that receipt can use
`tools/check_release.py --root . --public`; a clone or extracted source archive
uses the source-only command above. A receipt is not bundled in distributions.

These checks validate software and the stated compatibility scope. Keep a
numerical-regression record for changes that could affect model behavior.
Software checks do not establish gains on another metagenome; see
[validation](validation.md).

### Create or refresh the checkout signoff

A new clone does not need a verification receipt to install, run tests, build
distributions or pass source-only CI. `--public` is an additional maintainer
attestation for the particular release checkout; it does not generate evidence
or run the checks named in a receipt.

To issue that attestation:

1. Save completed check logs and environment records outside the source tree.
   Record what ran and on which platform. Never mark a configured or pending
   hosted job as passed. Inherited evidence needs a stated scope and verified
   identity of the implementation to which it applies.
2. Create an external JSON record with a `checks` object containing every name
   listed in `REQUIRED_RELEASE_CHECKS` in `tools/check_release.py`. Each entry
   needs `status` and `scope`, plus `evidence_files` mapping actual log/report
   paths to their SHA256 values. Use `PENDING` until its evidence has been
   reviewed; receipt creation must reject incomplete checks. For example, the
   shape of one entry is:

   ```json
   {"checks":{"clean_windows":{"status":"PENDING","scope":"Record the actual tested environment and commands","evidence_files":{}}}}
   ```

3. After all required checks have genuinely passed, create the local receipt
   from that external record using the code below. Replace the evidence path;
   do not substitute a checklist of intended tests for their results.
4. Run `python tools/check_release.py --root . --public`. Keep the external
   evidence and receipt with the release record, outside the public file set.
   Any later source change invalidates that signoff: rerun affected checks and
   regenerate it. Preserve prior release records rather than overwriting their
   evidence.

The following Python can be saved outside the checkout and run with the checkout
as the current directory. It hashes evidence files, checks the required statuses
and fingerprints the same local file population as the signoff checker. It
records the maintainer's reviewed evidence; it cannot determine whether a test
report is scientifically or operationally sufficient.

```python
import hashlib
import json
from pathlib import Path
import runpy
import tomllib

root = Path.cwd()
gate = runpy.run_path(str(root / "tools/check_release.py"))
evidence = json.loads(Path("/path/to/external/release-checks.json").read_text())
records = evidence["checks"]
for name in gate["REQUIRED_RELEASE_CHECKS"]:
    record = records[name]
    assert record["status"] == "PASSED", name
    assert record["scope"] and record["evidence_files"], name
    for filename, expected in record["evidence_files"].items():
        with Path(filename).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected

ignored = {
    ".git", "__pycache__", ".ruff_cache", ".mypy_cache", ".pytest_cache",
    "build", "dist", "runs", "site", ".cache", ".venv", ".venv-docs",
}
sources = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in root.rglob("*")
    if path.is_file()
    and not any(p in ignored or p.endswith(".egg-info") for p in path.relative_to(root).parts)
    and path.suffix not in {".pyc", ".partial", ".lock"}
    and path.name != "release_verification.json"
}
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
receipt = {
    "status": "VERIFIED", "version": version,
    "checks": {name: records[name]["status"] for name in gate["REQUIRED_RELEASE_CHECKS"]},
    "check_scopes": {name: records[name]["scope"] for name in gate["REQUIRED_RELEASE_CHECKS"]},
    "source_sha256": sources,
}
target = root / "release_verification.json"
if target.exists():
    raise FileExistsError("Preserve the previous receipt outside the checkout first")
target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
```

## Inspect the proposed Git contents

Before staging files, check whether the directory is already the intended
Git checkout. For an existing checkout:

```shell
git status --short --untracked-files=all
git ls-files --others --exclude-standard
```

Review the file set against the inclusion table. When explicitly authorized to
prepare a commit, stage those named software paths, then inspect the complete
staged diff and `git diff --cached --check`. Do not use an indiscriminate add
command in a directory containing research outputs. Ignore rules do not remove
files that were already tracked; review `git ls-files` as well.

Before an authorized first push, initialize Git only if this is not already a
checkout, confirm the remote against the repository in `pyproject.toml`, and
review authentication separately. Creating a repository, committing and pushing
are explicit maintainer actions, not part of tests or release checks.

## Hosted checks and release scope

The configured workflow exercises installed wheels and source distributions
on Windows and Ubuntu with Python 3.11/3.12. Local passing checks do not prove
that hosted jobs passed. Confirm the actual workflow results after the first
push before calling the release's hosted CI successful.

The optional real-encoder job downloads pinned external weights only when
explicitly selected. Ordinary CI and package checks do not run genome-quality
tools. Publish the package's beta status and documented scientific limitations;
do not turn a saved dataset result into a universal performance guarantee.

## Documentation checks

Use the isolated [documentation build workflow](documentation.md) to run
`zensical build --clean --strict`. Keep generated `site/` and `.cache/` directories
and the documentation virtual environment outside the public source file set.
Documentation CI validates pull requests; deployment also requires the explicit
publication setting described in that guide. A local preview is not a published
site, and prepared automation is not evidence of a successful hosted build.
