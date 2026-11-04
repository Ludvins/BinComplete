# Trust boundaries and safe sharing

Use saved models only from a source you trust. BinComplete loads Python model
objects with joblib/PyTorch; such files can execute code when loaded. Matching a
SHA256 proves identity against a receipt, not that the author is trustworthy.
Do not load a downloaded model merely to inspect an unknown manifest. See
[joblib's loading warning](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html).

The official DNABERT-S encoder includes custom Python code. Explicit model
installation uses a pinned upstream revision and records model assets; running
an encoder executes that code. Review the upstream source and licence, and use
an isolated environment. Locally supplied encoder directories are also trusted
code. Run commands do not download models automatically.

## Dependency advisories and reproducibility pins

The tested CPU profile pins PyTorch 2.13.0 and setuptools 84.2.1 in
`requirements.txt`. PyTorch 2.13.0 includes the
[TorchScript memory-corruption fix](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
and accepts the updated build tool. Setuptools 84.2.1 includes the fixes for
[package-index path traversal](https://github.com/pypa/setuptools/security/advisories/GHSA-5rjg-fvgr-3xxf),
[Unicode source-manifest exclusions](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c)
and [legacy Windows wheel-installation path escape](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w).
Package metadata requires PyTorch 2.13.0 or newer and setuptools 84.1.0 or newer
for neural extras. Source builds also require setuptools 84.1.0 or newer.

These minimums address the listed advisories; they are not a guarantee that a
dependency version has no other vulnerabilities. Review current upstream
advisories when updating an environment and run `python -m pip check` after
installation. The CPU installation guide and CI read the exact tested Torch
version from the same constraints file.

Keep historical results with their original frozen environment. A runtime
upgrade needs encoder, numerical and saved-artifact validation in a new
environment and new output directories. Passing synthetic and artificial-DNA
smokes establishes software compatibility within that scope; it does not prove
identical metagenome proposals or biological quality gains. See the
[validation scope](validation.md).

## Input and output sharing

Assembly, membership and coverage inputs are validated as data. Keep large data,
model weights and caches outside the source checkout. Configure paths instead of
copying them into a public repository. Keep completed scientific artifacts
separate from evictable caches and preserve their referenced inputs.

Manifests, preflight reports and candidate tables include paths and identifiers.
Review them for sensitive sample information before sharing. Do not place tokens
or passwords in recipe files. BinComplete provides no clinical interpretation or
human-sequence removal step; apply your own data-access and biosafety policies.

Send private security reports to [Luis A. Ortega](mailto:laoa@cs.aau.dk).
See [support information](https://github.com/Ludvins/BinComplete/blob/main/SUPPORT.md) for the source repository and the
information to include in a reproducible report.
