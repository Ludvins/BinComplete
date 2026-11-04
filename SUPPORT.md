# Support and bug reports

Author and maintainer: **Luis A. Ortega**.

Institutional contact: [laoa@cs.aau.dk](mailto:laoa@cs.aau.dk).

Source repository: [Ludvins/BinComplete](https://github.com/Ludvins/BinComplete).
Use the repository for reproducible bug reports and feature discussions. Send
private security reports to the institutional contact instead of posting them
publicly.

For a reproducible report, include:

- `bincomplete --version`, operating system and Python version;
- `bincomplete doctor --device cpu` (or the actual device);
- the command, resolved experiment configuration and `preflight.json`;
- the exception and relevant stage log; use `--debug` for a traceback;
- the smallest synthetic input that shows the problem, if possible.

Save the readiness report before opening an issue:

```bash
bincomplete preflight --config experiment.toml > preflight.json
```

Preflight returns a nonzero exit status when inputs need attention; the JSON
report is still written by this redirection. It does not launch an experiment.

Inspect reports before sharing. Manifests and logs can contain local paths,
sample names, contig IDs and sequence-derived information. Share confidential
assemblies or human-associated data only through your approved channel.

Do not edit a completed artifact to make a retry succeed. Preserve the failed
attempt and use a new output directory after changing inputs, code or settings.
See [trust boundaries](docs/security.md) for untrusted saved models and custom
encoder code.
