# Security policy

Report suspected vulnerabilities privately to
[Luis A. Ortega](mailto:laoa@cs.aau.dk). Include the affected version, operating
system, relevant configuration and a minimal reproduction without credentials
or restricted sequence data. Do not post sensitive details in a public issue.

The current research beta is the maintained version. Older saved models should
remain with their original frozen environment for reproducibility; this is not
a promise of security updates for those environments.

Saved model files and encoder custom code require trust in their source.
See [trust boundaries and safe sharing](docs/security.md) before loading models
or sharing artifacts. For ordinary installation and usage questions, see
[support](SUPPORT.md).
