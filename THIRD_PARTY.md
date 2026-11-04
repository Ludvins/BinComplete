# Third-party software and model assets

The MIT licence in this repository covers BinComplete. Dependencies, pretrained
models and input data retain their own licences. Installing BinComplete does not
change those terms.

DNABERT-S is distributed separately by its authors under Apache 2.0. BinComplete
references local weights or downloads an explicitly requested, pinned official
checkpoint; weights are not bundled in the Python wheel. Retain the upstream
licence and citation with redistributed model assets. See the
[official model card](https://huggingface.co/zhihan1996/DNABERT-S).

Core dependencies include NumPy, SciPy, scikit-learn, joblib and threadpoolctl.
Optional method dependencies include PyTorch, Transformers and einops. Consult
their installed distributions for licence notices. A release environment can be
recorded with `python -m pip freeze` and `bincomplete doctor --device cpu`.

The optional documentation build uses [Zensical](https://zensical.org/) and its
declared dependencies, installed separately from the scientific runtime. The
website uses [MathJax](https://www.mathjax.org/) for formulas and
[Mermaid](https://mermaid.js.org/) for diagrams. These browser libraries retain
their upstream licences and are loaded from their configured external sources;
their distributions are not bundled in the BinComplete wheel.
