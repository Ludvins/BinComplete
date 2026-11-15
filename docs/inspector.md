---
template: inspector.html
---

# BinCompleteInspector

Open a local report to inspect bin refinement results and RNA evidence.

1. [Export a JSON report](reporting.md#generate-the-report) from a completed
   selection, or start with **Try synthetic report** in the viewer.
2. Open the Inspector under `/inspector/` in the
   [local documentation preview](documentation.md#preview-locally) or on
   [GitHub Pages](https://ludvins.github.io/BinComplete/inspector/) after publication.
3. Select **Open report** to read the JSON. Open **Bins** to inspect a bin,
   compare evaluated proposals, and view its saved RNA evidence.

The browser reads the JSON locally. The optional sequence viewer also reads
its matching FASTA locally. See
[added contig evidence](reporting.md#include-added-contig-evidence) for the
extra export fields needed for GC, coverage and RNA feature coordinates.
