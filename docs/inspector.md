---
template: inspector.html
---

# BinCompleteInspector

Open local reports to inspect bin refinement results and RNA evidence.

1. [Export a JSON report](reporting.md#generate-the-report) from a completed
   selection, or start with **Try synthetic report** in the viewer.
2. Open the Inspector under `/inspector/` in the
   [local documentation preview](documentation.md#preview-locally) or on
   [GitHub Pages](https://ludvins.github.io/BinComplete/inspector/) after publication.
3. Select **Open reports** to choose one or more JSON files, or drop them onto
   the welcome screen. When you select several files, the Inspector combines
   their bins and proposals into one report. The overview, graphs, lists, and
   method outcomes use the combined records.
4. Open **Bins** to inspect a bin, compare evaluated proposals, and view its saved
   RNA evidence.

The browser reads the JSON files locally. The optional sequence viewer also
reads its matching FASTA locally. See
[added contig evidence](reporting.md#include-added-contig-evidence) for the
extra export fields needed for GC, coverage and RNA feature coordinates.

## Combine reports

Choose valid BinComplete report JSON files that use the same HQ policy. The
Inspector combines matching bins and proposals from the same dataset and
counts repeated records once. Repeated IDs from different datasets get a
dataset number that you can look up in **Provenance**.

If a file is invalid or records disagree on a bin's measurements or selection,
the Inspector reports the conflict and keeps the current view. **Provenance**
shows settings and receipts for each source file. **Methods → Method
comparison** shows the saved overlap, validation, and benchmark results for
each file.
