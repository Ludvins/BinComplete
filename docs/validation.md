# Validation status and interpretation

BinComplete is research beta software. It proposes alternative memberships for
external assessment. A completed run or high membership score does not establish
that an added contig belongs biologically in a bin.

## Software checks and scientific evidence

The release verification receipt records installed-package tests, dependency
checks, source equivalence and saved-output regression checks. These validate
software behavior within their recorded scope. They do not establish biological
accuracy, a dataset-independent quality gain or fresh encoding reproducibility.

Scientific runs need their own resolved configuration, source/input hashes,
numerical environment and external assessment evidence. Keep these with the
project, separately from the software distribution. Older fitted artifacts retain
their strict source guard; see [migration](migration.md).

## Software support and numerical portability

CI is configured for installed wheels on Windows and Ubuntu with Python 3.11 and
3.12. A configured job is not a completed hosted CI run. The release verification
receipt records which clean local environments actually passed. CPU encoder
smokes use ten artificial contigs and reduced toy fits; they establish dependency
and workflow viability, not metagenomic performance. The public demo uses supplied
synthetic vectors and needs no checkpoint or quality tools.

| Environment | Direct evidence for this beta |
| --- | --- |
| Windows, Python 3.12.10, PyTorch 2.13.0+cpu and setuptools 84.2.1 | Isolated dependency install and ten-contig real encoder/six-method smoke passed. |
| Ubuntu 24.04 under WSL2, Python 3.12.3, PyTorch 2.13.0+cpu and setuptools 84.2.1 | Separate isolated Linux install and the same real encoder/six-method smoke passed. |
| Python 3.11 on hosted Windows/Linux | CI configured; hosted execution not established by local checks. |
| CUDA | Numerical identities are recorded; this release does not establish new whole-assembly CUDA reproducibility. |
| macOS/Apple accelerator | Not tested for this beta. |

Installation smokes use ten artificial contigs and six reduced toy fits. Their
runtime is not a metagenome runtime estimate. The smoke declares a 4 GiB allocation
budget and 1 GiB derived-cache ceiling; these are not measured RSS or OS-enforced
limits. The official checkpoint is about 468 MB; keep it external and reusable.
Clean dependency installations also need several GB of disk. Full metagenome
cost depends on assembly size, source masks, candidate pools, encoder windows and
hardware; preflight does not provide a calibrated completion-time forecast.

The pinned Python 3.12 dependencies are in `requirements.txt`. Generic version
ranges in package metadata are not a numerical reproducibility guarantee.
Record the resolved recipe, source hashes, encoder revision, dependency versions,
arithmetic precision, device identity, batch layout and supplied-vector identity.
The dependency update changes the tested numerical environment. Existing artifacts
retain their frozen environment; use a new environment and output directory for
new fits. The installation smokes above do not establish equality between the
old and new runtimes on a metagenome.
GPU architecture and numerical buffers can change training and ranking even with
the same seed. Identical proposals or quality gains are not promised on a fresh
encoding or another dataset. CPU and GPU runs are distinct numerical environments.

## Scientific limits

Same-dataset training and refinement are disclosed as transductive. Related-group
separation during masking/selection does not make the entire metagenome an
independent test. A changed training policy, search budget or method set needs
its own validation. Statistics from another policy do not automatically apply.

Performance across independent metagenomes remains unestablished. Do not treat
results from one dataset as a general improvement rate or a biological default
proven for every binner. Use the prospective plan below to test those claims.

Original bin assignments are training labels, not independently established
biological truth. Contamination or misassigned contigs in those bins can enter
the positive labels, and an imperfect relatedness grouping can affect negative
labels. Separating related groups reduces a declared source of overlap; it does
not verify the original bins or establish that every cross-group negative is
biologically unrelated.

## Counting quality changes

HQ means satisfying the study's explicitly declared external quality policy;
BinComplete does not define a universal HQ rule. Freeze the tools, versions,
settings and acceptance criteria before inspecting proposal outcomes. Apply the
same policy to original bins and full proposed constructions.

After external assessment and reconciliation, compare each original recipient
with its final selected construction. A recipient with no selected construction
retains its original bin. Count recipients once, rather than counting alternatives
or methods that proposed the same construction:

| Original status | Final status | Count |
| --- | --- | --- |
| Fails the declared HQ policy | Passes | One HQ gain |
| Passes | Fails | One HQ loss |
| Passes | Passes | Retained HQ status; any metric changes are separate results |
| Fails | Fails | No HQ gain or loss; other metric changes may still matter |
| Either status is unknown | Any | Unassessed pair; do not treat it as a failure or a zero gain |

Report original and final HQ totals, gains, losses and their net change on the
same assessable recipient set. Declare the full recipient population and give
both its total size and the number of pairs with assessable original/final
statuses. Use the stated assessable-pair denominator
for paired rates, and report unassessed pairs separately so missing assessments
do not disappear from the result. Report approval counts and compatible-selection
counts separately from quality changes. For example, two approved alternatives
for one previously non-HQ recipient can contribute at most one compatible HQ gain.

Quality-status gains do not by themselves prove correct recruitment of every
added contig. Where evidence permits, report misrecruitment, chimerism and other
biological errors alongside the chosen genome-quality criteria. The
[evaluation walkthrough](evaluation.md) explains identity matching and export.

## Prospective external benchmark

Before inspecting outcomes, freeze the BinComplete version, six profiles, search
budget, device/encoder policy, quality-tool versions and acceptance thresholds.
Register independently obtained datasets spanning community types, binners,
assembly quality and coverage distributions. Specify exclusions and all metrics
in advance; retain failed or negative cases.

Use original memberships for training; never train on HQ repairs or assessment
labels. Keep related genomes grouped for development and statistical evaluation.
Report same-dataset transductive refinement separately from cross-dataset model
transfer. Assess original bins and proposed full constructions with the same
external tools and criteria, then reconcile mutually compatible selections.
Where reference genomes are available, measure contamination, chimerism and
misrecruitment as well as completeness and RNA support.

Report compatible HQ gains and losses using the counting rules above, uncertainty
by independent dataset/group, unassessed proposals, runtime, peak memory and
storage. Compare the original
binner, composition-only and six-method settings under declared proposal budgets.
Do not select seeds or alter thresholds after seeing test outcomes. Archive
resolved recipes, all proposals/scores, logs and raw assessment outputs so the
result can be independently checked. No such new benchmark was run for this
software release.
