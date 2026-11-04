# Glossary

These terms describe BinComplete's membership workflow. They do not assign
biological truth or genome-quality verdicts.

| Term | Meaning |
| --- | --- |
| Original bin | The contig membership supplied by the upstream analysis, before BinComplete additions. Its assignments are proxy training labels. |
| Source bin | An original bin supplying masked training examples; a multi-contig bin can provide both retained and hidden members. |
| Core / hidden members | In a training episode, the original members retained as input / temporarily held out as positive examples. This mask changes only the episode's inputs; it does not rewrite the original assignments. Proposal generation retains the recipient's complete original membership. |
| Episode | A training example formed by masking an original bin and choosing candidate positives and negatives. |
| Recipient | An original bin selected to receive proposed additions. |
| Candidate contig | A contig considered for membership in a particular core. Labelled training candidates are hidden original members (positives) or members of other assigned source bins in different declared groups within the current partition (negatives), subject to the sampling policy. Unassigned contigs have unknown training labels. |
| Donor | An unassigned assembly contig available as an inference candidate for recruitment, optionally restricted by `donor_ids`. Eligibility and ranking are applied per recipient. Original assigned members stay protected, including those temporarily hidden in training. Availability does not establish correct membership. |
| Related group | A user-declared set of biologically related source bins kept together for sampling and selection splits. Different group IDs are an assumption, not proof of independence. |
| Construction | One recipient's complete proposed contig membership: its original members plus specified additions. |
| Proposal | A saved, method-specific suggestion for a construction, identified by `proposal_id`. Several proposals may describe the same construction. |
| Compatible selection | Approved constructions chosen with at most one per recipient and no contig shared between selected full memberships. Compatibility is structural. |
| HQ | High quality under the study's declared external acceptance policy. BinComplete supplies no universal HQ threshold. |
| Transductive use | Training from a dataset's supplied memberships and applying the models back to that dataset. This does not measure generalization to an independent metagenome. |
| HGB | Histogram gradient boosting, used by several membership classifiers. |
| Embedding / encoder space | A contig vector / the declared encoder, tokenizer and production protocol defining those vectors. Equal vector dimensions do not establish compatibility. |
| Stable source ID (`source_ids`) | A user-provided bin identifier used to keep source-specific random sampling stable when bin names change. It is distinct from a related-group ID. |
| Implementation source identity | Recorded identity of the package code used to produce an artifact. It is distinct from training `source_ids`. |
| Receipt / artifact | A recorded identity and verification record / a saved prepared dataset, model, proposal set, selection or export. Completed artifacts and their receipts remain immutable. |

For read, contig, length, depth, variance, centroid, distance, similarity,
normalization, loss, recall and numerical settings, see
[understanding the quantities](quantities.md). Each explanation distinguishes
the measurement, its interpretation and its use in the workflow.

See [input contracts](inputs.md), [training](training_protocol.md),
[external assessment and export](evaluation.md) and
[quality-change counting](validation.md#counting-quality-changes).
