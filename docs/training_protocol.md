# Membership-training protocol

Training uses the supplied **original bin assignments as proxy labels**. A hidden
member of a source bin is treated as positive; a contig from an unrelated source
group is treated as negative. Those assignments can contain contamination or
misassignments. Group separation prevents declared related sources from crossing
partitions; it does not establish that the labels are biologically correct.
Previously accepted repairs, CheckM2/RNA results and HQ status never enter the
labels, feature values, negative selection or ranking scores.

The [feature reference](features.md) defines every scoring input and its formulas.
The [glossary](glossary.md) distinguishes a source, group, episode and recipient.

For plain-language meanings of a label, training weight, loss, epoch,
temperature and recall, see [understanding the quantities](quantities.md#labels-scores-and-training-metrics).

## Sources, masks and candidates

Each multi-contig source supplies **episodes**: one retained nonempty core,
hidden original members and candidate contigs with membership labels. All masks
come from that source alone. A source with fewer than two contigs, or without
negative sources in other groups of the current partition, cannot supply these
supervised episodes. Unassigned assembly contigs have unknown labels and are
not presumed negative during training.

| Standard profiles | Training masks per source with $m$ contigs | Positive candidates | Negative selection |
| --- | --- | --- | --- |
| `kmers`, `dnabert_s` | $m$ leave-one-out episodes, then 20 length-fraction episodes; duplicate masks retained | Hidden members that survive the composition shortlist | Up to 64 hard composition-shortlist negatives plus 32 random eligible negatives |
| `hgb_full_pool`, `logistic_full_pool` | $m$ leave-one-out episodes, then 20 count-based episodes; no absent episodes | All hidden members | Up to 64 embedding-hard negatives plus 32 random remaining negatives, without inference eligibility/shortlist filtering |
| `single` | $m$ leave-one-out episodes | The one hidden member | Same embedding-hard/random rule as full-pool classifiers |
| `cosine` | No supervised membership fit | Not applicable | Not applicable |

Leave-one-out hides each original member once. The two extra-mask families keep
at least one contig in the core:

- **Length fractions:** shuffle source members, then choose the nonempty prefix
  of at most $m-1$ contigs whose cumulative length fraction is closest to the
  target. Targets cycle through $0.05,0.10,0.15,0.20$ over the 20 episodes. A tie
  uses the first, shorter prefix. Every fifth extra episode is **absent**: its
  hidden members stay withheld and no positive candidate is supplied. They are
  not relabelled as negatives.
- **Counts:** the extra hidden count cycles from 1 to $\min(4,m-1)$; that many
  source contigs are sampled without replacement for each episode. Repeated masks
  are retained. The standard full-pool methods have no absent episodes.

The negative population is restricted to other supplied source bins whose
declared group differs from the episode's source group, **within the current
training or validation partition**. Counts above are upper bounds when that
population is small. The composition-hard rule takes the first eligible negatives
in the composition retrieval order; its random part samples eligible negatives
not already chosen, including those outside the 1,000-contig shortlist.
The embedding-hard rule uses the highest stored-vector dot products to the
normalized **unweighted** mean core embedding, then samples random negatives
without replacement from the remaining known population. Contig IDs break score
ties. This negative-selection mean differs from the length-weighted feature
centroid in [the feature formulas](features.md#embedding-similarity-six-features).

Standard seeds are 73019. Source-specific random streams depend on that seed,
stable source ID, operation and episode number; changing IDs can change masks or
negative samples. No seed is selected using genome-quality outcomes. The saved
example artifact records the actual core, hidden members, candidates, labels,
group and absent flag for every episode. Use it to inspect what was fitted;
nominal counts do not guarantee that every source contributes identical examples.

## Source-balanced classifier weights

These weights apply to all HGB profiles and the scaled logistic classifier.
Let $n_{sey}$ be the number of candidate rows with label $y\in\{0,1\}$ in
episode $e$ of source $s$. Let

$$
M_s=\sum_{e\in E_s}\left(\mathbf1\{n_{se1}>0\}+\mathbf1\{n_{se0}>0\}\right).
$$

$E_s$ is the set of generated episodes for source $s$; $\mathbf1$ is an indicator.
Each row $r$ of a nonempty class $y$ in episode $e$ receives

$$
a_r=\frac{1}{M_s\,n_{sey}}.
$$

Consequently each present episode/class contributes total mass $1/M_s$, and all
rows from a contributing source sum to 1, up to floating-point rounding. A
two-class episode has twice the total mass of a one-class episode from that
same source. Positive and negative classes are balanced **within each episode**;
their total masses over a whole source need not be equal when absent or
positive-unretrieved episodes occur. Empty classes receive no rows or mass.
This is not weighting by contig length or by the global class frequencies.
Logistic regression uses these same weights to fit both its scaler and classifier.

For example, if a source has one two-class episode with two positives and
eight negatives, $M_s=2$. Each positive has weight $1/(2\cdot2)=1/4$ and each
negative has weight $1/(2\cdot8)=1/16$. Each class contributes 1/2 and the
whole source contributes 1. These are contributions to the fitting objective,
not probabilities that particular contigs are correctly assigned.

Attention instead samples episodes: each epoch performs eight passes over a
randomly permuted list of sources, drawing one of each source's nonempty
episodes uniformly on each pass. It averages episode losses within each training
batch; it does not fit the classifier row-weight array above.

## Attention completion loss

For the standard one-query attention model, let $q_C$ be the normalized output
query for an episode's core, $A_e$ its supplied candidate set, and $P_e\subseteq A_e$
its positive candidates. Define training logits

$$
t_x=\frac{q_C^\top z_x}{\tau},\qquad \tau=0.1.
$$

With at least one positive, the implemented completion loss is

$$
\mathcal L_e=
\log\sum_{x\in A_e}e^{t_x}-\log\sum_{p\in P_e}e^{t_p}
-\frac{1}{|P_e|}\sum_{p\in P_e}\log\frac{e^{t_p}}{\sum_{x\in A_e}e^{t_x}}.
$$

The first term pair rewards probability mass on any positive; the final term
requires each positive to score well. With one positive these are two equal
negative-log-probability terms. The denominator contains the **sampled candidates
of that episode**, not the entire assembly or the whole inference donor pool.
For an absent episode, the completion objective uses
$\mathcal L_e=|A_e|^{-1}\sum_{x\in A_e}\log(1+e^{t_x})$ instead; the standard
attention training recipe uses leave-one-out masks with a positive.

Loss measures fit to the sampled membership task. With one positive assigned
softmax mass 0.8, its loss is $-2\ln0.8\approx0.446$; at mass 0.2 it is
$-2\ln0.2\approx3.219$. Lower is better for this training objective, but neither
value establishes HQ status or an externally calibrated membership probability.

The optimizer is AdamW with learning rate 0.001, batch size 16 and gradient-norm
clipping at 1.0. Inference uses the raw dot product $q_C^\top z_x$; dividing by
the training temperature is not an additional inference feature.

## Attention epoch selection

Default selection requires at least four unrelated multi-contig groups. It
partitions groups before generating masks and negative populations. For $g$
eligible groups, the holdout count is
$\min(\max(2,\lceil0.2g\rceil),g-2)$: it keeps at least two groups in each
partition, so a four-group dataset holds out two groups, not just 20%.
The held-out groups come from a seeded permutation of sorted group IDs.
The holdout mask family is
leave-one-out plus 20 count-based masks (at most four hidden members), without
absent episodes. Its candidate universe includes the hidden source members and
all known negatives from other groups in that holdout partition.

For a holdout episode, let $H_e$ be the total number of hidden true members,
$s_x=q_C^\top z_x$, and define the pessimistic rank of positive $p$ as
$R_p=|\{x\in A_e:s_x\ge s_p\}|$. Its membership recall is

$$
\operatorname{recall}_e@5=
\frac{\sum_{p\in P_e}\mathbf1\{R_p\le5\}}{H_e}.
$$

For example, two hidden members found within the first five positions out of
three hidden members give recall $2/3$. This counts recovered members, not the
fraction of the first five that are true. That other fraction is precision;
neither metric is a count of externally assessed HQ genomes.

Every tied candidate counts ahead of a positive, so a tie spanning the cutoff
does not receive optimistic credit. Unretrieved hidden members remain in the
denominator. Absent and empty-truth episodes are excluded. Average episode
recalls within each source, then average those source means; large sources do
not gain extra weight merely by supplying more episodes.

Selection runs for at most 40 epochs, with patience 6. An epoch replaces the
best one only if recall improves by more than $10^{-6}$; otherwise the stale
counter increases. After six stale epochs, selection stops. A fresh full-source
model is then initialized and trained for the selected epoch count, with the
same configured seed. The selected count is an outcome, not a forced constant.

HGB/logistic profiles need at least two unrelated multi-contig source groups.
Their standard recipes use fixed classifier settings rather than attention's
inner epoch selection. None of these checks is an independent-metagenome test
or an external genome-quality evaluation.

## Application and reproducibility

Applying models to their training dataset is supported and recorded as
**transductive overlap**. It does not establish biological generalization.
A separate compatible recipient dataset can also be declared; numerical feature
and encoder identities must match. Protected original memberships stay unchanged
and candidate pools are explicit. See [inputs](inputs.md) and [validation](validation.md).

The standard settings ship in the wheel. No quality-driven hyperparameter or
lucky-seed search occurs automatically. Demo/CI fixtures reduce settings on
artificial data; those reductions are not deployment recommendations. Exact
replay requires the recorded inputs, dependency builds, numerical sources,
random-stream identities and execution configuration.
