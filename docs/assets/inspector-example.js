// Deterministic fictional data for previewing the inspector without a report file.
import { methods } from "./inspector-data.js";

const rna = (five = true, sixteen = true, twentyThree = true, types = 19) => ({
  rrna_5s: five, rrna_16s: sixteen, rrna_23s: twentyThree, trna_types: types,
  status: [five, sixteen, twentyThree, types].includes(null) ? "unknown" : five && sixteen && twentyThree && types >= 18 ? "pass" : "fail",
});
const unknownRna = rna(null, null, null, null);

function quality(completeness, contamination, evidence = rna(), added = false) {
  const metricsPass = completeness > 90 && contamination < 5;
  const hq_status = completeness === null || contamination === null || (metricsPass && evidence.status === "unknown")
    ? "unknown" : metricsPass && evidence.status === "pass" ? "pass" : "fail";
  const checkm2 = completeness === null ? {} : {
    completeness_model: "General", genome_size: added ? 2198600 : 2170000, contig_count: added ? 87 : 84,
    contig_n50: 42600, gc_content: .528, coding_density: added ? .892 : .891, average_gene_length: 309.2,
    coding_sequences: added ? 2071 : 2044, translation_table: 11, max_contig_length: 173400, notes: "None",
  };
  return { completeness, contamination, hq_status, rna: evidence, checkm2 };
}

const unassessed = quality(null, null, unknownRna);
const methodIds = Object.keys(methods);
const rows = [];

function addBin(id, original, final, method, recipient = true) {
  const selected = method ? `${id}-selected` : null;
  const bin = {
    id, original, final, recipient, selected_proposal_id: selected, proposals: [], assessment_mismatches: [],
    hq_transition: [original.hq_status, final.hq_status].includes("unknown") ? "unknown"
      : original.hq_status === "pass" ? final.hq_status === "pass" ? "retained" : "loss"
      : final.hq_status === "pass" ? "gain" : "non_hq",
    delta: Object.fromEntries(["completeness", "contamination"].map((metric) => [metric,
      original[metric] == null || final[metric] == null ? null : final[metric] - original[metric]])),
  };
  if (method) addProposal(bin, { id: selected, method, assessed: final !== unassessed, selected: true, quality: final });
  rows.push(bin);
  return bin;
}

function addProposal(bin, fields) {
  const donor = { id: `contig_${bin.id.slice(-3)}_${bin.proposals.length + 1}`, length_bp: 9200, mean_depth: 34.2, depth_variance: 12.8, score: .932, rank: 4 };
  const data = {
    id: `${bin.id}-${bin.proposals.length + 1}`, method: "logistic_full_pool", quality: unassessed, donors: [donor],
    assessed: false, selected: false, within_size_rank: 1, priority_rank: 1, assessment_tier: 1, ...fields,
  };
  data.artifact_id = data.method;
  data.added_bp = data.donors.reduce((sum, item) => sum + item.length_bp, 0);
  data.added_contig_count = data.donors.length;
  data.mean_added_score = data.donors.reduce((sum, item) => sum + item.score, 0) / data.donors.length;
  data.min_added_score = Math.min(...data.donors.map((item) => item.score));
  data.approved = data.selected;
  data.exported = data.selected;
  bin.proposals.push(data);
  return data;
}

const focus = addBin("bin_042", quality(85.4, 2.4, rna(true, false, true, 17)), quality(94.7, 2.9, rna(), true));
const selected = addProposal(focus, { id: "P-042-L3", assessed: true, selected: true, quality: focus.final, priority_rank: 3, donors: [
  { id: "contig_514", length_bp: 10800, mean_depth: 34.2, depth_variance: 12.8, score: .939, rank: 3 },
  { id: "contig_276", length_bp: 9200, mean_depth: 33.6, depth_variance: 11.4, score: .917, rank: 6 },
  { id: "contig_982", length_bp: 8600, mean_depth: 32.9, depth_variance: 13.2, score: .904, rank: 10 },
] });
focus.selected_proposal_id = selected.id;
focus.contig_evidence = {
  original: Array.from({ length: 18 }, (_, i) => ({
    length_bp: 4200 + (i * 473 % 9400), mean_depth: 26 + (i * 3.1 % 13), gc_fraction: .46 + (i * .017 % .075),
  })),
  added: [
    { id: "contig_514", gc_fraction: .45, sequence_sha256: "4e4cd8094e1041716f728927e25d08c7d19cf94c01d896fb01c5d07d67b6879c" },
    { id: "contig_276", gc_fraction: .50, sequence_sha256: "71d75289d2936269e4e78b2a9759c2f1e68934dc3dbd8c5db320c175e6aa0350" },
    { id: "contig_982", gc_fraction: .55, sequence_sha256: "cfe291575dab07e3cd3f1b70535dd60719b7475b709d61d6818b96ae3318f1e1" },
  ],
  rna_features: [
    { contig_id: "contig_514", feature: "16S rRNA", start_bp: 2401, end_bp: 3900, strand: "+" },
    { contig_id: "contig_276", feature: "tRNA-Lys", start_bp: 301, end_bp: 380, strand: "-" },
    { contig_id: "contig_982", feature: "tRNA-Trp", start_bp: 521, end_bp: 600, strand: "+" },
  ],
};
for (const [id, count, rank, sizeRank, first, assessed] of [
  ["L1", 1, 1, 1, .947, true], ["L2", 2, 2, 1, .943, true], ["L4", 1, 4, 2, .912, true],
  ["L5", 2, 5, 2, .917, true], ["L6", 3, 6, 2, .917, true], ["L41", 2, 41, 14, .747, false],
]) addProposal(focus, { id: `P-042-${id}`, assessed, priority_rank: rank, within_size_rank: sizeRank,
  assessment_tier: rank > 40 ? 2 : 1, quality: assessed ? quality(90.2, 4.8, rna(true, false, true, 17)) : unassessed,
  donors: Array.from({ length: count }, (_, index) => ({ id: `contig_${id}_${index + 1}`, length_bp: 9100 + index * 900,
    mean_depth: 33 + index, depth_variance: 13.4 + index, score: first - index * .022, rank: rank * 3 + index })) });
addProposal(focus, { id: "P-042-H22", method: "hgb_full_pool", assessed: true, priority_rank: 22, quality: quality(95.2, 6.3) });
addProposal(focus, { id: "P-042-C41", method: "cosine", priority_rank: 41, within_size_rank: 14, assessment_tier: 2 });
addBin("bin_063", quality(76.8, 4.9, rna(false, false, true, 16)), quality(90.2, 3.7, rna(true, true, true, 18), true), "hgb_full_pool");
addBin("bin_118", quality(92, 2.8), quality(87.6, 3.4, rna(), true), "logistic_full_pool");
addBin("bin_156", quality(70.1, 5.6, rna(false, true, true, 15)), quality(91.5, 4, rna(true, true, true, 17), true), "hgb_full_pool");
addBin("bin_181", quality(93.9, 3), quality(86.3, 3.8, rna(), true), "logistic_full_pool");
addBin("bin_207", quality(78.5, 6.1, unknownRna), quality(91, 4.6, unknownRna, true), "hgb_full_pool");
const occupied = new Set(rows.map((row) => row.id));
for (let i = 1; i <= 248; i++) {
  const id = `bin_${String(i).padStart(3, "0")}`;
  if (occupied.has(id)) continue;
  const recipient = rows.length < 190;
  const complete = 61 + (i * 13 % 380) / 10;
  const contamination = 1 + (i * 11 % 93) / 10;
  const original = quality(complete, contamination, i % 13 === 0 ? unknownRna : rna(true, true, i % 8 !== 0, 17 + i % 4));
  const method = recipient && i % 4 === 0 ? methodIds[i % methodIds.length] : null;
  const final = method ? quality(Math.min(100, complete + 7.7), contamination + .4, rna(), true) : original;
  const bin = addBin(id, original, final, method, recipient);
  if (recipient) addProposal(bin, { method: methodIds[(i + 1) % methodIds.length], priority_rank: 2 });
}

const recipients = rows.filter((row) => row.recipient);
const comparable = recipients.filter((row) => row.hq_transition !== "unknown");
const transitions = (value) => comparable.filter((row) => row.hq_transition === value).length;
const rnaComparable = recipients.filter((row) => [row.original, row.final].every((q) => q.rna.status !== "unknown"));
const rnaChanges = (before, after) => rnaComparable.filter((row) => row.original.rna.status === before && row.final.rna.status === after).length;
const proposals = rows.flatMap((row) => row.proposals);
const methodRows = Object.fromEntries(methodIds.map((method, i) => {
  const total = proposals.filter((row) => row.method === method).length;
  return [method, { method, evaluated_recipients: 190, scored_candidates: 12840 - (i % 3) * 260,
    proposals: total, distinct_constructions: total, constructions_unique_to_method: total,
    training: method === "cosine" ? { learned: false, sources: 0, groups: 0, rows: 0, episodes: 0,
      scorer: "cosine_membership", retriever: "all_eligible" } : { learned: true, sources: 42, groups: 34, rows: 15480,
      positives: 4070, episodes: 2560, weighting: "source_balanced" },
    search: { name: "expanded_ranked_alternatives", max_checks: 128, singleton_depth: 50, boundary_width: 2, max_added_fraction: 1 },
    ...(method === "single" ? { attention_validation: { epoch: 18, recall_at_5: .62, loss: 1.36,
      held_out_sources: 6, held_out_groups: 4 } } : {}),
    ...(["hgb_full_pool", "logistic_full_pool"].includes(method) ? { benchmark: {
      precision_at_5: method === "hgb_full_pool" ? .55 : .48, recall_at_5: method === "hgb_full_pool" ? .66 : .61,
      benchmark_id: "Synthetic benchmark", protocol: "Group-disjoint test", run_id: "Synthetic run",
      evaluated_bins: 24, evaluated_groups: 12, evaluated_episodes: 96, report_sha256: "f".repeat(64) } } : {}) }];
}));

export const example = {
  format: "bincomplete-report", schema_version: 2, synthetic: true,
  hq_policy: { mode: "external", standard: "mimag", description: "MIMAG high-quality draft · >90% completeness and <5% contamination" },
  provenance: {
    bincomplete_version: "Synthetic example", dataset_identity: "synthetic-inspector-example", selection_optimal: true,
    selection_canonical_tie_break: true, selection_upper_bound: recipients.filter((row) => row.selected_proposal_id).length,
    priority: { strategy: "size_balanced_mean", version: 1, first_tier_limit: 40 },
    assessment_sources: [{ tool: "CheckM2", version: "Example", settings: "Synthetic quality measurements", format: "checkm2" },
      { tool: "External RNA annotation", version: "Example", settings: "Synthetic marker calls and tRNA amino acid types", format: "tsv" }],
  },
  summary: {
    original_bins: rows.length, recipients: recipients.length, modified_bins: recipients.filter((row) => row.selected_proposal_id).length,
    comparable_pairs: comparable.length, unassessed_pairs: recipients.length - comparable.length,
    original_hq: comparable.filter((row) => row.original.hq_status === "pass").length,
    final_hq: comparable.filter((row) => row.final.hq_status === "pass").length,
    hq_gains: transitions("gain"), hq_losses: transitions("loss"), net_hq_change: transitions("gain") - transitions("loss"),
  },
  rna_summary: { comparable_pairs: rnaComparable.length, incomplete_pairs: recipients.length - rnaComparable.length,
    gains: rnaChanges("fail", "pass"), losses: rnaChanges("pass", "fail") },
  comparison: { methods: methodRows, common_recipients_all_methods: 190, distinct_construction_union: proposals.length,
    donor_contigs_proposed_for_multiple_recipients: 0, pairwise: [{ methods: ["cosine", "dnabert_s"], common_recipients: 190,
      identical_constructions: 0, recipients_with_identical_constructions: 0 }] },
  bins: rows,
};
