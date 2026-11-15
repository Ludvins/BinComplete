export const methods = {
  cosine: ["Length-weighted DNABERT-S centroid", "Full eligible pool", "Cosine similarity"],
  kmers: ["Composition and coverage HGB", "Composition shortlist", "Classifier score"],
  dnabert_s: ["DNABERT-S, composition and coverage HGB", "Composition shortlist", "Classifier score"],
  single: ["Single-query transformer attention", "Learned-query shortlist", "Raw dot product"],
  hgb_full_pool: ["Composition, coverage and DNABERT-S HGB", "Full eligible pool", "Classifier score"],
  logistic_full_pool: ["Scaled logistic regression", "Full eligible pool", "Classifier score"],
};

const transitions = new Set(["gain", "loss", "retained", "non_hq", "unknown"]);
const statuses = new Set(["pass", "fail", "unknown"]);
const transitionFor = {
  "fail,pass": "gain", "pass,fail": "loss", "pass,pass": "retained", "fail,fail": "non_hq",
};
const qualityFields = ["completeness", "contamination"];
const rnaFields = ["rrna_5s", "rrna_16s", "rrna_23s", "trna_types"];
export const trnaMinTypes = 18;
const assessmentFields = [...qualityFields, "hq_status", ...rnaFields];
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value) => typeof value === "string" && value.length > 0;
const sha256 = (value) => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
const finite = (value) => Number.isFinite(value);
const number = (value) => value === null || finite(value);
const bool = (value) => typeof value === "boolean";
const count = (value) => Number.isSafeInteger(value) && value >= 0;
const near = (left, right, scale = 1) => left === right || (finite(left) && finite(right)
  && Math.abs(left - right) <= 1e-12 * Math.max(scale, Math.abs(left), Math.abs(right)));
const emptyRna = { rrna_5s: null, rrna_16s: null, rrna_23s: null, trna_types: null, status: "unknown" };
const summaryFields = ["original_bins", "recipients", "modified_bins", "comparable_pairs", "unassessed_pairs", "original_hq", "final_hq", "hq_gains", "hq_losses", "net_hq_change"];

function quality(value) {
  if (!record(value) || !number(value.completeness) || !number(value.contamination) || !statuses.has(value.hq_status)
      || (value.completeness !== null && (value.completeness < 0 || value.completeness > 100)) || value.contamination < 0) return false;
  const { rna, checkm2 } = value;
  return (rna === undefined || (record(rna) && statuses.has(rna.status)
    && [rna.rrna_5s, rna.rrna_16s, rna.rrna_23s].every((call) => call === null || bool(call))
    && (rna.trna_types === null || (count(rna.trna_types) && rna.trna_types <= 20))))
    && (checkm2 === undefined || (record(checkm2) && Object.values(checkm2).every((field) => text(field) || finite(field))))
    && (value.hq_conflicts === undefined || (Array.isArray(value.hq_conflicts) && value.hq_conflicts.length > 0
      && value.hq_conflicts.every(text) && value.reported_hq_status === "pass" && value.hq_status === "unknown"));
}

function proposal(row) {
  return record(row) && [row.id, row.method, row.artifact_id].every(text)
    && [row.mean_added_score, row.min_added_score].every(finite)
    && [row.added_contig_count, row.added_bp, row.within_size_rank, row.priority_rank, row.assessment_tier].every(count)
    && quality(row.quality) && [row.assessed, row.approved, row.selected].every(bool)
    && (row.exported === null || bool(row.exported))
    && (row.donors === undefined || (Array.isArray(row.donors) && row.donors.every((donor) => record(donor)
      && text(donor.id) && count(donor.length_bp) && count(donor.rank) && finite(donor.mean_depth)
      && number(donor.depth_variance) && finite(donor.score))));
}

function bin(row) {
  return record(row) && text(row.id) && bool(row.recipient) && quality(row.original) && quality(row.final)
    && record(row.delta) && number(row.delta.completeness) && number(row.delta.contamination)
    && transitions.has(row.hq_transition) && Array.isArray(row.proposals) && row.proposals.every(proposal)
    && (row.selected_proposal_id === null || row.proposals.some((item) => item.id === row.selected_proposal_id))
    && (row.assessment_mismatches === undefined || (Array.isArray(row.assessment_mismatches)
      && row.assessment_mismatches.every((field) => assessmentFields.includes(field))));
}

function sameQuality(left, right) {
  return [...qualityFields, "hq_status", "reported_hq_status"].every((field) => left[field] === right[field])
    && [...rnaFields, "status"].every((field) => rna(left)[field] === rna(right)[field])
    && JSON.stringify(left.hq_conflicts ?? []) === JSON.stringify(right.hq_conflicts ?? [])
    && Object.keys({ ...left.checkm2, ...right.checkm2 }).every((field) => left.checkm2?.[field] === right.checkm2?.[field]);
}

function checkQuality(quality, isMimag, id) {
  const { rrna_5s, rrna_16s, rrna_23s, trna_types: types, status } = rna(quality);
  const markers = [rrna_5s, rrna_16s, rrna_23s];
  const rnaFails = markers.includes(false) || (types !== null && types < trnaMinTypes);
  const rnaStatus = markers.includes(null) || types === null ? "unknown" : rnaFails ? "fail" : "pass";
  if (status !== rnaStatus) throw new Error(`Inconsistent RNA verdict in ${id}.`);
  if (isMimag && quality.hq_status === "pass" && (rnaFails
      || (quality.completeness !== null && quality.completeness <= 90) || quality.contamination >= 5)) {
    throw new Error(`MIMAG HQ conflicts with the saved measurements in ${id}. Regenerate the report to flag the conflict.`);
  }
}

function checkSelection(bin, proposal) {
  if (proposal.selected !== (bin.selected_proposal_id === proposal.id)
      || (proposal.selected && (!bin.recipient || !proposal.approved)) || (proposal.exported && !proposal.selected)) {
    throw new Error(`Inconsistent approval, selection, or export status for proposal ${proposal.id}.`);
  }
  if (proposal.donors === undefined) return;
  let bases = 0;
  let total = 0;
  let minimum = Infinity;
  let scale = 1;
  const names = new Set();
  for (const donor of proposal.donors) {
    bases += donor.length_bp;
    total += donor.score;
    minimum = Math.min(minimum, donor.score);
    scale = Math.max(scale, Math.abs(donor.score));
    names.add(donor.id);
  }
  if (names.size !== proposal.donors.length || names.size !== proposal.added_contig_count
      || bases !== proposal.added_bp || minimum !== proposal.min_added_score
      || !near(total / names.size, proposal.mean_added_score, scale)) {
    throw new Error(`Donor records disagree with the totals or scores for proposal ${proposal.id}.`);
  }
}

function checkContigEvidence(bin) {
  if (bin.contig_evidence === undefined) return;
  const evidence = bin.contig_evidence;
  const donors = new Map(bin.proposals.find((item) => item.selected)?.donors?.map((row) => [row.id, row]) ?? []);
  const validGc = (value) => value === null || (finite(value) && value >= 0 && value <= 1);
  if (!record(evidence) || !donors.size || [...donors.values()].some((row) => row.length_bp < 1 || row.mean_depth < 0)
      || !Array.isArray(evidence.original) || !evidence.original.length
      || !evidence.original.every((row) => record(row) && count(row.length_bp) && row.length_bp > 0
        && finite(row.mean_depth) && row.mean_depth >= 0 && validGc(row.gc_fraction))
      || !Array.isArray(evidence.added) || evidence.added.length !== donors.size
      || !evidence.added.every((row) => record(row) && donors.has(row.id) && validGc(row.gc_fraction)
        && sha256(row.sequence_sha256))
      || new Set(evidence.added.map((row) => row.id)).size !== donors.size
      || !Array.isArray(evidence.rna_features) || !evidence.rna_features.every((row) => record(row)
        && donors.has(row.contig_id) && text(row.feature) && count(row.start_bp) && row.start_bp > 0
        && count(row.end_bp) && row.end_bp >= row.start_bp && row.end_bp <= donors.get(row.contig_id).length_bp
        && ["+", "-"].includes(row.strand))) {
    throw new Error(`Invalid contig evidence for bin ${bin.id}.`);
  }
}

function summarize(report, checkPairs = false) {
  const totals = { original_bins: report.bins.length, recipients: 0, modified_bins: 0, comparable_pairs: 0,
    unassessed_pairs: 0, original_hq: 0, final_hq: 0, hq_gains: 0, hq_losses: 0, net_hq_change: 0 };
  const rnaTotals = { comparable_pairs: 0, incomplete_pairs: 0, gains: 0, losses: 0 };
  for (const bin of report.bins) {
    const mismatches = bin.assessment_mismatches ?? [];
    const hqMismatch = [...qualityFields, "hq_status", ...(report.hq_policy.standard === "mimag" ? rnaFields : [])]
      .some((field) => mismatches.includes(field));
    const transition = hqMismatch ? "unknown" : transitionFor[[bin.original.hq_status, bin.final.hq_status].join(",")] ?? "unknown";
    if (checkPairs) {
      const savedFinal = bin.selected_proposal_id === null ? bin.original : bin.proposals.find((row) => row.id === bin.selected_proposal_id).quality;
      if (!sameQuality(savedFinal, bin.final) || transition !== bin.hq_transition) {
        throw new Error(`Inconsistent final quality or HQ transition for bin ${bin.id}.`);
      }
      for (const field of qualityFields) {
        const before = bin.original[field];
        const after = bin.final[field];
        const delta = before === null || after === null || mismatches.includes(field) ? null : after - before;
        if (!near(bin.delta[field], delta)) throw new Error(`Inconsistent ${field} change for bin ${bin.id}.`);
      }
    }
    if (!bin.recipient) continue;
    totals.recipients++;
    if (bin.selected_proposal_id !== null) totals.modified_bins++;
    if (transition === "unknown") totals.unassessed_pairs++;
    else {
      totals.comparable_pairs++;
      if (bin.original.hq_status === "pass") totals.original_hq++;
      if (bin.final.hq_status === "pass") totals.final_hq++;
      if (transition === "gain") totals.hq_gains++;
      if (transition === "loss") totals.hq_losses++;
    }
    const pair = pairRna(bin);
    if (pair.includes("unknown") || rnaFields.some((field) => mismatches.includes(field))) rnaTotals.incomplete_pairs++;
    else {
      rnaTotals.comparable_pairs++;
      if (pair.join(",") === "fail,pass") rnaTotals.gains++;
      if (pair.join(",") === "pass,fail") rnaTotals.losses++;
    }
  }
  totals.net_hq_change = totals.hq_gains - totals.hq_losses;
  if (!totals.comparable_pairs) {
    for (const field of ["original_hq", "final_hq", "hq_gains", "hq_losses", "net_hq_change"]) totals[field] = null;
  }
  return { summary: totals, rna_summary: rnaTotals };
}

function guardLegacyComparisons(report) {
  const sources = report.provenance.assessment_sources.filter((source) => source.format === "checkm2");
  const signatures = new Set(sources.map((source) => JSON.stringify(
    [source.tool, source.version, source.settings, source.completeness_column ?? "Completeness"],
  )));
  if (signatures.size <= 1 || sources.every((source) => source.scope !== undefined)) return report;
  const bins = report.bins.map((row) => row.selected_proposal_id === null ? row : {
    ...row, hq_transition: "unknown",
    assessment_mismatches: [...new Set([...(row.assessment_mismatches ?? []), ...qualityFields])],
    delta: { completeness: null, contamination: null },
  });
  const { summary } = summarize({ ...report, bins });
  return { ...report, bins, summary, uncheckedCheckm2Settings: true };
}

function comparison(value) {
  return record(value) && count(value.distinct_construction_union) && count(value.common_recipients_all_methods)
    && record(value.methods) && Object.values(value.methods).every((row) => record(row) && text(row.method)
      && [row.evaluated_recipients, row.scored_candidates, row.proposals, row.distinct_constructions].every(count)
      && (row.attention_validation == null || (record(row.attention_validation)
        && count(row.attention_validation.epoch) && number(row.attention_validation.recall_at_5) && number(row.attention_validation.loss))))
    && Array.isArray(value.pairwise) && value.pairwise.every((pair) => record(pair) && Array.isArray(pair.methods)
      && pair.methods.length === 2 && pair.methods.every((method) => Object.hasOwn(value.methods, method))
      && count(pair.common_recipients) && count(pair.identical_constructions));
}

export function prepareReport(report) {
  if (!record(report) || report.format !== "bincomplete-report") throw new Error("Choose a BinComplete report JSON file.");
  if (![1, 2].includes(report.schema_version)) throw new Error("This report schema version is unsupported.");
  const { bins, summary, rna_summary: rnaSummary, hq_policy: policy, provenance } = report;
  const rnaSource = provenance?.contig_rna_source;
  if (!record(summary) || !summaryFields.every((key) => number(summary[key]))
      || !record(policy) || !text(policy.description) || !record(provenance)
      || !text(provenance.bincomplete_version) || !text(provenance.dataset_identity) || !bool(provenance.selection_optimal)
      || !Array.isArray(provenance.assessment_sources) || !provenance.assessment_sources.every((source) =>
        record(source) && [source.tool, source.version, source.settings].every(text))
      || !Array.isArray(bins) || !bins.every(bin)
      || (rnaSource !== undefined && (!record(rnaSource) || ![rnaSource.tool, rnaSource.version, rnaSource.settings].every(text)
        || !sha256(rnaSource.report_sha256)))
      || (rnaSummary !== undefined && (!record(rnaSummary) || !["comparable_pairs", "incomplete_pairs", "gains", "losses"].every((key) => count(rnaSummary[key]))))
      || (report.comparison !== undefined && !comparison(report.comparison))) {
    throw new Error("The report is missing required data or contains invalid values.");
  }
  const byBin = new Map();
  const byProposal = new Map();
  for (const row of bins) {
    if (byBin.has(row.id)) throw new Error("The report contains duplicate bin IDs.");
    byBin.set(row.id, row);
    checkQuality(row.original, isMimag(report), row.id);
    checkQuality(row.final, isMimag(report), row.id);
    for (const item of row.proposals) {
      if (byProposal.has(item.id)) throw new Error("The report contains duplicate proposal IDs.");
      checkSelection(row, item);
      checkQuality(item.quality, isMimag(report), item.id);
      byProposal.set(item.id, { ...item, recipient: row.id });
    }
    checkContigEvidence(row);
  }
  const totals = summarize(report, true);
  if (Object.entries(totals.summary).some(([field, value]) => summary[field] !== value)
      || (rnaSummary && Object.entries(totals.rna_summary).some(([field, value]) => rnaSummary[field] !== value))) {
    throw new Error("The report summary disagrees with the saved bin results.");
  }
  const safeReport = guardLegacyComparisons(report);
  for (const row of safeReport.bins) byBin.set(row.id, row);
  return { ...safeReport, byBin, byProposal, proposals: [...byProposal.values()] };
}

export const rna = (quality) => quality.rna ?? emptyRna;
export const pairRna = (row) => [rna(row.original).status, rna(row.final).status];
export const isMimag = (report) => report.hq_policy.standard === "mimag";
export const decision = (row) => row.selected ? row.exported ? "Selected · Exported" : "Selected"
  : row.approved ? "Approved" : row.assessed ? "Evaluated" : "Not assessed";
