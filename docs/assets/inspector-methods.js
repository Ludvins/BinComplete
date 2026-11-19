import { methods } from "./inspector-data.js";
import { browserTable, column, details, el, link, num, pageTitle, panel, pct, pill, score, staticTable, stats } from "./inspector-ui.js";

const rate = (value) => value == null ? "—" : pct(value * 100);
const code = (value) => el("code", "", value);
const titleFor = (rows, row) => rows.filter((other) => other.method === row.method).length > 1
  ? `${row.method} · ${row.id.slice(0, 8)}` : row.method;

function methodRows(report) {
  const rows = new Map();
  function add(id, method, saved = {}) {
    rows.set(id, { ...saved, id, method, generated: 0, assessed: 0, approved: 0, selected: 0, gain: 0, loss: 0, comparable: 0 });
  }
  for (const [id, row] of Object.entries(report.methodArtifacts ?? report.comparison?.methods ?? {})) add(id, row.method, row);
  for (const proposal of report.proposals) {
    if (!rows.has(proposal.artifact_id)) add(proposal.artifact_id, proposal.method);
    const row = rows.get(proposal.artifact_id);
    row.generated++;
    for (const key of ["assessed", "approved", "selected"]) row[key] += Number(proposal[key]);
  }
  for (const bin of report.bins) {
    if (!bin.selected_proposal_id || bin.hq_transition === "unknown") continue;
    const proposal = report.byProposal.get(bin.selected_proposal_id);
    const row = rows.get(proposal.artifact_id);
    row.comparable++;
    if (bin.hq_transition === "gain" || bin.hq_transition === "loss") row[bin.hq_transition]++;
  }
  return Array.from(rows.values()).sort((a, b) => a.method.localeCompare(b.method, "en") || a.id.localeCompare(b.id, "en"));
}

function trainingDetails(row) {
  const saved = row.training;
  const attention = row.attention_validation;
  if (!saved && !attention) return el("p", "description", "Training metrics are unavailable in this report.");
  return details([
    ...(saved ? [
      ["Fit", saved.learned ? "Learned" : "Analytic"],
      ["Training bins", num(saved.sources)], ["Training groups", num(saved.groups)],
      ["Training rows", num(saved.rows)], ["Positive rows", num(saved.positives)],
      ["Episodes", num(saved.episodes)], ["Features", num(saved.features)],
      ["Weighting", saved.weighting ?? "—"], ...(saved.objective ? [["Objective", saved.objective]] : []),
    ] : []),
    ...(attention ? [
      ["Epochs used for refit", num(attention.epoch)], ["Held-out bins", num(attention.held_out_sources)],
      ["Held-out groups", num(attention.held_out_groups)], ["Held-out Precision@5", rate(attention.precision_at_5)],
      ["Held-out Recall@5", rate(attention.recall_at_5)], ["Training loss at selected epoch", score(attention.loss)],
    ] : []),
  ]);
}

function methodEvidence(rows, training, benchmarks) {
  const method = column("Method", (row) => titleFor(rows, row), undefined, true);
  const trainingMetrics = [
    method,
    column("Precision@5", (row) => rate(row.attention_validation.precision_at_5)),
    column("Recall@5", (row) => rate(row.attention_validation.recall_at_5)),
    column("Held-out bins", (row) => num(row.attention_validation.held_out_sources)),
    column("Held-out groups", (row) => num(row.attention_validation.held_out_groups)),
  ];
  const benchmarkMetrics = [
    method,
    column("Benchmark", (row) => row.benchmark.benchmark_id, (row) => el("div", "",
      row.benchmark.benchmark_id, ...(row.benchmark.run_id ? [el("small", "method-run", row.benchmark.run_id)] : [])), true),
    column("Protocol", (row) => row.benchmark.protocol, undefined, true),
    column("Precision@5", (row) => rate(row.benchmark.precision_at_5)),
    column("Recall@5", (row) => rate(row.benchmark.recall_at_5)),
    column("Test sample", (row) => [
      ["evaluated_bins", "bins"], ["evaluated_groups", "groups"], ["evaluated_episodes", "episodes"],
    ].filter(([key]) => row.benchmark[key] != null).map(([key, label]) => `${num(row.benchmark[key])} ${label}`).join(" · ") || "—", undefined, true),
  ];
  return el("div", "split method-evidence",
    panel("Training validation",
      el("p", "chart-caption method-caption", "Held-out groups used to select the training epoch."),
      staticTable("Training validation", trainingMetrics, training)),
    panel("Benchmark results",
      el("p", "chart-caption method-caption", "Compare scores from the same benchmark and protocol."),
      staticTable("Benchmark results", benchmarkMetrics, benchmarks)));
}

function rankingDetails(row) {
  const profile = Object.hasOwn(methods, row.method) ? methods[row.method] : undefined;
  return details([
    ["Method", profile?.[0] ?? row.method],
    ["Candidate pool", profile?.[1] ?? "—"],
    ["Membership score", profile?.[2] ?? "—"],
    ["Scorer", row.training?.scorer ?? "—"],
    ["Retriever", row.training?.retriever ?? "—"],
    ...(row.training?.classifier ? [["Classifier", row.training.classifier]] : []),
    ["Proposal search", row.search?.name ?? "—"],
    ["Proposal budget per bin", num(row.search?.max_checks)],
    ["Singleton depth", num(row.search?.singleton_depth)],
    ["Boundary width", num(row.search?.boundary_width)],
    ["Max added length fraction", rate(row.search?.max_added_fraction)],
  ]);
}

export function methodCatalog(report) {
  const rows = methodRows(report);
  const training = rows.filter(({ attention_validation: saved }) =>
    saved?.precision_at_5 != null || saved?.recall_at_5 != null);
  const benchmarks = rows.filter((row) => row.benchmark);
  const hasComparison = report.comparison || report.sourceReports?.some((source) => source.comparison);
  const selectedDetails = el("div", "method-detail");
  let current = rows[0];
  let currentTable;
  function choose(row) {
    current = row;
    for (const button of currentTable?.querySelectorAll("[data-artifact]") ?? []) {
      const active = button.dataset.artifact === row.id;
      button.setAttribute("aria-pressed", String(active));
      button.closest("tr").classList.toggle("selected", active);
    }
    const outcome = [
      ["Targets", num(row.evaluated_recipients)], ["Candidates scored", num(row.scored_candidates)],
      ["Distinct constructions", num(row.distinct_constructions)], ["Unique to this artifact", num(row.constructions_unique_to_method)],
      ["Targets overlapping training sequences", num(row.overlap?.overlapping_recipients)],
      ["Shared declared training groups", num(row.overlap?.shared_declared_groups?.length)],
      ["Training overlap", row.overlap?.mode === "transductive" ? "Transductive"
        : row.overlap?.mode === "no_exact_sequence_overlap_detected" ? "No exact overlap detected" : "—"],
    ];
    selectedDetails.replaceChildren(
      el("h2", "section-title", titleFor(rows, row)),
      el("div", "trio", panel("Training and validation", trainingDetails(row)),
        panel("Ranking and search", rankingDetails(row)), panel("Coverage and overlap", details(outcome))));
  }
  const numeric = (label, field) => column(label, (row) => row[field], (row) => num(row[field]));
  const columns = [
    column("Method", (row) => titleFor(rows, row), (row) => {
      const button = el("button", "text-button method-name", titleFor(rows, row));
      button.type = "button";
      button.dataset.artifact = row.id;
      button.setAttribute("aria-pressed", String(row === current));
      button.addEventListener("click", () => choose(row));
      return button;
    }),
    numeric("Targets", "evaluated_recipients"), numeric("Candidates scored", "scored_candidates"),
    numeric("Proposals", "generated"), numeric("Evaluated", "assessed"),
    numeric("Approved", "approved"), numeric("Selected", "selected"),
    column("HQ gains / losses", (row) => row.comparable ? row.gain - row.loss : null, (row) =>
      row.comparable ? el("span", "", el("span", "method-gain", `+${num(row.gain)}`), " / ", el("span", "method-loss", `−${num(row.loss)}`)) : "—"),
  ];
  const profiles = el("details", "panel bin-disclosure method-profiles",
    el("summary", "", "Standard method profiles"), staticTable("Standard method profiles", [
      column("Profile", (row) => row[0], (row) => code(row[0])),
      column("Method", (row) => row[1][0]), column("Candidate pool", (row) => row[1][1]), column("Membership score", (row) => row[1][2]),
    ], Object.entries(methods)));
  if (current) choose(current);
  return el("div", "", pageTitle("Methods and ranking", { badge: report.synthetic ? pill("Synthetic run", "info") : undefined }),
    stats([["Method artifacts", num(rows.length)], ["Proposals", num(report.proposals.length)],
      ["With training validation", num(training.length)], ["With benchmark results", num(benchmarks.length)]]),
    methodEvidence(rows, training, benchmarks),
    el("div", "method-toolbar", el("h2", "", "Proposal outcomes"),
      ...(hasComparison ? [link("Method comparison →", "#methods/comparison")] : [])),
    el("p", "chart-caption method-caption", "HQ gains and losses count comparable bins with selected proposals."),
    browserTable({ label: "Proposal outcomes", columns, rows, selected: (row) => row === current,
      afterRows: (_, table) => { currentTable = table; } }),
    selectedDetails, profiles,
    el("p", "footnote", report.sourceReports
      ? "Assessment settings are listed per file in Provenance. Scores are scoped to each artifact and recipient."
      : `Assessment order: ${report.provenance.priority?.strategy ?? "size_balanced_mean"} · Tier 1: first ${num(report.provenance.priority?.first_tier_limit ?? 40)} per artifact and recipient · Scores are scoped to that artifact and recipient.`));
}

function comparisonPanels(data, rows) {
  const names = new Map(rows.map((row) => [row.id, titleFor(rows, row)]));
  const counts = [
    ["Common targets", "common_recipients"], ["Identical constructions", "identical_constructions"],
    ["Targets with identical constructions", "recipients_with_identical_constructions"],
    ["Left only", "left_only_constructions_on_common_targets"], ["Right only", "right_only_constructions_on_common_targets"],
  ];
  const columns = [
    column("Method pair", (row) => row.methods.map((id) => names.get(id)).join(" / ")),
    ...counts.map(([label, field]) => column(label, (row) => row[field], (row) => num(row[field]))),
  ];
  const patterns = data.exact_method_intersections ?? [];
  return [
    stats([["Artifact count", num(Object.keys(data.methods).length)], ["Shared targets", num(data.common_recipients_all_methods)],
      ["Target union", num(data.distinct_recipient_union)], ["Distinct constructions", num(data.distinct_construction_union)],
      ["Contigs proposed for multiple targets", num(data.donor_contigs_proposed_for_multiple_recipients)]]),
    panel("Pairwise construction overlap", staticTable("Pairwise construction overlap", columns, data.pairwise)),
    el("p", "footnote", "Identical constructions share a recipient and full contig membership. Left and right only counts use the common targets."),
    ...(patterns.length ? [el("div", "method-detail", panel("Exact construction sharing", staticTable("Exact construction sharing", [
      column("Artifacts sharing the same construction", (row) => row.methods.map((id) => names.get(id)).join(" / "), undefined, true),
      column("Constructions", (row) => row.constructions, (row) => num(row.constructions)),
    ], patterns)))] : []),
  ];
}

export function methodComparison(report) {
  const title = pageTitle("Method comparison", { breadcrumb: link("Methods", "#methods") });
  if (report.sourceReports) {
    const sections = report.sourceReports.filter((source) => source.comparison).map((source) => {
      const rows = Object.entries(source.comparison.methods).map(([id, row]) => ({ ...row, id }));
      return el("section", "source-report", el("h2", "section-title", `Dataset ${source.datasetNumber} · ${source.name}`),
        methodEvidence(rows, rows.filter((row) => row.attention_validation?.precision_at_5 != null
          || row.attention_validation?.recall_at_5 != null), rows.filter((row) => row.benchmark)),
        ...comparisonPanels(source.comparison, rows));
    });
    if (sections.length) return el("div", "", title, ...sections);
  }
  if (!report.comparison) return el("div", "", title, el("p", "notice", "Method comparison is unavailable in this report."));
  return el("div", "", title, ...comparisonPanels(report.comparison, methodRows(report)));
}
