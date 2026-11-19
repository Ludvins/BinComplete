import { addedContigEvidence } from "./inspector-contigs.js";
import { decision, isMimag, methods, pairRna, rna, trnaMinTypes } from "./inspector-data.js";
import {
  basePairs, browserTable, change, column, details, el, hqBadge, link, matches, num, pageTitle,
  paired, panel, pct, pill, rnaTypes, routeTo, score, staticTable, stats, verdict, yesNo,
} from "./inspector-ui.js";

const markers = [["5S rRNA", "rrna_5s"], ["16S rRNA", "rrna_16s"], ["23S rRNA", "rrna_23s"]];
const qualityPairs = [["Completeness", "completeness"], ["Contamination", "contamination"]];
const hqOptions = [["gain", "Gained HQ"], ["loss", "Lost HQ"], ["retained", "Kept HQ"], ["non_hq", "Below HQ"], ["unknown", "Unknown"]];
const rnaOptions = [["pass", "Pass"], ["fail", "Fail"], ["unknown", "Unknown"]];
const all = [["", "All"]];
const search = (label, display) => ({ label, test: (row, query) => matches(display(row), query) });
const select = (label, options, value, initial) => ({ label, options: [...all, ...options], initial, test: (row, key) => value(row) === key });
const states = (label, value) => select(label, [["yes", "Yes"], ["no", "No"], ["unknown", "Unknown"]],
  (row) => value(row) == null ? "unknown" : value(row) ? "yes" : "no");
const status = (row) => pill(decision(row), row.selected ? "good" : row.assessed ? "info" : "");
const idLink = (page, id) => link(id, routeTo(page, id));
const binId = column("Bin ID", (row) => row.id, (row) => idLink("bins", row.id));
const headTable = (title, columns, rows, selected) => el("section", "panel table-panel", el("h2", "", title), staticTable(title, columns, rows, selected));
const syntax = (value) => el("code", "", String(value ?? "—"));
const reportMode = (report) => isMimag(report) ? "MIMAG HQ" : "HQ";
const contigCount = (value) => `${num(value)} ${value === 1 ? "contig" : "contigs"}`;
const fraction = (value) => value == null ? "—" : pct(value * 100);
const assessmentLabels = { completeness: "completeness", contamination: "contamination", hq_status: "HQ verdict",
  rrna_5s: "5S rRNA", rrna_16s: "16S rRNA", rrna_23s: "23S rRNA", trna_types: "tRNA amino acid types" };
const rnaMismatch = (row) => row.assessment_mismatches?.some((field) => ["rrna_5s", "rrna_16s", "rrna_23s", "trna_types"].includes(field));

function deltaBadge(label, metric, delta) {
  const formatted = change(delta);
  const improves = metric === "completeness" ? delta > 0 : delta < 0;
  const badge = pill(`${formatted} pp`, formatted === "0" ? "" : improves ? "good" : "warn");
  badge.title = `${label} change ${formatted} percentage points`;
  return badge;
}

function percentPair(label, metric, showDelta = false) {
  return column(label, (row) => row.final[metric], (row) => {
    const pair = paired(row.original[metric], row.final[metric]);
    const delta = row.delta[metric];
    if (!showDelta || delta == null) return pair;
    return el("span", "quality-pair", el("span", "", pair), deltaBadge(label, metric, delta));
  });
}

function hqConflict(quality, label) {
  if (!quality.hq_conflicts?.length) return [];
  return [el("p", "notice warn", `${label}: external MIMAG HQ pass conflicts with ${quality.hq_conflicts.join(", ")}. HQ status is unresolved.`)];
}

function selectedProposal(report, bin) {
  return report.byProposal.get(bin.selected_proposal_id);
}

function markerCall(value, compact = false) {
  const label = value == null ? "Not assessed" : value ? "Detected" : "Not detected";
  if (!compact) return pill(label, value == null ? "" : value ? "good" : "warn");
  const node = el("span", `rna-call ${value == null ? "" : value ? "detected" : "absent"}`, value == null ? "?" : value ? "✓" : "×");
  node.title = label;
  node.setAttribute("role", "img");
  node.setAttribute("aria-label", label);
  return node;
}

function markerPair(before, after) {
  return el("span", "", markerCall(before, true), " → ", markerCall(after, true));
}

function comparisonTable(title, rows) {
  const cols = [column("Metric", (row) => row[0]), column("Original", (row) => row[1], (row) => row[1], true),
    column("Final", (row) => row[2], (row) => row[2], true)];
  return el("div", "compact description", staticTable(title, cols, rows));
}

function comparisonPanel(title, rows) {
  return panel(title, comparisonTable(title, rows));
}

function bases(value) {
  if (value == null) return "—";
  if (value >= 1_000_000) return `${num(value / 1_000_000)} Mb`;
  return value >= 1_000 ? `${num(value / 1_000)} kb` : basePairs(value);
}

function rnaPanel(title, bin) {
  const before = rna(bin.original);
  const after = rna(bin.final);
  return comparisonPanel(title, [
    ...markers.map(([label, key]) => [label, markerCall(before[key]), markerCall(after[key])]),
    ["tRNA amino acid types", rnaTypes(before.trna_types), rnaTypes(after.trna_types)],
    ["RNA verdict", verdict(before.status), verdict(after.status)],
  ]);
}

function transitionRna(bin) {
  const before = rna(bin.original);
  const after = rna(bin.final);
  return el("div", "",
    el("div", "", verdict(before.status), " → ", verdict(after.status),
      ...(rnaMismatch(bin) ? [" ", pill("Different settings", "warn")] : [])),
    el("div", "transition-markers",
      ...markers.map(([label, field]) => el("span", "", label, " ", markerPair(before[field], after[field]))),
      el("span", "", "tRNA types ", paired(before.trna_types, after.trna_types, rnaTypes))));
}

function qualityChart(report, transitions) {
  if (!transitions.length) return el("p", "notice description", "No HQ gains or losses to plot.");
  const rows = transitions.filter((bin) => bin.delta.completeness != null && bin.delta.contamination != null);
  if (!rows.length) return el("p", "notice description", "None of the HQ transitions has paired completeness and contamination measurements.");
  const plot = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  plot.setAttribute("viewBox", "0 0 910 310");
  plot.setAttribute("class", "chart");
  plot.setAttribute("role", "img");
  plot.setAttribute("aria-label", "Original and final bin completeness and contamination for HQ gains and losses. Open circles are originals, and filled circles are final.");
  const svg = (tag, fields, label, parent = plot) => {
    const node = document.createElementNS(plot.namespaceURI, tag);
    for (const [key, value] of Object.entries(fields)) node.setAttribute(key, value);
    if (label) node.textContent = label;
    parent.append(node);
    return node;
  };
  const minimum = rows.reduce((result, bin) => Math.min(result, bin.original.completeness, bin.final.completeness), 80);
  const left = Math.max(0, Math.floor(minimum / 10) * 10);
  const x = (value) => 66 + (value - left) * 820 / (100 - left);
  const maxContamination = rows.reduce((maximum, bin) => Math.max(maximum, bin.original.contamination, bin.final.contamination), 10);
  const top = Math.ceil(maxContamination / 2) * 2;
  const y = (value) => 260 - value * 238 / top;
  const line = (x1, y1, x2, y2, stroke, extra = {}) => svg("line", { x1, y1, x2, y2, stroke, ...extra });
  const labels = (x, y, label) => svg("text", { x, y, "text-anchor": "middle" }, label);
  const thresholds = isMimag(report) ? [90, 5] : report.hq_policy.mode === "thresholds"
    ? [report.hq_policy.completeness?.value, report.hq_policy.contamination?.value] : [];
  const hasThresholds = thresholds.length === 2 && thresholds.every(Number.isFinite);
  if (hasThresholds) {
    const [complete, contamination] = thresholds;
    const cutoff = Math.min(top, contamination);
    svg("rect", { x: x(Math.max(left, complete)), y: y(cutoff), width: x(100) - x(Math.max(left, complete)), height: y(0) - y(cutoff), fill: "#e7f5eb" });
  }
  for (let tick = left; tick <= 100; tick += (100 - left <= 50 ? 5 : 10)) {
    line(x(tick), y(0), x(tick), y(top), "#e4eaf2");
    labels(x(tick), 279, String(tick));
  }
  for (let i = 0; i <= 5; i++) {
    const tick = top * i / 5;
    line(x(left), y(tick), x(100), y(tick), "#e4eaf2");
    svg("text", { x: 55, y: y(tick) + 4, "text-anchor": "end" }, num(tick));
  }
  if (hasThresholds) {
    const [complete, contamination] = thresholds;
    const dashed = { "stroke-dasharray": "4 4" };
    if (complete >= left) line(x(complete), y(0), x(complete), y(top), "#69829b", dashed);
    if (contamination <= top) line(x(left), y(contamination), x(100), y(contamination), "#69829b", dashed);
  }
  const hitLines = svg("g", { "aria-hidden": "true" });
  const marks = svg("g", {});
  const highlight = svg("g", { "aria-hidden": "true", "pointer-events": "none" });
  function drawTransition(parent, { x1, y1, x2, y2, color }, active = false) {
    svg("line", { x1, y1, x2, y2, stroke: color, "stroke-width": active ? 3 : 1,
      opacity: active ? 1 : .65, "pointer-events": "none" }, undefined, parent);
    svg("circle", { cx: x1, cy: y1, r: active ? 6.5 : 5, fill: "white", stroke: color,
      "stroke-width": active ? 2 : 1 }, undefined, parent);
    svg("circle", { cx: x2, cy: y2, r: active ? 5.5 : 3.5, fill: color,
      ...(active ? { stroke: "white", "stroke-width": 1.5 } : {}) }, undefined, parent);
  }
  function drawEndpointLabels({ x1, y1, x2, y2, original, final }) {
    const width = 151, height = 58, gap = 10;
    const minX = x(left), maxX = x(100) - width;
    const minY = y(top), maxY = y(0) - height;
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
    const points = [[x1, y1], [x2, y2]];
    const clear = ([boxX, boxY]) => points.every(([px, py]) =>
      px < boxX - 6 || px > boxX + width + 6 || py < boxY - 6 || py > boxY + height + 6);
    const corners = [[minX, minY], [maxX, minY], [minX, maxY], [maxX, maxY]];
    const choices = (px, py) => [
      [px - width / 2, py - height - gap], [px - width / 2, py + gap],
      [px + gap, py - height / 2], [px - width - gap, py - height / 2],
      [px - width / 2, py - 2 * (height + gap)], [px - width / 2, py + height + 2 * gap],
      ...corners,
    ].map(([bx, by]) => [clamp(bx, minX, maxX), clamp(by, minY, maxY)]).filter(clear);
    const separate = ([ax, ay], [bx, by]) => Math.abs(ax - bx) >= width + 4 || Math.abs(ay - by) >= height + 4;
    const finalChoices = choices(x2, y2);
    const originalBox = choices(x1, y1).find((box) => finalChoices.some((other) => separate(box, other)));
    const finalBox = finalChoices.find((box) => separate(originalBox, box));
    for (const [[bx, by], title, quality] of [[originalBox, "Original", original], [finalBox, "Final", final]]) {
      const label = svg("g", { class: "transition-label" }, undefined, highlight);
      svg("rect", { x: bx, y: by, width, height, rx: 5 }, undefined, label);
      svg("text", { x: bx + 9, y: by + 16, class: "transition-label-title" }, title, label);
      svg("text", { x: bx + 9, y: by + 33 }, `Completeness ${pct(quality.completeness)}`, label);
      svg("text", { x: bx + 9, y: by + 49 }, `Contamination ${pct(quality.contamination)}`, label);
    }
  }
  function watchTransition(target, transition) {
    target.addEventListener("pointerenter", () => {
      marks.setAttribute("opacity", ".3");
      highlight.replaceChildren();
      drawTransition(highlight, transition, true);
      drawEndpointLabels(transition);
    });
    target.addEventListener("pointerleave", () => {
      marks.removeAttribute("opacity");
      highlight.replaceChildren();
    });
  }
  for (const { id, hq_transition: transition, original, final } of rows) {
    const color = transition === "gain" ? "#1766bf" : "#a44422";
    const title = `${id} · ${transition === "gain" ? "Gained HQ" : "Lost HQ"}`;
    const coordinates = { x1: x(original.completeness), y1: y(original.contamination),
      x2: x(final.completeness), y2: y(final.contamination) };
    const points = { ...coordinates, color, original, final };
    // A wider line under all the points keeps endpoints easy to pick in a dense plot.
    const hitLine = svg("line", { ...coordinates, stroke: "transparent", "stroke-width": 10,
      "pointer-events": "stroke" }, undefined, hitLines);
    svg("title", {}, title, hitLine);
    const group = svg("g", { "data-transition": transition }, undefined, marks);
    svg("title", {}, title, group);
    drawTransition(group, points);
    watchTransition(hitLine, points);
    watchTransition(group, points);
  }
  labels(475, 304, "Completeness (%)");
  svg("text", { x: 16, y: 139, transform: "rotate(-90 16 139)", "text-anchor": "middle" }, "Contamination (%)");
  const scroll = el("div", "chart-scroll", plot);
  scroll.tabIndex = 0;
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", "Quality transitions chart, scroll horizontally on narrow screens");
  const plotted = `${num(rows.length)} of ${num(transitions.length)} transitions plotted`;
  const omitted = rows.length < transitions.length ? " · Some transitions lack paired measurements" : "";
  return el("div", "", scroll, el("p", "footnote", `○ Original · ● Final · Blue: gained HQ · Rust: lost HQ · ${plotted}${omitted}${hasThresholds ? " · Shaded: completeness and contamination cutoffs" : ""}`));
}

function checkm2Changes(bins) {
  if (!bins.length) return panel("CheckM2 changes", el("p", "description", "No modified bins in this report."));
  const rows = qualityPairs.map(([label, metric]) => {
    const values = bins.map((bin) => bin.delta[metric]).filter(Number.isFinite).sort((a, b) => a - b);
    const up = values.filter((value) => value > 0).length;
    const down = values.filter((value) => value < 0).length;
    const same = values.length - up - down;
    const missing = bins.length - values.length;
    const median = values.length ? (values[Math.floor((values.length - 1) / 2)] + values[Math.floor(values.length / 2)]) / 2 : null;
    const counts = [`${num(up)} up`, `${num(down)} down`];
    if (same) counts.push(`${num(same)} unchanged`);
    if (missing) counts.push(`${num(missing)} unavailable`);
    return el("div", "", el("dt", "", label), el("dd", "",
      el("span", `change-count ${metric}`, counts.join(" · ")),
      el("span", "muted", median == null ? "Median unavailable" : `Median ${change(median)} pp`)));
  });
  return panel("CheckM2 changes", el("p", "chart-caption", `${num(bins.length)} modified bins`), el("dl", "change-summary", ...rows));
}

const proposalSeries = [["Approved", "approved"], ["Selected", "selected"]];
const rnaSeries = [["Gained", "gained"], ["Lost", "lost"], ["Unchanged", "unchanged"], ["Incomplete", "incomplete"]];

function barLegend(series) {
  return el("ul", "bar-legend", ...series.map(([label, kind]) =>
    el("li", "", el("span", `bar-fill ${kind}`), label)));
}

function barAxis(maximum) {
  const axis = el("div", "bar-axis", el("span", "", "0"), el("span", "", `${num(maximum)}`));
  axis.setAttribute("aria-hidden", "true");
  return axis;
}

function proposalOutcomes(report) {
  const byMethod = new Map(Object.values(report.methodArtifacts ?? report.comparison?.methods ?? {})
    .map(({ method }) => [method, { approved: 0, selected: 0 }]));
  for (const proposal of report.proposals) {
    if (!byMethod.has(proposal.method)) byMethod.set(proposal.method, { approved: 0, selected: 0 });
    const counts = byMethod.get(proposal.method);
    counts.approved += Number(proposal.approved);
    counts.selected += Number(proposal.selected);
  }
  if (!byMethod.size) return panel("Proposal outcomes by method", el("p", "description", "No proposal methods in this report."));
  const rows = [...byMethod].sort(([left, a], [right, b]) =>
    b.selected - a.selected || b.approved - a.approved || left.localeCompare(right, "en"));
  const mostApproved = rows.reduce((maximum, [, counts]) => Math.max(maximum, counts.approved), 0);
  const maximum = Math.max(1, Math.ceil(mostApproved / 5) * 5);
  const chart = el("ul", "bar-chart proposal-bars", ...rows.map(([method, counts]) => {
    const bars = proposalSeries.map(([label, kind]) => {
      const count = counts[kind];
      const fill = el("span", `bar-fill ${kind}`);
      fill.style.width = `${count / maximum * 100}%`;
      const bar = el("div", "proposal-bar", el("div", "bar-track", fill), el("span", "", num(count)));
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", `${label} ${num(count)}`);
      return bar;
    });
    return el("li", "", syntax(method), el("div", "proposal-bar-group", ...bars));
  }));
  return panel("Proposal outcomes by method", chart,
    el("div", "proposal-axis", barAxis(maximum)), barLegend(proposalSeries));
}

function rnaChanges(bins) {
  if (!bins.length) return panel("RNA changes by criterion", el("p", "description", "No modified bins in this report."));
  const rows = [...markers, [`tRNA ≥${trnaMinTypes} types`, "trna_types"]].map(([label, field]) => {
    const counts = { gained: 0, lost: 0, unchanged: 0, incomplete: 0 };
    for (const bin of bins) {
      const before = rna(bin.original)[field];
      const after = rna(bin.final)[field];
      if (before == null || after == null || bin.assessment_mismatches?.includes(field)) {
        counts.incomplete++;
        continue;
      }
      const passedBefore = field === "trna_types" ? before >= trnaMinTypes : before;
      const passedAfter = field === "trna_types" ? after >= trnaMinTypes : after;
      counts[passedBefore === passedAfter ? "unchanged" : passedAfter ? "gained" : "lost"]++;
    }
    const parts = rnaSeries.filter(([, kind]) => counts[kind] > 0).map(([name, kind]) => {
      const part = el("span", `bar-fill ${kind}`, el("span", "", num(counts[kind])));
      part.style.flexGrow = counts[kind];
      part.title = `${name} ${num(counts[kind])}`;
      return part;
    });
    const bar = el("div", "rna-bar bar-track", ...parts);
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", rnaSeries.map(([name, kind]) => `${name} ${num(counts[kind])}`).join(", "));
    return el("li", "", el("span", "", label), bar);
  });
  return panel("RNA changes by criterion", el("p", "chart-caption", `${num(bins.length)} modified bins`),
    el("ul", "bar-chart rna-bars", ...rows), el("div", "rna-axis", barAxis(bins.length)), barLegend(rnaSeries));
}

export function overview(report) {
  const s = report.summary;
  const hq = reportMode(report);
  const r = report.rna_summary;
  const mixed = report.bins.filter((row) => row.recipient && row.assessment_mismatches?.length
    && !(row.uncheckedCheckm2Settings ?? report.uncheckedCheckm2Settings)).length;
  const conflicts = report.bins.filter((row) => row.recipient && (row.original.hq_conflicts?.length || row.final.hq_conflicts?.length)).length;
  const transitions = report.bins.filter((bin) => bin.recipient && ["gain", "loss"].includes(bin.hq_transition));
  const modified = report.bins.filter((bin) => bin.recipient && bin.selected_proposal_id !== null);
  const method = (row) => {
    const proposal = selectedProposal(report, row);
    return proposal ? link(proposal.method, routeTo("proposals", proposal.id)) : "—";
  };
  return el("div", "",
    pageTitle("Bin refinement results", { badge: report.synthetic ? pill("Synthetic run", "info") : undefined }),
    ...(mixed ? [el("p", "notice warn audit-note", `${num(mixed)} recipient pairs have different assessment settings. Affected comparisons are withheld.`)] : []),
    ...(conflicts ? [el("p", "notice warn audit-note", `${num(conflicts)} recipient pairs have conflicting MIMAG HQ evidence. Their HQ transitions are unresolved.`)] : []),
    ...(report.uncheckedCheckm2Settings ? [el("p", "notice warn audit-note", report.sourceReports
      ? "Some source files predate pairwise CheckM2 setting checks and list different settings. Comparisons for their changed bins are withheld."
      : "This report lists different CheckM2 settings and predates pairwise setting checks. Comparisons for changed bins are withheld. Regenerate the report to identify affected pairs.")] : []),
    stats([["Original bins", num(s.original_bins)], ["Recipients", num(s.recipients), "", `${num(s.modified_bins)} modified`],
      ["Comparable pairs", `${num(s.comparable_pairs)} / ${num(s.recipients)}`], [`${hq} before`, num(s.original_hq)],
      [`${hq} after`, num(s.final_hq)], [`Net ${hq} change`, change(s.net_hq_change), s.net_hq_change > 0 ? "positive" : "",
        s.hq_gains == null ? "No comparable HQ pairs" : `${num(s.hq_gains)} gains · ${num(s.hq_losses)} losses`]]),
    el("div", "split", panel("HQ transitions plot", qualityChart(report, transitions)), el("div", "stack overview-sidebar",
      panel(isMimag(report) ? "MIMAG policy" : "HQ policy", el("p", "description", report.hq_policy.description),
        ...(isMimag(report) ? [el("p", "description notice", `5S, 16S and 23S rRNA · ≥${trnaMinTypes} distinct tRNA amino acid types`)] : [])),
      panel("RNA coverage", el("p", "description", r ? `${num(r.comparable_pairs)} / ${num(s.recipients)} comparable RNA pairs` : "No RNA evidence bundled in this report."),
        ...(r ? [el("p", "description notice warn", `${num(r.incomplete_pairs)} recipient pairs with incomplete RNA evidence`)] : [])),
      checkm2Changes(modified))),
    el("div", "split equal overview-outcomes", proposalOutcomes(report), rnaChanges(modified)),
    transitions.length
      ? headTable("HQ transitions", [binId, ...qualityPairs.map(([label, metric]) => percentPair(label, metric, true)),
        column("RNA checks", (row) => rna(row.final).status, transitionRna, true),
        column("Proposal method", (row) => selectedProposal(report, row)?.method, method),
        column(`${hq} transition`, (row) => row.hq_transition, (row) => hqBadge(row.hq_transition))], transitions)
      : panel("HQ transitions", el("p", "description", "No HQ gains or losses in this report.")),
  );
}

export function binsView(report) {
  const selectedMethod = (row) => selectedProposal(report, row)?.method ?? (row.recipient ? "Unchanged" : "Outside recipient set");
  const columns = [binId, ...qualityPairs.map(([label, metric]) => percentPair(label, metric)),
    column("Final RNA", (row) => rna(row.final).status, (row) => verdict(rna(row.final).status)),
    column(`${reportMode(report)} transition`, (row) => row.recipient ? row.hq_transition : null,
      (row) => row.recipient ? hqBadge(row.hq_transition) : "—"),
    column("Final selection", selectedMethod, (row) => row.selected_proposal_id ? syntax(selectedMethod(row)) : selectedMethod(row))];
  const finalQuality = {
    label: "Final quality",
    options: [...all, ["hq-rna", "HQ and RNA pass"], ["pass", "HQ pass"], ["fail", "Below HQ"], ["unknown", "Unknown HQ"]],
    test: (row, key) => key === "hq-rna"
      ? row.final.hq_status === "pass" && rna(row.final).status === "pass"
      : row.final.hq_status === key,
  };
  const filters = [search("Search bins", (row) => row.id), finalQuality,
    select("HQ transition", hqOptions, (row) => row.recipient ? row.hq_transition : null),
    select("Final RNA status", rnaOptions, (row) => rna(row.final).status),
    select("Final selection", [["selected", "Selected"], ["unchanged", "Unchanged"], ["outside", "Outside recipient set"]],
      (row) => row.selected_proposal_id ? "selected" : row.recipient ? "unchanged" : "outside"),
    select("Quality metrics", [["paired", "Both metric pairs available"], ["missing", "Missing metric values"]],
      (row) => row.delta.completeness != null && row.delta.contamination != null ? "paired" : "missing")];
  return el("div", "", pageTitle("Bins", { subtitle: `${num(report.summary.original_bins)} original bins · ${num(report.summary.recipients)} recipients` }),
    browserTable({ label: "Bins", columns, rows: report.bins, filters }));
}

const assemblyMetrics = [
  ["Genome size", "genome_size", bases], ["Contigs", "contig_count", num], ["N50", "contig_n50", bases],
  ["GC content", "gc_content", fraction], ["Coding density", "coding_density", fraction], ["Coding sequences", "coding_sequences", num],
  ["Average gene length", "average_gene_length", (value) => value == null ? "—" : `${num(value)} aa`],
  ["Translation table", "translation_table", num], ["Longest contig", "max_contig_length", bases],
];
const checkm2Metrics = [["Completeness model", "completeness_model"], ["General completeness", "completeness_general", pct],
  ["Specific completeness", "completeness_specific", pct], ["Model cosine similarity", "cosine_similarity", score], ["CheckM2 notes", "notes"]];

const proposalRanks = [column("Mean score", (row) => row.mean_added_score, (row) => score(row.mean_added_score)),
  column("Min score", (row) => row.min_added_score, (row) => score(row.min_added_score)),
  column("Size rank", (row) => row.within_size_rank), column("Priority rank", (row) => row.priority_rank), column("Tier", (row) => row.assessment_tier)]
  .map((item) => ({ ...item, scoped: true }));

function rangeTrack(first, second, minimum, maximum) {
  const position = (value) => minimum === maximum ? 50 : (value - minimum) / (maximum - minimum) * 100;
  const start = position(first);
  const end = position(second);
  const track = el("div", "range-track", el("span", "range-line"), el("span", "range-dot first"), el("span", "range-dot last"));
  track.style.setProperty("--range-first", `${start}%`);
  track.style.setProperty("--range-last", `${end}%`);
  track.style.setProperty("--range-left", `${Math.min(start, end)}%`);
  track.style.setProperty("--range-width", `${Math.abs(end - start)}%`);
  track.setAttribute("aria-hidden", "true");
  return track;
}

function rangeLegend(first, second) {
  return el("div", "range-legend",
    el("span", "", el("i", "range-dot first"), first),
    el("span", "", el("i", "range-dot last"), second));
}

function rangeAxis(values, format) {
  const ticks = values.map((value, index) => {
    const tick = el("span", "", format(value));
    tick.style.setProperty("--tick", `${values.length === 1 ? 50 : index / (values.length - 1) * 100}%`);
    return tick;
  });
  const axis = el("div", "range-axis", ...ticks);
  axis.setAttribute("aria-hidden", "true");
  return axis;
}

function binQualityPanel(bin) {
  const metrics = qualityPairs.map(([label, metric]) => {
    const original = bin.original[metric];
    const final = bin.final[metric];
    const delta = bin.delta[metric];
    const values = el("div", "quality-endpoints", el("span", "", `Original ${pct(original)}`), el("span", "", `Final ${pct(final)}`));
    const row = el("div", "bin-quality-row", el("div", "bin-card-line", el("h3", "", label),
      ...(delta == null ? [] : [deltaBadge(label, metric, delta)])), values);
    if (delta == null) {
      row.append(el("p", "chart-caption", bin.assessment_mismatches?.includes(metric)
        ? "Different assessment settings" : "Paired measurements unavailable"));
      return row;
    }
    const isComplete = metric === "completeness";
    const minimum = isComplete ? Math.min(70, Math.floor(Math.min(original, final) / 10) * 10) : 0;
    const maximum = isComplete ? 100 : Math.max(5, Math.ceil(Math.max(original, final) / 5) * 5);
    const step = isComplete ? (maximum - minimum <= 30 ? 5 : 10) : maximum / 5;
    const ticks = Array.from({ length: Math.round((maximum - minimum) / step) + 1 }, (_, index) => minimum + step * index);
    row.append(rangeTrack(original, final, minimum, maximum), rangeAxis(ticks, pct));
    return row;
  });
  return panel("CheckM2 before and after", rangeLegend("Original", "Final"), ...metrics);
}

function binRnaPanel(bin) {
  const before = rna(bin.original);
  const after = rna(bin.final);
  const types = [["Original", before.trna_types], ["Final", after.trna_types]].map(([label, count]) => {
    const cells = el("span", "trna-count-strip");
    cells.setAttribute("aria-hidden", "true");
    if (count != null) {
      cells.append(...Array.from({ length: 20 }, (_, index) => el("i", index < count ? "detected" : "")));
    }
    return el("div", "trna-count-row", el("span", "", label), cells, el("strong", "", rnaTypes(count)));
  });
  const rnaRows = markers.map(([label, field]) => [label, markerCall(before[field]), markerCall(after[field])]);
  return panel("RNA evidence", comparisonTable("rRNA evidence", rnaRows),
    el("div", "bin-trna", el("h3", "", "tRNA amino acid types"), ...types),
    el("div", "bin-card-line bin-rna-verdict", el("strong", "", "RNA verdict"),
      el("span", "", verdict(before.status), rnaMismatch(bin) ? " / " : " → ", verdict(after.status))),
    ...(rnaMismatch(bin) ? [el("p", "chart-caption", "Original and final RNA assessments use different settings.")] : []));
}

function selectedProposalPanel(selected) {
  if (!selected) return panel("Selected proposal", el("p", "description", "No proposal was selected for this bin."));
  const selectedMethod = methods[selected.method];
  return panel("Selected proposal", el("div", "bin-proposal-summary",
    el("div", "bin-method", link(selected.method, routeTo("proposals", selected.id)),
      ...(selectedMethod?.[0] ? [el("p", "chart-caption", selectedMethod[0])] : [])),
    stats([["Added contigs", num(selected.added_contig_count)], ["Added sequence", bases(selected.added_bp)]]),
    stats([["Priority rank", num(selected.priority_rank)], ["Size rank", num(selected.within_size_rank)], ["Tier", num(selected.assessment_tier)]]),
    stats([["Mean score", score(selected.mean_added_score)], ["Min score", score(selected.min_added_score)]]),
    link("Open proposal", routeTo("proposals", selected.id))));
}

function disclosure(label, ...children) {
  return el("details", "panel bin-disclosure", el("summary", "", label), ...children);
}

function methodArtifactOptions(rows) {
  const artifacts = [...new Map(rows.map(({ artifact_id, method }) => [artifact_id, method]))];
  return artifacts.map(([id, method]) =>
    [id, artifacts.some(([otherId, name]) => otherId !== id && name === method) ? `${method} · ${id.slice(0, 8)}` : method]);
}

function proposalRanking(bin, selected) {
  if (!bin.proposals.length) return panel("Proposal ranking within method", el("p", "description", "No proposals in this report for this bin."));
  const picker = el("select", "", ...methodArtifactOptions(bin.proposals).map(([id, caption]) => {
    const option = el("option", "", caption);
    option.value = id;
    return option;
  }));
  if (selected) picker.value = selected.artifact_id;
  const chart = el("div", "ranking-chart");
  function render() {
    const rows = bin.proposals.filter((proposal) => proposal.artifact_id === picker.value);
    const { minimum, maximum, evaluated } = rows.reduce((range, proposal) => ({
      minimum: Math.min(range.minimum, proposal.min_added_score, proposal.mean_added_score),
      maximum: Math.max(range.maximum, proposal.min_added_score, proposal.mean_added_score),
      evaluated: range.evaluated + Number(proposal.assessed),
    }), { minimum: Infinity, maximum: -Infinity, evaluated: 0 });
    const rankingColumns = [
      column("Priority rank", (row) => row.priority_rank, (row) => link(num(row.priority_rank), routeTo("proposals", row.id))),
      column("Membership score", (row) => row.mean_added_score, (row) => el("div", "ranking-score",
        rangeTrack(row.min_added_score, row.mean_added_score, minimum, maximum),
        el("span", "ranking-score-values", `${score(row.min_added_score)} → ${score(row.mean_added_score)}`))),
      column("Decision", decision, status),
    ];
    const ticks = minimum === maximum ? [minimum] : [minimum, minimum / 2 + maximum / 2, maximum];
    chart.replaceChildren(
      el("div", "bin-card-line ranking-caption", el("p", "chart-caption", `${num(rows.length)} proposals · ${num(evaluated)} evaluated`),
        rangeLegend("Min", "Mean")),
      browserTable({ label: "Proposal ranking", columns: rankingColumns, rows, selected: (row) => row.selected,
        afterRows: (_, table) => {
          table.rows[0].cells[1].append(el("div", "ranking-score ranking-axis", rangeAxis(ticks, score), el("span", "", "Min → Mean")));
        } }));
  }
  picker.addEventListener("change", render);
  render();
  return panel("Proposal ranking within method", el("label", "ranking-artifact", "Method artifact", picker), chart);
}

function contigLengths(selected) {
  const title = "Contigs added by the selected proposal";
  if (!selected) return panel(title, el("p", "description", "No proposal was selected for this bin."));
  const summary = el("div", "bin-card-line contig-chart-caption",
    el("p", "chart-caption", `${contigCount(selected.added_contig_count)} · ${bases(selected.added_bp)}`),
    link("Open proposal", routeTo("proposals", selected.id)));
  if (!selected.donors?.length) return panel(title, summary, el("p", "description", "Per-contig evidence is unavailable in this report."));
  const maximum = selected.donors.reduce((largest, donor) => Math.max(largest, donor.length_bp), 1);
  const columns = [
    column("Contig ID", (row) => row.id, (row) => syntax(row.id), true),
    column("Length", (row) => row.length_bp, (row) => {
      const fill = el("span", "bar-fill selected");
      fill.style.width = `${row.length_bp / maximum * 100}%`;
      const track = el("span", "bar-track", fill);
      track.setAttribute("aria-hidden", "true");
      const length = el("span", "", bases(row.length_bp));
      length.title = basePairs(row.length_bp);
      return el("div", "contig-length", track, length);
    }),
    column("Mean depth", (row) => row.mean_depth, (row) => `${num(row.mean_depth)}×`),
  ];
  return panel(title, summary, el("div", "compact contig-chart",
    browserTable({ label: "Added contig lengths", columns, rows: selected.donors, sortIndex: 1 })));
}

function proposalDrilldown(bin, selected) {
  const section = disclosure("Proposal ranking and added contigs");
  section.classList.add("proposal-drilldown");
  section.addEventListener("toggle", () => {
    if (section.open && section.children.length === 1) {
      section.append(el("div", "split bin-drilldown", proposalRanking(bin, selected), contigLengths(selected)));
    }
  });
  return section;
}

export function binDetail(report, bin) {
  const compare = (label, field, format = (value) => String(value ?? "—")) =>
    [label, format(bin.original.checkm2?.[field]), format(bin.final.checkm2?.[field])];
  const hasFields = (fields) => fields.some(([, field]) => bin.original.checkm2?.[field] != null || bin.final.checkm2?.[field] != null);
  const checkm2Details = (title, fields) => hasFields(fields)
    ? disclosure(title, comparisonTable(title, fields.map((field) => compare(...field))))
    : disclosure(`${title} · Unavailable in report`, el("p", "chart-caption", `This report has no saved ${title.toLowerCase()} for the bin.`));
  const selected = bin.proposals.find((row) => row.selected);
  const proposalQuality = qualityPairs.map(([label, metric]) => {
    const original = bin.original[metric];
    const measured = (row) => row.assessed ? row.quality[metric] : null;
    return column(label, measured, (row) => {
      const after = measured(row);
      return row.assessed && (original != null || after != null) ? paired(original, after) : "—";
    });
  });
  const columns = [column("Method", (row) => row.method, (row) => link(row.method, routeTo("proposals", row.id))), ...proposalRanks,
    column("Added", (row) => row.added_bp, (row) => `${contigCount(row.added_contig_count)} · ${bases(row.added_bp)}`),
    ...proposalQuality, column("Decision", decision, status)];
  return el("div", "", pageTitle(bin.id, { breadcrumb: link("Bins", "#bins"), badge: bin.recipient ? hqBadge(bin.hq_transition) : pill("Outside recipient set") }),
    ...hqConflict(bin.original, "Original bin"), ...hqConflict(bin.final, "Final bin"),
    ...(bin.assessment_mismatches?.length ? [el("p", "notice warn audit-note", (bin.uncheckedCheckm2Settings ?? report.uncheckedCheckm2Settings)
      ? "This report predates pairwise CheckM2 setting checks. Comparisons for this changed bin are withheld."
      : `Different assessment settings for ${bin.assessment_mismatches.map((field) => assessmentLabels[field]).join(", ")}. Affected comparisons are withheld.`)] : []),
    el("div", "trio bin-summary", binQualityPanel(bin), binRnaPanel(bin), selectedProposalPanel(selected)),
    el("div", "bin-disclosures", checkm2Details("CheckM2 details", checkm2Metrics), checkm2Details("Assembly metrics", assemblyMetrics),
      ...(bin.proposals.length ? [proposalDrilldown(bin, selected)] : []),
      ...(selected ? [addedContigEvidence(report, bin, selected)] : [])),
    el("section", "bin-proposals", el("h2", "section-title", `Proposals for ${bin.id}`),
      browserTable({ label: `Proposals for ${bin.id}`, columns, rows: bin.proposals, selected: (row) => row.selected,
        pinnedRow: selected, scope: (row) => row.artifact_id })),
    el("p", "footnote", "Quality values show original bin → evaluated proposal. Membership scores and ranks are scoped to the method artifact and recipient."));
}

export function proposalsView(report) {
  const rows = report.proposals;
  const recipients = [...new Set(rows.map((row) => row.recipient))].sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
  const columns = [column("Proposal", (row) => row.id, (row) => idLink("proposals", row.id)),
    column("Bin ID", (row) => row.recipient, (row) => idLink("bins", row.recipient)), column("Method", (row) => row.method, (row) => syntax(row.method)),
    column("Added contigs", (row) => row.added_contig_count), column("Added bp", (row) => row.added_bp, (row) => num(row.added_bp)),
    ...proposalRanks, column("Status", decision, status)];
  const counts = [rows.length, ...["assessed", "approved", "selected"].map((field) => rows.filter((row) => row[field]).length)];
  const preview = el("div", "");
  return el("div", "", pageTitle("Proposals"), stats(["Generated", "Evaluated", "Approved", "Selected"].map((label, i) => [label, num(counts[i])])),
    browserTable({ label: "Proposals and ranking", columns, rows, selected: (row) => row.selected,
      scope: (row) => `${row.artifact_id}\t${row.recipient}`,
      filters: [search("Search bins or proposals", (row) => `${row.recipient} ${row.id}`),
        select("Method artifact", methodArtifactOptions(rows), (row) => row.artifact_id),
        select("Recipient", recipients.map((id) => [id, id]), (row) => row.recipient), states("Assessment", (row) => row.assessed),
        states("Approved", (row) => row.approved), states("Selected", (row) => row.selected), states("Final export", (row) => row.exported)],
      afterRows: (visible) => {
        const row = visible.find((item) => item.selected);
        preview.replaceChildren(...(row ? [el("div", "panel stage-list", idLink("proposals", row.id),
          el("span", "", `Quality · ${pct(row.quality.completeness)} complete · ${pct(row.quality.contamination)} contamination`),
          el("span", "", "RNA · ", verdict(rna(row.quality).status)), status(row))] : []));
      } }), preview,
    el("p", "footnote", "Priority cycles through addition sizes. Compare scores and ranks within the same method artifact and recipient."));
}

export function proposalDetail(report, item) {
  const evidence = rna(item.quality);
  const columns = [column("Contig ID", (row) => row.id), column("Length", (row) => row.length_bp, (row) => basePairs(row.length_bp)),
    column("Mean depth", (row) => row.mean_depth, (row) => `${num(row.mean_depth)}×`), column("Depth variance", (row) => row.depth_variance, (row) => num(row.depth_variance)),
    column("Membership score", (row) => row.score, (row) => score(row.score)), column("Donor rank", (row) => row.rank)];
  return el("div", "", pageTitle(item.id, { breadcrumb: link("Proposals", "#proposals"), badge: status(item), subtitle: `${item.recipient} · ${item.method}` }),
    ...hqConflict(item.quality, "Proposal"),
    stats([["Mean added score", score(item.mean_added_score)], ["Min added score", score(item.min_added_score)],
      ["Within-size rank", num(item.within_size_rank)], ["Priority rank", num(item.priority_rank)], ["Tier", num(item.assessment_tier)]]),
    el("div", "split", panel("Added contigs", el("div", "description", staticTable("Added contigs", columns, item.donors ?? [])),
      el("p", "footnote", `${contigCount(item.added_contig_count)} · ${basePairs(item.added_bp)}`),
      el("p", "footnote", "Donor scores compare candidates within this method artifact and bin.")),
    panel("External assessment", details([["Completeness", pct(item.quality.completeness)], ["Contamination", pct(item.quality.contamination)],
      ...markers.map(([label, key]) => [label, markerCall(evidence[key])]), ["tRNA amino acid types", rnaTypes(evidence.trna_types)],
      ["RNA verdict", verdict(evidence.status)], [reportMode(report), verdict(item.quality.hq_status)]]))),
    el("div", "panel stage-list", ...[["Evaluated", item.assessed], ["Approved", item.approved], ["Selected", item.selected], ["Final export", item.exported]]
      .map(([label, value]) => pill(`${label} · ${yesNo(value)}`, value ? "good" : ""))),
    el("p", "footnote", idLink("bins", item.recipient)));
}

export function rnaView(report) {
  const s = report.rna_summary;
  const selected = el("div", "");
  selected.id = "inspector-rna-evidence";
  selected.tabIndex = -1;
  const announcement = el("p", "rna-selection");
  announcement.setAttribute("aria-live", "polite");
  let current;
  const choose = (row, table) => {
    current = row;
    for (const button of table?.querySelectorAll(".text-button") ?? []) {
      const active = button.textContent === row.id;
      button.setAttribute("aria-pressed", String(active));
      button.closest("tr").classList.toggle("selected", active);
    }
    const jump = link(row.id, `#${selected.id}`);
    jump.addEventListener("click", (event) => {
      event.preventDefault();
      selected.focus();
      selected.scrollIntoView({ block: "nearest" });
    });
    announcement.replaceChildren("Showing RNA evidence for ", jump);
    selected.replaceChildren(rnaPanel(`${row.id} RNA evidence`, row),
      ...(rnaMismatch(row) ? [el("p", "notice warn", "Original and final RNA evidence use different assessment settings.")] : []),
      el("p", "footnote", link(`Open ${row.id}`, routeTo("bins", row.id))));
  };
  const rnaBin = column("Bin ID", (row) => row.id, (row) => {
    const button = el("button", "text-button", row.id);
    button.type = "button";
    button.setAttribute("aria-controls", selected.id);
    button.setAttribute("aria-pressed", String(current === row));
    button.addEventListener("click", () => choose(row, button.closest("table")));
    return button;
  });
  const columns = [rnaBin, ...markers.map(([label, field]) => column(label, (row) => rna(row.final)[field] == null ? null : Number(rna(row.final)[field]),
    (row) => markerPair(rna(row.original)[field], rna(row.final)[field]))),
  column("tRNA amino acid types", (row) => rna(row.final).trna_types, (row) => paired(rna(row.original).trna_types, rna(row.final).trna_types, rnaTypes)),
  column("Final RNA verdict", (row) => rna(row.final).status, (row) => verdict(rna(row.final).status))];
  return el("div", "", pageTitle("RNA checks", { subtitle: "Compare original and final RNA evidence across recipient bins." }),
    stats([["Comparable RNA pairs", s ? `${num(s.comparable_pairs)} / ${num(report.summary.recipients)}` : "—"],
      ["Gained RNA pass", num(s?.gains)], ["Lost RNA pass", num(s?.losses)], ["Incomplete evidence", num(s?.incomplete_pairs)]]),
    el("div", "split", el("div", "", announcement,
      browserTable({ label: "RNA evidence — original to final", columns, rows: report.bins.filter((row) => row.recipient),
        filters: [search("Search bins", (row) => row.id), select("Final verdict", rnaOptions, (row) => rna(row.final).status),
          select("RNA transition", [["fail,pass", "Gained RNA pass"], ["pass,fail", "Lost RNA pass"], ["unknown", "Incomplete evidence"]],
            (row) => pairRna(row).includes("unknown") || rnaMismatch(row) ? "unknown" : pairRna(row).join(","))],
        afterRows: (rows, table) => {
          const focus = rows.includes(current) ? current : rows[0];
          if (focus) choose(focus, table);
          else {
            selected.replaceChildren();
            announcement.replaceChildren();
          }
        } }),
      el("div", "legend", "✓ Detected", "× Not detected", "? Not assessed")),
    el("div", "stack", selected, panel("RNA requirement", el("p", "notice description", "5S, 16S and 23S detected · ≥18 distinct tRNA amino acid types"),
      el("p", "footnote", "MIMAG HQ also requires >90% completeness and <5% contamination.")))));
}

function provenancePanels(report) {
  const p = report.provenance;
  const sources = [column("Tool", (row) => row.tool), column("Version", (row) => row.version), column("Input", (row) => row.format),
    column("Scope", (row) => row.scope), column("Completeness column", (row) => row.completeness_column),
    column("Settings", (row) => row.settings, (row) => row.settings, true), column("Source SHA-256", (row) => row.report_sha256, (row) => syntax(row.report_sha256), true),
    column("Export receipt SHA-256", (row) => row.export_receipt_sha256, (row) => syntax(row.export_receipt_sha256), true)];
  return el("div", "split", el("div", "stack",
    headTable("Assessment sources", sources, p.assessment_sources), panel("Ranking and reconciliation", details([
      ["Priority strategy", syntax(p.priority?.strategy)], ["Tier 1 limit", num(p.priority?.first_tier_limit)], ["Candidate inventory SHA-256", syntax(p.candidate_inventory_sha256)],
      ["Approval list SHA-256", syntax(p.approval_list_sha256)], ["Approved recipients", num(p.approved_recipient_union)],
      ["Selection optimum proved", yesNo(p.selection_optimal)], ["Selection upper bound", num(p.selection_upper_bound)],
      ["Canonical tie-break complete", yesNo(p.selection_canonical_tie_break)], ["Selection receipt SHA-256", syntax(p.selection_receipt_sha256)],
      ["Final export receipt SHA-256", syntax(p.final_export_receipt_sha256)],
    ]))), el("div", "stack", panel("Quality policy", el("p", "description", report.hq_policy.description),
      ...(isMimag(report) ? [el("p", "notice description", "5S, 16S and 23S rRNA · ≥18 distinct tRNA amino acid types")] : [])),
    panel("Report metadata", details([["Schema", num(report.schema_version)], ["BinComplete", p.bincomplete_version], ["Dataset identity", syntax(p.dataset_identity)],
      ["Comparable HQ pairs", `${num(report.summary.comparable_pairs)} / ${num(report.summary.recipients)}`]]))));
}

export function provenance(report) {
  const title = pageTitle("Assessment and provenance");
  if (!report.sourceReports) return el("div", "", title, provenancePanels(report));
  return el("div", "", title, ...report.sourceReports.map((source) => el("section", "source-report",
    el("h2", "section-title", `Dataset ${source.datasetNumber} · ${source.name}`), provenancePanels(source))));
}
