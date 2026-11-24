import { addedContigs } from "./inspector-contigs.js";
import { basePairs, el, link, num, panel, pct, pill, routeTo } from "./inspector-ui.js";

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return undefined;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function coveragePoints(report, bins) {
  const relative = [];
  const plotted = new Map();
  for (const { id, contig_evidence: evidence, selected_proposal_id: proposal } of bins) {
    if (!evidence) continue;
    const original = evidence.original;
    const added = addedContigs(evidence, report.byProposal.get(proposal));
    const medianGc = median(original.map((row) => row.gc_fraction));
    const medianDepth = median(original.map((row) => row.mean_depth));
    const points = [];
    for (const [kind, rows] of [["original", original], ["added", added]]) {
      for (const row of rows) {
        if (!Number.isFinite(row.gc_fraction) || !Number.isFinite(row.mean_depth)) continue;
        const point = { bin: id, kind, row, x: row.gc_fraction * 100, y: row.mean_depth };
        points.push(point);
        if (kind === "added" && Number.isFinite(medianGc) && medianDepth > 0) {
          const ratio = row.mean_depth / medianDepth;
          if (Number.isFinite(ratio)) relative.push({ ...point, x: (row.gc_fraction - medianGc) * 100, y: ratio });
        }
      }
    }
    if (points.length) plotted.set(id, { points, addedCount: added.length });
  }
  return { raw: [...plotted.values()].flatMap(({ points }) => points), relative, plotted };
}

function extent(values) {
  let min = Infinity, max = -Infinity;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return [min, max];
}

function paddedExtent(values, padding) {
  const [min, max] = extent(values);
  const margin = Math.max(padding, (max - min) * .045);
  return [min - margin, max + margin];
}

function ticks(min, max) {
  const spacing = (max - min) / 5;
  const power = 10 ** Math.floor(Math.log10(spacing));
  const scaled = spacing / power;
  const factor = [1, 2, 2.5, 5, 10].reduce((nearest, value) =>
    Math.abs(value - scaled) < Math.abs(nearest - scaled) ? value : nearest);
  const step = factor * power;
  const first = Math.ceil(min / step), last = Math.floor(max / step);
  return Array.from({ length: last - first + 1 }, (_, index) => (first + index) * step);
}

const depthValue = (value) => `${value > 0 && value < .01 ? value.toExponential(1) : num(value)}×`;

function depthAxis(values) {
  const logs = values.filter((value) => value > 0).map(Math.log10);
  if (!logs.length) return { ticks: [0], position: () => .06, positive: false, zero: true };
  const [min, max] = paddedExtent(logs, .09);
  const zero = values.includes(0);
  const bottom = zero ? .15 : 0;
  let labels = [];
  if (max - min < 1.15) {
    labels = ticks(10 ** min, 10 ** max);
  } else {
    const stride = Math.max(1, Math.ceil((max - min) / 5));
    const factors = max - min < 2.5 ? [1, 2, 5] : [1];
    for (let decade = Math.floor(min); decade <= max; decade += stride) {
      for (const factor of factors) {
        const log = decade + Math.log10(factor);
        if (log >= min && log <= max) labels.push(factor * 10 ** decade);
      }
    }
  }
  return {
    ticks: zero ? [0, ...labels] : labels,
    position: (value) => value === 0 ? .035 : bottom + (1 - bottom) * (Math.log10(value) - min) / (max - min),
    positive: true,
    zero,
  };
}

function gridLine(area, axis, position, label, reference = false) {
  const line = el("div", `coverage-grid-${axis}${reference ? " coverage-reference" : ""}`);
  line.style[axis === "x" ? "left" : "bottom"] = `${position * 100}%`;
  if (label !== undefined) line.append(el("span", "", label));
  area.append(line);
}

function scatter(title, points, { largest, pointDetails, marks, relative = false, sharedDepth, depthRange }) {
  const xValues = points.map(({ x }) => x);
  const yValues = points.map(({ y }) => y);
  if (relative) {
    xValues.push(0);
    yValues.push(1);
  }
  let [left, right] = paddedExtent(xValues, relative ? .5 : 1);
  if (!relative) {
    left = Math.max(0, left);
    right = Math.min(100, right);
  }
  const xPosition = (value) => (value - left) / (right - left);
  const yAxis = sharedDepth ?? depthAxis(yValues);
  const grid = el("div", "coverage-grid");
  grid.setAttribute("aria-hidden", "true");
  if (depthRange) {
    const [lower, upper] = depthRange.map(yAxis.position);
    const band = el("div", "coverage-depth-range");
    band.style.bottom = `${lower * 100}%`;
    band.style.height = `${(upper - lower) * 100}%`;
    grid.append(band);
  }
  for (const value of ticks(left, right)) gridLine(grid, "x", xPosition(value), num(value));
  for (const value of yAxis.ticks) gridLine(grid, "y", yAxis.position(value), depthValue(value));
  if (relative) {
    gridLine(grid, "x", xPosition(0), undefined, true);
    gridLine(grid, "y", yAxis.position(1), undefined, true);
  }
  const tooltip = el("div", "coverage-tooltip");
  tooltip.hidden = true;
  tooltip.setAttribute("aria-hidden", "true");
  const area = el("div", "coverage-plot", grid, tooltip);
  area.setAttribute("role", "group");
  area.setAttribute("aria-label", title);
  for (const point of points) {
    const { bin, kind, row, x, y } = point;
    const label = kind === "added" ? row.id : "Original contig";
    const values = `${pct(row.gc_fraction * 100)} GC · ${depthValue(row.mean_depth)} depth · ${basePairs(row.length_bp)}`;
    const comparison = relative ? `${num(x)} pp GC difference · ${depthValue(y)} original median depth` : "";
    const mark = el("button", `coverage-point ${kind}`);
    mark.type = "button";
    mark.tabIndex = -1;
    mark.dataset.coverageBin = bin;
    mark.setAttribute("aria-label", [bin, label, values, comparison].filter(Boolean).join(" · "));
    mark.style.left = `${xPosition(x) * 100}%`;
    mark.style.bottom = `${yAxis.position(y) * 100}%`;
    mark.style.setProperty("--coverage-point-size", `${5 + 10 * Math.sqrt(row.length_bp / largest)}px`);
    area.append(mark);
    if (!marks.has(bin)) marks.set(bin, []);
    marks.get(bin).push(mark);
    pointDetails.set(mark, { area, tooltip, bin, label, values, comparison });
  }

  const yLabel = relative ? "Depth ÷ original median (log scale)"
    : yAxis.positive ? "Mean read depth (log scale)" : "Mean read depth";
  const xLabel = relative ? "GC difference from original median (percentage points)" : "GC content (%)";
  const chart = el("div", relative ? "coverage-chart" : "coverage-chart coverage-absolute",
    el("p", "coverage-y-title", yLabel), area, el("p", "coverage-x-title", xLabel));
  return { chart, yAxis };
}

function plotLegend(yAxis, relative = false) {
  const legend = relative
    ? el("div", "legend coverage-legend", el("span", "coverage-median-key", "Original medians"))
    : el("div", "legend contig-legend coverage-legend",
      el("span", "original", "Original contigs"), el("span", "added", "Selected additions"), "Larger dots show longer contigs");
  if (yAxis.positive && yAxis.zero) legend.append(el("span", "", "Zero depth shown below the log scale"));
  return legend;
}

function binCoverageGrid(plotted, options, sharedDepth) {
  const grid = el("div", "coverage-bin-grid");
  const legend = plotLegend(sharedDepth);
  legend.append(el("span", "coverage-range-key", "Original depth range"), "Shared depth scale");
  for (const [id, { points }] of plotted) {
    const original = points.filter(({ kind }) => kind === "original");
    const depthRange = original.length > 1 ? extent(original.map(({ y }) => y)) : undefined;
    const { chart } = scatter(id, points, { ...options, sharedDepth, depthRange });
    const counts = `${num(original.length)} original · ${num(points.length - original.length)} added`;
    grid.append(el("section", "panel padded coverage-bin-card",
      el("h3", "", link(id, routeTo("bins", id))), el("p", "chart-caption", counts), chart));
  }
  return el("div", "coverage-bin-content", legend, grid);
}

function placeTooltip(mark, { area, tooltip, bin, label, values, comparison }) {
  tooltip.replaceChildren(el("strong", "", bin), el("span", "", label), el("span", "", values),
    ...(comparison ? [el("span", "", comparison)] : []));
  tooltip.hidden = false;
  const bounds = area.getBoundingClientRect();
  const dot = mark.getBoundingClientRect();
  const left = dot.right - bounds.left + 8;
  const above = dot.top - bounds.top - tooltip.offsetHeight - 8;
  tooltip.style.left = `${Math.max(0, Math.min(left, area.clientWidth - tooltip.offsetWidth))}px`;
  tooltip.style.top = `${above >= 0 ? above : dot.bottom - bounds.top + 8}px`;
}

export function coverageOverview(report, bins) {
  const { raw, relative, plotted } = coveragePoints(report, bins);
  if (!raw.length) {
    const empty = panel("GC and read depth", el("p", "description", bins.length
      ? "No paired contig GC and read depth are saved for the selected bins." : "No modified bins in this report."));
    return el("div", "overview-coverage", empty);
  }
  const largest = raw.reduce((length, { row }) => Math.max(length, row.length_bp), 1);
  const pointDetails = new Map();
  const marks = new Map();
  const options = { largest, pointDetails, marks };
  const rawTitle = "GC and depth across modified bins";
  const relativeTitle = "Additions relative to their recipient bin";
  const rawPlot = scatter(rawTitle, raw, options);
  const relativePlot = relative.length ? scatter(relativeTitle, relative, { ...options, relative: true }) : null;
  const relativePanel = relativePlot ? panel(relativeTitle, relativePlot.chart, plotLegend(relativePlot.yAxis, true))
    : panel(relativeTitle, el("p", "description", "No additions have both measurements and original medians with positive depth."));
  const list = el("div", "coverage-bin-list");
  const summary = el("summary", "coverage-control coverage-trigger", "Highlight a bin");
  const picker = el("details", "coverage-bins", summary, list);
  const buttons = new Map();
  for (const [id, { addedCount }] of plotted) {
    const button = el("button", "", el("span", "", id), el("span", "muted", `${num(addedCount)} added`));
    button.type = "button";
    button.dataset.coverageBin = id;
    button.setAttribute("aria-pressed", "false");
    list.append(button);
    buttons.set(id, button);
  }
  const initial = `${num(plotted.size)} of ${num(bins.length)} modified bins plotted`;
  const statusLabel = pill(initial);
  const statusCount = pill("");
  statusCount.hidden = true;
  const status = el("div", "coverage-status", statusLabel, statusCount);
  status.setAttribute("role", "status");
  const clear = el("button", "coverage-control", "Clear highlight");
  clear.type = "button";
  clear.hidden = true;
  const toolbar = el("div", "coverage-toolbar", status, el("div", "coverage-controls", clear, picker));
  const panels = el("div", "split equal coverage-panels",
    panel(rawTitle, rawPlot.chart, plotLegend(rawPlot.yAxis)), relativePanel);
  const perBin = el("details", "panel bin-disclosure coverage-by-bin",
    el("summary", "", "GC and read depth by modified bin", pill(`${num(plotted.size)} ${plotted.size === 1 ? "bin" : "bins"}`)));
  const section = el("div", "overview-coverage", toolbar, panels, perBin);
  section.setAttribute("role", "region");
  section.setAttribute("aria-label", "GC and read depth of selected additions");

  const target = (node) => {
    const mark = node instanceof Element ? node.closest("[data-coverage-bin]") : null;
    return section.contains(mark) ? mark : null;
  };
  let hovered, focused, selected, active, tooltipMark;
  perBin.addEventListener("toggle", () => {
    if (!perBin.open || perBin.children.length > 1) return;
    perBin.append(binCoverageGrid(plotted, options, rawPlot.yAxis));
    for (const point of marks.get(active) ?? []) point.classList.add("highlighted");
  });

  function refresh() {
    const mark = hovered ?? focused;
    const bin = mark?.dataset.coverageBin ?? selected;
    if (bin !== active) {
      for (const point of marks.get(active) ?? []) point.classList.remove("highlighted");
      for (const point of marks.get(bin) ?? []) point.classList.add("highlighted");
      active = bin;
      panels.classList.toggle("has-highlight", bin !== undefined);
      statusLabel.textContent = bin ?? initial;
      statusLabel.title = bin ?? "";
      statusLabel.classList.toggle("info", bin !== undefined);
      const count = plotted.get(bin)?.addedCount;
      statusCount.hidden = count === undefined;
      statusCount.textContent = count === undefined ? "" : `${num(count)} selected ${count === 1 ? "addition" : "additions"}`;
    }
    if (mark === tooltipMark) return;
    if (pointDetails.has(tooltipMark)) pointDetails.get(tooltipMark).tooltip.hidden = true;
    if (pointDetails.has(mark)) placeTooltip(mark, pointDetails.get(mark));
    tooltipMark = mark;
  }

  function select(bin) {
    buttons.get(selected)?.setAttribute("aria-pressed", "false");
    selected = bin;
    buttons.get(selected)?.setAttribute("aria-pressed", "true");
    clear.hidden = selected === undefined;
    refresh();
  }

  section.addEventListener("pointerover", (event) => {
    hovered = target(event.target);
    refresh();
  });
  section.addEventListener("pointerout", (event) => {
    hovered = target(event.relatedTarget);
    refresh();
  });
  section.addEventListener("focusin", (event) => {
    focused = event.target.matches(":focus-visible") ? target(event.target) : null;
    refresh();
  });
  section.addEventListener("focusout", () => {
    focused = null;
    refresh();
  });
  section.addEventListener("click", (event) => {
    const mark = target(event.target);
    if (!mark) return;
    const bin = mark.dataset.coverageBin;
    if (list.contains(mark)) {
      picker.open = false;
      hovered = focused = null;
      summary.focus();
    }
    select(selected === bin ? undefined : bin);
  });
  clear.addEventListener("click", () => {
    select(undefined);
    summary.focus();
  });
  section.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    picker.open = false;
    hovered = focused = null;
    select(undefined);
    summary.focus();
  });
  return section;
}
