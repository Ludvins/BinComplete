import { basePairs, column, el, num, panel, pct, staticTable } from "./inspector-ui.js";

async function readContig(file, contig, signal) {
  const sequence = [];
  let length = 0;
  let found = false;
  let tail = "";
  let finished = false;
  const readLine = (line) => {
    if (line.startsWith(">")) {
      if (found) return true;
      found = line.slice(1).trim().split(/\s+/, 1)[0] === contig.id;
    } else if (found) {
      const part = line.replace(/\s/g, "").toUpperCase();
      length += part.length;
      if (length > contig.length_bp || !/^[ACGTNRYKMSWBDHVU.\-]*$/.test(part)) {
        throw new Error("The FASTA record differs from the saved contig.");
      }
      sequence.push(part);
    }
    return false;
  };
  for await (const chunk of file.stream().pipeThrough(new TextDecoderStream(), { signal })) {
    const lines = (tail + chunk).split("\n");
    tail = lines.pop();
    if (lines.some(readLine)) {
      finished = true;
      break;
    }
  }
  if (!finished && tail) readLine(tail);
  if (!found) throw new Error(`No FASTA record named ${contig.id} was found.`);
  if (length !== contig.length_bp) throw new Error("The FASTA record length differs from the saved contig.");
  const bases = sequence.join("");
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(bases));
  const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  if (hash !== contig.sequence_sha256) throw new Error("The FASTA sequence differs from the saved contig.");
  return bases;
}

function fastaText(bases, contig, feature) {
  if (!feature) return `>${contig.id}\n${bases.match(/.{1,60}/g).join("\n")}\n`;
  const { start_bp: start, end_bp: end, strand } = feature;
  const slice = bases.slice(start - 1, end);
  const complement = { A: "T", T: "A", C: "G", G: "C", N: "N", R: "Y", Y: "R", K: "M", M: "K",
    S: "S", W: "W", B: "V", V: "B", D: "H", H: "D", U: "A", ".": ".", "-": "-" };
  const result = strand === "+" ? slice : Array.from(slice).reverse().map((base) => complement[base]).join("");
  return `>${contig.id}:${start}-${end}(${strand}) ${feature.feature.replace(/\s+/g, " ")}\n${result.match(/.{1,60}/g).join("\n")}\n`;
}

function sequenceViewer() {
  const input = el("input", "");
  input.type = "file";
  input.accept = ".fa,.fna,.fasta";
  const message = el("p", "chart-caption");
  message.setAttribute("role", "status");
  const sequence = el("textarea", "");
  sequence.readOnly = true;
  sequence.spellcheck = false;
  sequence.setAttribute("aria-label", "FASTA sequence");
  const copy = el("button", "", "Copy FASTA");
  copy.type = "button";
  const output = el("div", "contig-sequence-output", sequence, copy);
  output.hidden = true;
  const view = el("details", "contig-sequence", el("summary", "", "View sequence"),
    el("label", "", "Open a matching local FASTA", input), message, output);
  let contig, feature, file, cached, controller;
  async function render() {
    controller?.abort();
    const reading = new AbortController();
    controller = reading;
    output.hidden = true;
    sequence.value = "";
    if (!view.open || !file || !contig) {
      message.textContent = "";
      return;
    }
    message.textContent = "Reading local FASTA…";
    try {
      const bases = cached?.id === contig.id ? cached.bases : await readContig(file, contig, reading.signal);
      if (reading.signal.aborted) return;
      cached = { id: contig.id, bases };
      message.textContent = feature ? `${feature.feature} · ${feature.strand} strand` : contig.id;
      sequence.value = fastaText(bases, contig, feature);
      output.hidden = false;
    } catch (error) {
      if (!reading.signal.aborted) message.textContent = error.message;
    }
  }
  view.addEventListener("toggle", render);
  input.addEventListener("change", () => {
    file = input.files[0];
    cached = undefined;
    render();
  });
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(sequence.value);
      message.textContent = "Copied FASTA.";
    } catch {
      sequence.focus();
      sequence.select();
      message.textContent = "FASTA selected. Use your keyboard shortcut to copy it.";
    }
  });
  return { view, choose: (row, annotation) => { contig = row; feature = annotation; render(); } };
}

function scatterPanel(original, added, choose) {
  const points = [...original, ...added].filter((row) => row.gc_fraction != null);
  const additions = new Set(added);
  const buttons = new Map();
  if (!points.length) return [panel("Original vs. added contigs", el("p", "description", "No contigs have a GC measurement.")), buttons];
  let lowest = 1, highest = 0, depth = 0, length = 1;
  for (const row of points) {
    lowest = Math.min(lowest, row.gc_fraction);
    highest = Math.max(highest, row.gc_fraction);
    depth = Math.max(depth, row.mean_depth);
    length = Math.max(length, row.length_bp);
  }
  const right = Math.min(1, (Math.floor(highest * 20) + 1) / 20);
  const left = Math.max(0, Math.min(Math.floor(lowest * 20) / 20, right - .05));
  const top = Math.max(5, Math.ceil(depth / 5) * 5);
  const plot = el("div", "contig-scatter-plot");
  for (const row of points) {
    const isAdded = additions.has(row);
    const point = el(isAdded ? "button" : "span", "contig-point");
    point.style.left = `${(row.gc_fraction - left) / (right - left) * 100}%`;
    point.style.bottom = `${row.mean_depth / top * 100}%`;
    point.style.setProperty("--point-size", `${5 + 11 * Math.sqrt(row.length_bp / length)}px`);
    if (isAdded) {
      point.type = "button";
      point.title = `${row.id} · ${basePairs(row.length_bp)} · ${pct(row.gc_fraction * 100)} GC · ${num(row.mean_depth)}×`;
      point.setAttribute("aria-label", point.title);
      point.addEventListener("click", () => choose(row));
      buttons.set(row.id, point);
    }
    plot.append(point);
  }
  const ticks = (start, end, format) => Array.from({ length: 5 }, (_, index) =>
    el("span", "", format(start + (end - start) * index / 4)));
  const graph = el("div", "contig-scatter", el("div", "contig-axis-title", "Mean depth (×)"),
    el("div", "contig-y-ticks", ...ticks(top, 0, num)), plot,
    el("div", "contig-x-ticks", ...ticks(left, right, (value) => pct(value * 100))),
    el("p", "contig-x-title", "GC content (%)"));
  const legend = el("div", "legend contig-legend",
    el("span", "original", "Original contigs"), el("span", "added", "Added contigs"), "Larger dots show longer contigs");
  return [panel("Original vs. added contigs", graph, legend), buttons];
}

export function addedContigEvidence(report, bin, selected) {
  const section = el("details", "panel bin-disclosure contig-evidence",
    el("summary", "", "Added contig evidence"));
  section.addEventListener("toggle", () => {
    if (!section.open || section.children.length > 1) return;
    if (!bin.contig_evidence) {
      section.append(el("p", "description", "Export a report with contig evidence to see GC, depth and saved RNA calls here."));
      return;
    }
    const { original, added: metadata, rna_features: features } = bin.contig_evidence;
    const byId = new Map(selected.donors.map((donor) => [donor.id, donor]));
    const added = metadata.map(({ id, gc_fraction, sequence_sha256 }) => ({ ...byId.get(id), gc_fraction, sequence_sha256 }));
    const contigs = new Map(added.map((row) => [row.id, row]));
    const details = el("div", "selected-contig");
    const viewer = sequenceViewer();
    const picker = el("select", "", ...added.map((row) => {
      const option = el("option", "", row.id);
      option.value = row.id;
      return option;
    }));
    const featureButtons = new Map();
    const [scatter, dots] = scatterPanel(original, added, choose);
    function choose(row, active = features.find((call) => call.contig_id === row.id)) {
      picker.value = row.id;
      for (const [id, dot] of dots) dot.setAttribute("aria-pressed", String(id === row.id));
      for (const [call, button] of featureButtons) {
        button.setAttribute("aria-pressed", String(call === active));
        button.closest("tr").classList.toggle("selected", call === active);
      }
      const numbers = `${basePairs(row.length_bp)} · ${num(row.mean_depth)}× mean depth${row.gc_fraction == null ? "" : ` · ${pct(row.gc_fraction * 100)} GC`}`;
      const track = features.filter((call) => call.contig_id === row.id).map((call) => {
        const marker = el("span", `contig-feature-mark ${call.strand === "-" ? "reverse" : ""}`);
        marker.style.left = `${(call.start_bp - 1) / row.length_bp * 100}%`;
        marker.style.width = `${(call.end_bp - call.start_bp + 1) / row.length_bp * 100}%`;
        const label = el("span", "", `${call.feature} · ${num(call.start_bp)}–${num(call.end_bp)} bp (${call.strand})`);
        return el("div", "contig-feature", label, el("div", "bar-track", marker));
      });
      const summary = el("div", "bin-card-line", el("p", "chart-caption", numbers));
      if (active) {
        const whole = el("button", "text-button", "Select whole contig");
        whole.type = "button";
        whole.addEventListener("click", () => { choose(row, null); viewer.view.querySelector("summary").focus(); });
        summary.append(whole);
      }
      details.replaceChildren(summary, ...track,
        ...(track.length ? [el("div", "contig-track-scale", el("span", "", "0"), el("span", "", basePairs(row.length_bp)))] : []));
      viewer.choose(row, active);
    }
    picker.addEventListener("change", () => choose(contigs.get(picker.value)));
    const rnaColumns = [
      column("Feature", (row) => row.feature, (row) => {
        const button = el("button", "text-button", row.feature);
        button.type = "button";
        button.addEventListener("click", () => choose(contigs.get(row.contig_id), row));
        featureButtons.set(row, button);
        return button;
      }),
      column("Contig", (row) => row.contig_id, undefined, true),
      column("Coordinates", (row) => row.start_bp, (row) => `${num(row.start_bp)}–${num(row.end_bp)} bp (${row.strand})`),
    ];
    const source = report.provenance.contig_rna_source;
    const annotation = panel("RNA calls on added contigs",
      ...(source ? [el("p", "chart-caption", `${source.tool} · ${source.version}`)] : []),
      features.length ? el("div", "compact description", staticTable("RNA calls on added contigs", rnaColumns, features))
        : el("p", "description", "No per-contig RNA calls are saved in this report."),
      el("label", "ranking-artifact", "Selected contig", picker), details, viewer.view);
    section.append(el("div", "split equal contig-evidence-panels", scatter, annotation));
    choose(added[0]);
  });
  return section;
}
