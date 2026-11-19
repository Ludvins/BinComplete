import { prepareReport } from "./inspector-data.js";
import { example } from "./inspector-example.js";
import { mergeReports } from "./inspector-merge.js";
import { methodCatalog, methodComparison } from "./inspector-methods.js";
import { el, link, pageTitle } from "./inspector-ui.js";
import {
  binDetail, binsView, overview, proposalDetail, proposalsView, provenance, rnaView,
} from "./inspector-views.js";

const element = (id) => document.getElementById(`inspector-${id}`);
const routes = { overview, bins: binsView, proposals: proposalsView, rna: rnaView, methods: methodCatalog, provenance };
const reportName = element("report-name");
const statusText = element("status");
const progressBar = element("progress");
let report;
let openRequest = 0;
const listViews = new Map();

function locationView() {
  const [page, ...rest] = location.hash.slice(1).split("/");
  if (!Object.hasOwn(routes, page)) return ["overview", ""];
  try {
    return [page, decodeURIComponent(rest.join("/"))];
  } catch {
    return [page, ""];
  }
}

function render() {
  if (!report) return;
  const [page, id] = locationView();
  let view;
  if (id && (page === "bins" || page === "proposals")) {
    const row = (page === "bins" ? report.byBin : report.byProposal).get(id);
    if (row) view = (page === "bins" ? binDetail : proposalDetail)(report, row);
    else view = el("div", "", pageTitle("Record unavailable"), link(`Return to ${page}`, `#${page}`));
  } else {
    const comparison = page === "methods" && id === "comparison";
    const key = comparison ? "methods/comparison" : page;
    if (!listViews.has(key)) listViews.set(key, (comparison ? methodComparison : routes[page])(report));
    view = listViews.get(key);
  }
  for (const anchor of element("nav").querySelectorAll("a")) {
    if (anchor.getAttribute("href") === `#${page}`) anchor.setAttribute("aria-current", "page");
    else anchor.removeAttribute("aria-current");
  }
  element("view").replaceChildren(view);
  view.querySelector("h1")?.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

function status(message, { error = false, loading = false, completed = 0, total } = {}) {
  statusText.toggleAttribute("data-error", error);
  statusText.toggleAttribute("data-loading", loading);
  statusText.textContent = message;
  progressBar.hidden = !loading;
  if (total === undefined) progressBar.removeAttribute("value");
  else {
    progressBar.max = total;
    progressBar.value = completed;
  }
}

function paintStatus() {
  // Give the status a frame before parsing, merging, or rendering on the main thread.
  return new Promise((resolve) => {
    let frame = requestAnimationFrame(() => { frame = requestAnimationFrame(finish); });
    const fallback = setTimeout(finish, 100);
    function finish() {
      cancelAnimationFrame(frame);
      clearTimeout(fallback);
      resolve();
    }
  });
}

function openReport(data, names) {
  const useBookmark = !report && Object.hasOwn(routes, location.hash.slice(1).split("/")[0]);
  if (!report) element("main").before(element("feedback"));
  report = data;
  listViews.clear();
  reportName.textContent = names.length === 1 ? names[0] : `${names.length} files combined`;
  reportName.title = names.join("\n");
  reportName.hidden = false;
  element("welcome").hidden = true;
  element("view").hidden = false;
  element("nav").hidden = false;
  if (!useBookmark && location.hash !== "#overview") history.replaceState(null, "", "#overview");
  render();
}

async function readFiles(files) {
  if (!files.length) return;
  const request = ++openRequest;
  const opened = [];
  const errors = [];
  const multiple = files.length > 1;
  for (const [index, file] of files.entries()) {
    const message = multiple ? `Reading and validating file ${index + 1} of ${files.length}…` : `Reading ${file.name}…`;
    status(message, { loading: true, completed: index, total: multiple ? files.length : undefined });
    if (index === 0) {
      await paintStatus();
      if (request !== openRequest) return;
    }
    try {
      const contents = await file.text();
      if (request !== openRequest) return;
      opened.push({ name: file.name, data: prepareReport(JSON.parse(contents)) });
    } catch (error) {
      if (request !== openRequest) return;
      const message = error instanceof SyntaxError ? "Invalid JSON." : error.message;
      errors.push(`${file.name} — ${message}`);
    }
  }
  if (errors.length) {
    if (multiple) errors.unshift("Could not combine the selected files.");
    status(errors.join("\n"), { error: true });
    return;
  }
  let data = opened[0].data;
  if (multiple) {
    status(`Combining ${opened.length} reports…`, { loading: true });
    await paintStatus();
    if (request !== openRequest) return;
    try {
      data = mergeReports(opened);
    } catch (error) {
      status(`Could not combine the selected files.\n${error.message}`, { error: true });
      return;
    }
  }
  status("Building the report view…", { loading: true });
  await paintStatus();
  if (request !== openRequest) return;
  try {
    openReport(data, opened.map(({ name }) => name));
  } catch (error) {
    status(`Could not display the report.\n${error.message}`, { error: true });
    return;
  }
  status(multiple ? `Combined ${opened.length} reports` : `Opened ${opened[0].name}`);
}

const picker = element("file");
for (const id of ["open", "choose"]) element(id).addEventListener("click", () => picker.click());
picker.addEventListener("change", () => {
  const files = [...picker.files];
  picker.value = "";
  readFiles(files);
});
const drop = element("drop");
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.setAttribute("data-dragging", "");
});
drop.addEventListener("dragleave", () => drop.removeAttribute("data-dragging"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.removeAttribute("data-dragging");
  readFiles([...event.dataTransfer.files]);
});
element("example").addEventListener("click", () => {
  openRequest++;
  openReport(prepareReport(example), ["synthetic report"]);
  status("Opened synthetic report");
});
window.addEventListener("hashchange", () => {
  if (location.hash !== "#inspector-main") render();
});
