import { prepareReport } from "./inspector-data.js";
import { example } from "./inspector-example.js";
import { el, link, pageTitle } from "./inspector-ui.js";
import {
  binDetail, binsView, methodCatalog, methodComparison, overview, proposalDetail, proposalsView, provenance, rnaView,
} from "./inspector-views.js";

const element = (id) => document.getElementById(`inspector-${id}`);
const routes = { overview, bins: binsView, proposals: proposalsView, rna: rnaView, methods: methodCatalog, provenance };
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

function status(message, error = false) {
  const node = element("status");
  node.toggleAttribute("data-error", error);
  node.textContent = message;
}

function openReport(data, name) {
  const opened = prepareReport(data);
  const useBookmark = !report && Object.hasOwn(routes, location.hash.slice(1).split("/")[0]);
  report = opened;
  listViews.clear();
  element("welcome").hidden = true;
  element("view").hidden = false;
  element("nav").hidden = false;
  status(`Opened ${name}`);
  element("report-name").textContent = name;
  element("report-name").title = name;
  element("report-name").hidden = false;
  if (!useBookmark && location.hash !== "#overview") history.replaceState(null, "", "#overview");
  render();
}

async function readFile(file) {
  if (!file) return;
  const request = ++openRequest;
  status(`Opening ${file.name}…`);
  try {
    const data = JSON.parse(await file.text());
    if (request === openRequest) openReport(data, file.name);
  } catch (error) {
    if (request === openRequest) {
      status(error instanceof SyntaxError ? "The selected file contains invalid JSON." : error.message, true);
    }
  }
}

const picker = element("file");
for (const id of ["open", "choose"]) element(id).addEventListener("click", () => picker.click());
picker.addEventListener("change", () => {
  const file = picker.files[0];
  picker.value = "";
  readFile(file);
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
  if (event.dataTransfer.files.length === 1) readFile(event.dataTransfer.files[0]);
  else status("Drop one BinComplete report JSON file.", true);
});
element("example").addEventListener("click", () => {
  openRequest++;
  openReport(example, "synthetic report");
});
window.addEventListener("hashchange", () => {
  if (location.hash !== "#inspector-main") render();
});
