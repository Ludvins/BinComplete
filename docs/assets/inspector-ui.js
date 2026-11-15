const decimal = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
const signed = new Intl.NumberFormat("en", { maximumFractionDigits: 2, signDisplay: "exceptZero" });
const scoreFormat = new Intl.NumberFormat("en", { minimumFractionDigits: 3, maximumFractionDigits: 5 });
const collator = new Intl.Collator("en", { numeric: true });

export function el(name, className, ...children) {
  const node = document.createElement(name);
  if (className) node.className = className;
  node.append(...children);
  return node;
}

export function link(label, href) {
  const node = el("a", "", label);
  node.href = href;
  return node;
}

export const routeTo = (page, id) => `#${page}/${encodeURIComponent(id)}`;
export const num = (value) => value == null ? "—" : decimal.format(value);
export const change = (value) => value == null ? "—" : signed.format(value);
export const pct = (value) => value == null ? "—" : `${num(value)}%`;
export const score = (value) => value == null ? "—" : scoreFormat.format(value);
export const paired = (before, after, format = pct) => `${format(before)} → ${format(after)}`;
export const yesNo = (flag) => flag == null ? "Unknown" : flag ? "Yes" : "No";
export const basePairs = (value) => value == null ? "—" : `${num(value)} bp`;
export const rnaTypes = (value) => value == null ? "—" : `${value} / 20`;
export const matches = (value, query) => value.toLocaleLowerCase("en").includes(query.trim().toLocaleLowerCase("en"));

export function pill(text, tone = "") {
  return el("span", `pill ${tone}`, text);
}

export function verdict(status) {
  if (status === "pass") return pill("Pass", "good");
  if (status === "fail") return pill("Fail", "warn");
  return pill("Unknown");
}

const hqLabels = { gain: ["Gained HQ", "good"], loss: ["Lost HQ", "warn"], retained: ["Kept HQ", "good"], non_hq: ["Below HQ", ""], unknown: ["Unknown", ""] };

export function hqBadge(transition) {
  return pill(...hqLabels[transition]);
}

export function pageTitle(text, { subtitle, badge, breadcrumb } = {}) {
  const title = el("div", "page-title");
  if (breadcrumb) title.append(el("div", "breadcrumb", breadcrumb, el("span", "", "/"), text));
  const heading = el("h1", "", text);
  heading.tabIndex = -1;
  title.append(el("div", "title-line", heading, ...(badge ? [badge] : [])));
  if (subtitle) title.append(el("p", "subtitle", subtitle));
  return title;
}

export function stats(items) {
  return el("dl", "panel stats", ...items.map(([label, value, tone = "", note]) =>
    el("div", "", el("dt", "", label), el("dd", tone, value), ...(note ? [el("small", "", note)] : []))));
}

export function panel(title, ...children) {
  return el("section", "panel padded", el("h2", "", title), ...children);
}

export function details(entries) {
  return el("dl", "details-list", ...entries.map(([label, value]) =>
    el("div", "", el("dt", "", label), el("dd", "", value))));
}

export function column(label, value, display = (row) => String(value(row) ?? "—"), wrap = false) {
  return { label, value, display, wrap };
}

export function dataTable(label, columns, { sortable = false, onSort, sortIndex, direction = 1 } = {}) {
  const head = el("tr", "", ...columns.map((item, index) => {
    const th = el("th", "");
    th.scope = "col";
    if (sortable && item.sortable !== false) {
      const arrow = el("span", "", index === sortIndex ? direction === 1 ? "↑" : "↓" : "↕");
      arrow.setAttribute("aria-hidden", "true");
      const button = el("button", "sort", item.label, arrow);
      button.type = "button";
      button.addEventListener("click", () => onSort(index));
      th.append(button);
      if (index === sortIndex) th.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
    } else th.textContent = item.label;
    return th;
  }));
  const body = el("tbody", "");
  const table = el("table", "data-table", el("thead", "", head), body);
  table.setAttribute("aria-label", label);
  const wrapper = el("div", "table-scroll", table);
  wrapper.tabIndex = 0;
  return { wrapper, body };
}

function tableRows(body, columns, rows, selected, empty) {
  if (!rows.length) {
    const cell = el("td", "empty", empty);
    cell.colSpan = columns.length;
    body.replaceChildren(el("tr", "", cell));
    return;
  }
  body.replaceChildren(...rows.map((row) => el("tr", selected(row) ? "selected" : "",
    ...columns.map((item) => el("td", item.wrap ? "wrap" : "", item.display(row))))));
}

export function staticTable(label, columns, rows, selected = () => false) {
  const { wrapper, body } = dataTable(label, columns);
  tableRows(body, columns, rows, selected, "No saved records.");
  return wrapper;
}

function compare(left, right, direction) {
  if (left === right) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  return direction * (typeof left === "number" ? left - right : collator.compare(left, right));
}

export function browserTable({
  label, columns, rows, filters = [], selected = () => false, pinnedRow, sortIndex = 0, pageSize = 10, afterRows, scope,
}) {
  const pageable = pinnedRow ? rows.filter((row) => row !== pinnedRow) : rows;
  const pageLimit = pinnedRow ? Math.max(1, pageSize - 1) : pageSize;
  const controls = el("div", "filters");
  const content = el("div", "panel table-panel");
  const count = el("p", "");
  count.setAttribute("role", "status");
  const prev = el("button", "", "Previous");
  const next = el("button", "primary", "Next");
  const pager = el("div", "pager", count, prev, next);
  const section = el("section", "", controls, content, pager);
  controls.hidden = !filters.length;
  let page = 0;
  let direction = 1;

  const inputs = filters.map((filter) => {
    const control = el(filter.options ? "select" : "input", "");
    if (filter.options) {
      for (const [value, caption] of filter.options) {
        const option = el("option", "", caption);
        option.value = value;
        control.append(option);
      }
    } else {
      control.type = "search";
      control.placeholder = filter.placeholder ?? filter.label;
    }
    control.value = filter.initial ?? "";
    control.addEventListener(filter.options ? "change" : "input", () => { page = 0; render(); });
    controls.append(el("label", filter.options ? "" : "search", filter.label, control));
    return control;
  });

  function render(focusSort = false) {
    const values = inputs.map((control) => control.value);
    const filtered = pageable.filter((row) => filters.every((filter, index) => !values[index] || filter.test(row, values[index])));
    const get = columns[sortIndex].value;
    filtered.sort((left, right) => {
      const group = columns[sortIndex].scoped && scope ? compare(scope(left), scope(right), 1) : 0;
      return group || compare(get(left), get(right), direction);
    });
    const start = page * pageLimit;
    const pageRows = filtered.slice(start, start + pageLimit);
    const visible = pinnedRow ? [pinnedRow, ...pageRows] : pageRows;
    const { wrapper, body } = dataTable(label, columns, {
      sortable: true, sortIndex, direction,
      onSort: (index) => { direction = sortIndex === index ? -direction : 1; sortIndex = index; page = 0; render(true); },
    });
    tableRows(body, columns, visible, selected, "No records match these filters.");
    content.replaceChildren(wrapper);
    const range = `${num(start + 1)}–${num(start + pageRows.length)} of ${num(filtered.length)}`;
    if (pinnedRow) {
      count.textContent = filtered.length ? `1 pinned · ${range} others` : "1 pinned";
    } else {
      count.textContent = filtered.length ? range : "0 results";
    }
    prev.disabled = page === 0;
    next.disabled = start + pageRows.length >= filtered.length;
    if (focusSort) wrapper.querySelectorAll("th")[sortIndex].querySelector("button")?.focus();
    if (afterRows) afterRows(visible, wrapper.querySelector("table"));
  }
  prev.addEventListener("click", () => { page--; render(); });
  next.addEventListener("click", () => { page++; render(); });
  render();
  return section;
}
