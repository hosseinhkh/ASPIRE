// Shared UI pieces: tabs, JSON viewer, badges, success bars, download buttons, lightbox.
import { esc, q, fileUrl, pct } from "./util.js";

export function badge(value, { yes = "yes", no = "no" } = {}) {
  if (value === true) return `<span class="badge ok">${yes}</span>`;
  if (value === false) return `<span class="badge bad">${no}</span>`;
  return `<span class="badge muted">-</span>`;
}

export function statusBadge(status) {
  const cls = { PASS: "ok", FAIL: "bad", WARN: "warn", BLOCKED: "warn", INTERRUPTED: "bad", SKIP: "muted" }[status] || "muted";
  return `<span class="badge ${cls}">${esc(status)}</span>`;
}

export function rate(k, n) {
  if (!n) return `<span class="muted">-</span>`;
  const p = pct(k, n);
  const cls = p >= 80 ? "ok" : p >= 40 ? "warn" : "bad";
  return `<span class="rate"><span class="bar"><span class="fill ${cls}" style="width:${p}%"></span></span><b>${k}/${n}</b> <span class="muted">${p}%</span></span>`;
}

export function downloads(path, label = "Download") {
  return `<span class="btn-group"><a class="btn" href="/download?path=${q(path)}&media=1" title="Everything, including videos and images">${esc(label)}</a><a class="btn" href="/download?path=${q(path)}&media=0" title="Logs, traces and JSON only">Logs only</a></span>`;
}

export function card(label, value, sub = "") {
  return `<div class="card"><div class="card-label">${esc(label)}</div><div class="card-value">${value}</div>${sub ? `<div class="card-sub">${sub}</div>` : ""}</div>`;
}

// Tabs: returns markup; call bindTabs after inserting it.
export function tabs(id, names, active) {
  return `<div class="tabs" id="${id}">${names.map((n) => `<button data-tab="${esc(n)}" class="${n === active ? "on" : ""}">${esc(n)}</button>`).join("")}</div><div id="${id}-body" class="tab-body"></div>`;
}
export function bindTabs(id, render, after = () => {}) {
  const box = document.getElementById(id);
  if (!box) return;
  const show = (button) => {
    box.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === button));
    document.getElementById(`${id}-body`).innerHTML = render(button.dataset.tab);
    bindZoom();
    after(button.dataset.tab);
  };
  box.querySelectorAll("button").forEach((b) => (b.onclick = () => show(b)));
  show(box.querySelector("button.on") || box.querySelector("button"));
}

// Collapsible JSON tree. Arrays of numbers and {"$array_inline": [...]} stay on one line.
export function jsonView(value, { open = 1, maxItems = 60 } = {}) {
  const render = (v, depth, key) => {
    const label = key === undefined ? "" : `<span class="j-key">${esc(key)}</span>: `;
    if (v === null || v === undefined) return `<div>${label}<span class="j-null">null</span></div>`;
    if (typeof v === "string") return `<div>${label}<span class="j-str">${esc(JSON.stringify(v).slice(0, 4000))}</span></div>`;
    if (typeof v === "number" || typeof v === "boolean") return `<div>${label}<span class="j-num">${esc(v)}</span></div>`;
    if (v && typeof v === "object" && Array.isArray(v.$array_inline)) {
      const values = v.$array_inline.map((x) => (typeof x === "number" ? +x.toFixed(4) : x));
      return `<div>${label}<span class="j-num">[${esc(values.slice(0, maxItems).join(", "))}${values.length > maxItems ? ", ..." : ""}]</span> <span class="muted">${esc(v.dtype || "")} ${esc(JSON.stringify(v.shape || ""))}</span></div>`;
    }
    if (Array.isArray(v) && v.every((x) => typeof x === "number")) {
      return `<div>${label}<span class="j-num">[${esc(v.slice(0, maxItems).map((x) => +x.toFixed(4)).join(", "))}${v.length > maxItems ? ", ..." : ""}]</span></div>`;
    }
    const entries = Array.isArray(v) ? v.map((x, i) => [i, x]) : Object.entries(v);
    const summary = Array.isArray(v) ? `[${v.length}]` : `{${entries.length}}${v.$type ? " " + esc(v.$type) : ""}`;
    const shown = entries.slice(0, maxItems).map(([k, x]) => render(x, depth + 1, k)).join("");
    const more = entries.length > maxItems ? `<div class="muted">... ${entries.length - maxItems} more</div>` : "";
    return `<details ${depth < open ? "open" : ""}><summary>${label}<span class="muted">${summary}</span></summary><div class="j-indent">${shown}${more}</div></details>`;
  };
  return `<div class="json">${render(value, 0)}</div>`;
}

export function bindZoom(root = document) {
  root.querySelectorAll("img[data-zoom]").forEach((img) => {
    img.onclick = () => {
      document.getElementById("lightbox-img").src = img.src;
      document.getElementById("lightbox-caption").textContent = img.alt || "";
      document.getElementById("lightbox").classList.add("on");
    };
  });
}

export const img = (path, alt = "", cls = "") => `<img data-zoom class="${cls}" loading="lazy" src="${fileUrl(path)}" alt="${esc(alt)}">`;
