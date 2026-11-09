// Small shared helpers: escaping, formatting, fetching.

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const q = (s) => encodeURIComponent(s ?? "");
export const fileUrl = (p) => `/files/${String(p).split("/").map(encodeURIComponent).join("/")}`;
export const num = (v, d = 1) => (v === null || v === undefined || v === "" || Number.isNaN(Number(v)) ? "" : Number(v).toFixed(d));
export const when = (t) => (t ? new Date(typeof t === "number" ? t * 1000 : t).toLocaleString() : "");
export const size = (b) =>
  b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : b > 1e3 ? (b / 1e3).toFixed(0) + " kB" : (b ?? 0) + " B";
export const pct = (k, n) => (n ? Math.round((100 * k) / n) : 0);
export const STATUS = { PASS: "ok", FAIL: "bad", WARN: "warn", BLOCKED: "warn", INTERRUPTED: "bad", SKIP: "muted" };

export async function api(url, opts) {
  const response = await fetch(url, opts);
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text };
  }
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

export const params = () => new URLSearchParams((location.hash.split("?")[1] || ""));

export function pretty(value, limit = 150000) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return text && text.length > limit ? text.slice(0, limit) + `\n... [${text.length - limit} more characters: download the file]` : text ?? "";
}

export function mmss(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "";
  const m = Math.floor(seconds / 60);
  const s = seconds - 60 * m;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
