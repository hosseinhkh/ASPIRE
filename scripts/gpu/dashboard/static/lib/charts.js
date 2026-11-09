// Inline SVG charts: line charts (with stage bands and a moving marker) and a Gantt track.
import { esc } from "./util.js";

export const PALETTE = ["#3b6fd8", "#d64545", "#1f9d55", "#c27c0e", "#8a63d2", "#0f8b8d", "#d4447c", "#6b7385"];

// colorKey(name) groups series that share a colour (e.g. "j0" and "j0 cmd"); dash(name) draws one dashed.
export function lineChart(series, title, { width = 900, height = 190, bands = [], marker = null, x = "", id = "", colorKey = null, dash = null } = {}) {
  const names = Object.keys(series).filter((k) => series[k] && series[k].length);
  if (!names.length) return "";
  const all = names.flatMap((k) => series[k]);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  }
  const n = Math.max(...names.map((k) => series[k].length));
  const L = 50, T = 22, B = 18, W = width - L - 12, H = height - T - B;
  const X = (i) => L + W * (i / Math.max(n - 1, 1));
  const Y = (v) => T + H * (1 - (v - lo) / (hi - lo));
  const keys = [...new Set(names.map((k) => (colorKey ? colorKey(k) : k)))];
  const colorOf = (k) => PALETTE[keys.indexOf(colorKey ? colorKey(k) : k) % 8];
  const legendRows = Math.ceil(keys.length / 4);
  let s = `<svg class="chart" ${id ? `id="${id}"` : ""} viewBox="0 0 ${width} ${height + 16 * legendRows}" data-l="${L}" data-w="${W}" data-n="${n}">`;
  s += `<text x="${L}" y="14" class="c-title">${esc(title)}</text>`;
  s += `<text x="6" y="${T + 9}" class="c-axis">${+hi.toPrecision(3)}</text><text x="6" y="${T + H}" class="c-axis">${+lo.toPrecision(3)}</text>`;
  s += `<line x1="${L}" y1="${T + H}" x2="${L + W}" y2="${T + H}" class="c-grid"/>`;
  bands.forEach((b, i) => {
    const x0 = X(b.start), x1 = X(b.end);
    s += `<rect x="${x0}" y="${T}" width="${Math.max(x1 - x0, 1)}" height="${H}" class="band ${i % 2 ? "b2" : "b1"}"><title>${esc(b.title)}</title></rect>`;
    if (b.label) s += `<text x="${x0 + 2}" y="${T + 10}" class="c-band">${esc(b.label)}</text>`;
  });
  names.forEach((k) => {
    const ys = series[k];
    const pts = ys.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    s += `<polyline fill="none" stroke="${colorOf(k)}" stroke-width="1.5" ${dash && dash(k) ? 'stroke-dasharray="5 4"' : ""} points="${pts}"/>`;
  });
  keys.forEach((k, j) => {
    const lx = L + (j % 4) * 210, ly = height + 10 + 16 * Math.floor(j / 4);
    s += `<rect x="${lx}" y="${ly - 9}" width="10" height="10" rx="2" fill="${PALETTE[j % 8]}"/><text x="${lx + 14}" y="${ly}" class="c-legend">${esc(k)}</text>`;
  });
  if (x) s += `<text x="${L + W}" y="${T + H + 13}" text-anchor="end" class="c-axis">${esc(x)}</text>`;
  s += `<line class="marker" x1="${marker === null ? -10 : X(marker)}" x2="${marker === null ? -10 : X(marker)}" y1="${T}" y2="${T + H}"/>`;
  return s + "</svg>";
}

// Move the vertical marker of every chart with data-n to a fraction (0..1) of its x range.
export function moveMarkers(root, fraction) {
  root.querySelectorAll("svg.chart").forEach((svg) => {
    const L = +svg.dataset.l, W = +svg.dataset.w;
    const line = svg.querySelector("line.marker");
    if (line) {
      const x = L + W * Math.min(Math.max(fraction, 0), 1);
      line.setAttribute("x1", x);
      line.setAttribute("x2", x);
    }
  });
}

// Horizontal bars over a time axis. items: {start, end, label, cls, title, key, lane}
// Overlapping bars in a lane get their own sub-rows, so every bar stays visible and clickable.
export function gantt(items, { width = 900, rowHeight = 16, total = null, lanes = null, maxRows = 8 } = {}) {
  if (!items.length) return "";
  const end = total ?? Math.max(...items.map((i) => i.end), 1);
  const laneNames = lanes ?? [...new Set(items.map((i) => i.lane ?? ""))];
  const L = 110, W = width - L - 10, top = 8, gap = 8;
  const X = (t) => L + (W * t) / Math.max(end, 1e-6);
  const minWidth = 4; // px: instantaneous calls still get a clickable sliver
  const placed = new Map();
  const laneRows = laneNames.map((lane) => {
    const inLane = items.filter((i) => (i.lane ?? "") === lane);
    // Calls that take no simulated time (state reads, perception) share one row of ticks;
    // calls that move the robot are packed below it so overlapping ones stay visible.
    const instant = inLane.filter((i) => X(i.end) - X(i.start) < 1);
    const timed = inLane.filter((i) => X(i.end) - X(i.start) >= 1);
    instant.forEach((item) => placed.set(item, { row: 0, x0: X(item.start) - 1, w: 3 }));
    const offset = instant.length ? 1 : 0;
    const rowEnds = [];
    timed.sort((a, b) => a.start - b.start || b.end - a.end).forEach((item) => {
      const x0 = X(item.start), x1 = Math.max(X(item.end), x0 + minWidth);
      let row = rowEnds.findIndex((e) => e <= x0 + 0.5);
      if (row < 0) row = rowEnds.length < maxRows ? rowEnds.length : rowEnds.indexOf(Math.min(...rowEnds));
      rowEnds[row] = x1 + 1;
      placed.set(item, { row: row + offset, x0, w: x1 - x0 });
    });
    return Math.max(rowEnds.length + offset, 1);
  });
  const laneTop = [];
  let y = top;
  laneRows.forEach((rows) => { laneTop.push(y); y += rows * rowHeight + gap; });
  const height = y + 16;
  let s = `<svg class="gantt" viewBox="0 0 ${width} ${height}" data-l="${L}" data-w="${W}" data-total="${end}">`;
  laneNames.forEach((lane, r) => {
    s += `<text x="4" y="${laneTop[r] + 12}" class="c-legend">${esc(lane)}</text><rect x="${L}" y="${laneTop[r]}" width="${W}" height="${laneRows[r] * rowHeight}" class="lane"/>`;
  });
  items.forEach((item) => {
    const r = Math.max(0, laneNames.indexOf(item.lane ?? ""));
    const p = placed.get(item);
    if (!p) return;
    const yy = laneTop[r] + p.row * rowHeight;
    s += `<rect x="${p.x0}" y="${yy + 2}" width="${p.w}" height="${rowHeight - 4}" rx="2" class="bar ${item.cls || ""}" data-key="${esc(item.key ?? "")}"><title>${esc(item.title || item.label || "")}</title></rect>`;
    if (p.w > 46 && item.label) s += `<text x="${p.x0 + 4}" y="${yy + rowHeight - 5}" class="bar-label" pointer-events="none">${esc(item.label.slice(0, Math.floor(p.w / 6.5)))}</text>`;
  });
  const height0 = height;
  const ticks = 8;
  for (let k = 0; k <= ticks; k++) {
    const t = (end * k) / ticks;
    s += `<text x="${X(t)}" y="${height0 - 4}" text-anchor="middle" class="c-axis">${t.toFixed(t < 10 ? 1 : 0)}s</text>`;
  }
  s += `<line class="playhead" x1="${L}" x2="${L}" y1="${top}" y2="${height0 - 16}"/>`;
  return s + "</svg>";
}

export function movePlayhead(svg, t) {
  if (!svg) return;
  const L = +svg.dataset.l, W = +svg.dataset.w, total = +svg.dataset.total || 1;
  const x = L + (W * Math.min(Math.max(t, 0), total)) / total;
  const line = svg.querySelector("line.playhead");
  line.setAttribute("x1", x);
  line.setAttribute("x2", x);
}
