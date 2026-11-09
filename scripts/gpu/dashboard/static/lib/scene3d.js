// A small 3D viewer on a 2D canvas (no libraries): point clouds, object boxes, grasps, motion
// targets and the hand's path, in the robot base frame (x forward, y left, z up).
// Drag to orbit, shift-drag to pan, wheel to zoom, double-click to reset.
import { esc, num } from "./util.js";

const COLORS = ["#ff5a5a", "#4cd964", "#5b8cff", "#ffb340", "#d070ff", "#36d1dc", "#ff7eb6", "#c0c040"];

export function scene3d(container, data, { onSelect } = {}) {
  const layers = buildLayers(data);
  if (!layers.length) {
    container.innerHTML = `<p class="muted">Nothing with a position was recorded in this episode. Point clouds, boxes and grasps are recorded by runs made with this version of the harness.</p>`;
    return;
  }
  container.innerHTML = `<div class="scene3d"><canvas></canvas><div class="legend">${layers.map((l, i) => `
    <label class="${l.call ? "has-call" : ""}"><input type="checkbox" data-layer="${i}" ${l.on ? "checked" : ""}>
      <span class="swatch" style="background:${l.color}"></span>${esc(l.label)}${l.call ? ` <a href="#" data-call="${esc(l.call)}">call</a>` : ""}</label>`).join("")}
    <p class="muted small">Drag to rotate · shift-drag to pan · wheel to zoom · double-click to reset. Frame: robot base (x forward, y left, z up), metres.</p></div></div>`;
  const canvas = container.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  // Frame what the program worked with (objects, boxes, grasps, targets, hand), not the whole floor.
  const focus = layers.filter((l) => !l.scene && !l.axes);
  const all = (focus.length ? focus : layers).flatMap((l) => l.points.concat(l.lines.flat()));
  const lo = [0, 1, 2].map((k) => Math.min(...all.map((p) => p[k])));
  const hi = [0, 1, 2].map((k) => Math.max(...all.map((p) => p[k])));
  const home = { yaw: -2.4, pitch: 0.55, zoom: 1.2, pan: [0, 0], center: lo.map((v, k) => (v + hi[k]) / 2), span: Math.max(...hi.map((v, k) => v - lo[k]), 0.2) };
  let view = { ...home, pan: [...home.pan] };

  const project = (p, w, h) => {
    const [x, y, z] = [p[0] - view.center[0], p[1] - view.center[1], p[2] - view.center[2]];
    const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    const rx = cy * x - sy * y, ry = sy * x + cy * y;          // yaw about z
    const depth = cp * ry + sp * z, up = -sp * ry + cp * z;     // pitch
    const scale = (Math.min(w, h) * 0.8 * view.zoom) / view.span;
    const persp = 1 / (1 + depth / (view.span * 3));
    return [w / 2 + view.pan[0] + rx * scale * persp, h / 2 + view.pan[1] - up * scale * persp, depth];
  };

  const draw = () => {
    const w = (canvas.width = canvas.clientWidth * devicePixelRatio);
    const h = (canvas.height = canvas.clientHeight * devicePixelRatio);
    ctx.fillStyle = "#14171f";
    ctx.fillRect(0, 0, w, h);
    const dot = Math.max(1.5, devicePixelRatio * 1.6);
    for (const layer of layers) {
      if (!layer.on) continue;
      layer.points.forEach((p, i) => {
        const [u, v] = project(p, w, h);
        ctx.fillStyle = layer.colors ? layer.colors[i] : layer.color;
        ctx.fillRect(u - dot / 2, v - dot / 2, dot, dot);
      });
      ctx.strokeStyle = layer.color;
      ctx.lineWidth = devicePixelRatio * (layer.width || 1.5);
      for (const line of layer.lines) {
        ctx.beginPath();
        line.forEach((p, i) => { const [u, v] = project(p, w, h); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); });
        ctx.stroke();
      }
    }
  };

  let drag = null;
  canvas.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) view.pan = [view.pan[0] + dx * devicePixelRatio, view.pan[1] + dy * devicePixelRatio];
    else { view.yaw -= dx * 0.01; view.pitch = Math.min(1.5, Math.max(-1.5, view.pitch + dy * 0.01)); }
    draw();
  };
  canvas.onpointerup = () => (drag = null);
  canvas.oncontextmenu = (e) => e.preventDefault();
  canvas.onwheel = (e) => { e.preventDefault(); view.zoom *= Math.exp(-e.deltaY * 0.0015); draw(); };
  canvas.ondblclick = () => { view = { ...home, pan: [...home.pan] }; draw(); };
  container.querySelectorAll("input[data-layer]").forEach((box) => (box.onchange = () => { layers[+box.dataset.layer].on = box.checked; draw(); }));
  container.querySelectorAll("a[data-call]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); onSelect && onSelect(a.dataset.call); }));
  new ResizeObserver(draw).observe(canvas);
  draw();
}

function buildLayers(data) {
  const layers = [];
  let c = 0;
  const next = () => COLORS[c++ % COLORS.length];
  const mm = (flat) => { const out = []; for (let i = 0; i + 2 < flat.length; i += 3) out.push([flat[i] / 1000, flat[i + 1] / 1000, flat[i + 2] / 1000]); return out; };
  for (const item of data.items || []) {
    if (item.kind === "cloud") {
      const points = mm(item.points_mm || []);
      const colors = item.scene && item.colors ? points.map((_, i) => `rgb(${item.colors[3 * i]},${item.colors[3 * i + 1]},${item.colors[3 * i + 2]})`) : null;
      layers.push({ label: `${item.scene ? "scene" : "object"} cloud · ${item.name} (${(item.point_count || 0).toLocaleString()} pts)`, points, lines: [], colors, color: item.scene ? "#888" : next(), on: true, call: item.call, scene: item.scene });
    } else if (item.kind === "box") {
      layers.push({ label: `box · ${item.name} · ${item.extents.map((v) => num(v * 100, 1)).join("×")} cm`, points: [], lines: boxLines(item.center, item.rotation, item.extents), color: next(), on: true, call: item.call, width: 2 });
    } else if (item.kind === "grasps" || item.kind === "grasp") {
      const chosen = item.kind === "grasp";
      layers.push({ label: chosen ? `chosen grasp · ${item.name}` : `${item.grasps.length} grasps · ${item.name}`, points: [], lines: item.grasps.flatMap((g) => gripperLines(g)), color: chosen ? "#ffffff" : next(), on: true, call: item.call, width: chosen ? 3 : 1.5 });
    }
  }
  if ((data.targets || []).length) {
    layers.push({ label: `motion targets (${data.targets.length})`, points: [], lines: data.targets.flatMap((t) => cross(t.position, 0.02)), color: "#ffe14d", on: true, width: 2 });
  }
  if ((data.ee_path || []).length > 1) layers.push({ label: "hand path", points: [], lines: [data.ee_path], color: "#9aa4b8", on: true });
  if (layers.length) layers.push({ label: "base axes (x red, y green, z blue)", points: [], lines: [], color: "#666", on: true, axes: true });
  const axes = layers.find((l) => l.axes);
  if (axes) {
    axes.lines = [[[0, 0, 0], [0.1, 0, 0]], [[0, 0, 0], [0, 0.1, 0]], [[0, 0, 0], [0, 0, 0.1]]];
  }
  return layers;
}

function boxLines(center, rotation, extents) {
  const [a, b, h] = extents.map((v) => v / 2);
  const corners = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    const local = [sx * a, sy * b, sz * h];
    corners.push([0, 1, 2].map((r) => center[r] + rotation[r][0] * local[0] + rotation[r][1] * local[1] + rotation[r][2] * local[2]));
  }
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  return edges.map(([i, j]) => [corners[i], corners[j]]);
}

function gripperLines(g) {
  const half = Math.max(g.width_m || 0.08, 0.02) / 2;
  const local = [[0, 0, 0], [0, 0, 0.0659], [0, -half, 0.0659], [0, half, 0.0659], [0, -half, 0.1034], [0, half, 0.1034]];
  const p = local.map((l) => [0, 1, 2].map((r) => g.position[r] + g.rotation[r][0] * l[0] + g.rotation[r][1] * l[1] + g.rotation[r][2] * l[2]));
  return [[p[0], p[1]], [p[2], p[3]], [p[2], p[4]], [p[3], p[5]]];
}

function cross(p, s) {
  return [[[p[0] - s, p[1], p[2]], [p[0] + s, p[1], p[2]]], [[p[0], p[1] - s, p[2]], [p[0], p[1] + s, p[2]]], [[p[0], p[1], p[2] - s], [p[0], p[1], p[2] + s]]];
}
