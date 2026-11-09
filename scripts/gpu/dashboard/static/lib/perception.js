// Perception evidence: SAM3 masks you can switch on and off, grasp overlays, 3D geometry chips.
import { esc, num, fileUrl } from "./util.js";
import { img } from "./ui.js";

export const MASK_COLORS = ["#ff4040", "#40e040", "#4080ff", "#ffc040", "#e040ff", "#40e0e0"];

// One SAM3 request: the clean camera image with each candidate's mask and box on top.
// Recordings made before masks were kept fall back to the annotated image.
export function maskViewer(s, key) {
  const o = s.overlay;
  const masks = (o.candidates || []).some((c) => c.mask) && o.base;
  const rows = (o.candidates || []).map((c, i) => {
    const color = MASK_COLORS[i % MASK_COLORS.length];
    const toggle = masks && c.mask ? `<input type="checkbox" data-mask="${i}" ${i === 0 ? "checked" : ""}>` : "";
    return `<tr class="${i === 0 ? "top-pick" : ""}"><td>${toggle}</td><td><span class="swatch" style="background:${color}"></span>${c.index ?? i}</td><td>${esc(c.label)}</td><td><b>${num(c.score, 3)}</b>${c.score < 0.3 ? ` <span class="badge warn">low</span>` : ""}</td><td class="mono small">${esc((c.box_xyxy || []).join(", "))}</td><td>${c.mask_pixels ?? ""}</td></tr>`;
  }).join("");
  const view = masks
    ? `<div class="maskview" data-key="${esc(key)}" data-w="${(o.image_shape || [])[1] || ""}" data-h="${(o.image_shape || [])[0] || ""}">
        <img class="base" src="${fileUrl(o.base)}" alt="camera image">
        ${(o.candidates || []).map((c, i) => (c.mask ? `<img class="mask" data-i="${i}" data-src="${fileUrl(c.mask)}" data-color="${MASK_COLORS[i % MASK_COLORS.length]}" alt="" ${i === 0 ? "" : "hidden"}>` : "")).join("")}
        ${(o.candidates || []).map((c, i) => (c.box_xyxy ? `<div class="mbox" data-i="${i}" style="border-color:${MASK_COLORS[i % MASK_COLORS.length]}" ${i === 0 ? "" : "hidden"}><span>#${(c.index ?? i) + 1} ${num(c.score, 2)}</span></div>` : "")).join("")}
      </div>
      <div class="toolbar small"><button class="small" data-all="1">All masks</button><button class="small" data-all="0">None</button>${o.image ? `<a class="small" target="_blank" href="${fileUrl(o.image)}">annotated image</a>` : ""}</div>`
    : o.image ? img(o.image, "SAM3 overlay", "big") : "";
  return `<h3>SAM3 <span class="muted small">#${s.index} ${esc(s.name)}${o.camera ? ` · ${esc(o.camera)}` : ""}</span></h3>
    <div class="overlay">${view}
      <table class="table compact"><thead><tr><th></th><th>#</th><th>label</th><th>score</th><th>box (x0,y0,x1,y1)</th><th>pixels</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="muted small">Row 0 is what <span class="mono">localize_object</span> uses. Tick rows to compare masks.</p>`;
}

// Colour the 1-bit mask PNGs (canvas; works in every browser) and wire the toggles.
export function bindMaskViewers(root) {
  root.querySelectorAll(".maskview").forEach((view) => {
    const w = +view.dataset.w, h = +view.dataset.h;
    view.querySelectorAll(".mbox").forEach((box) => {
      const i = +box.dataset.i;
      const table = view.parentElement.querySelectorAll("tbody tr")[i];
      const coords = table ? table.querySelector(".mono").textContent.split(",").map(Number) : [];
      if (coords.length === 4 && w && h) {
        const [x0, y0, x1, y1] = coords;
        Object.assign(box.style, { left: `${(100 * x0) / w}%`, top: `${(100 * y0) / h}%`, width: `${(100 * (x1 - x0)) / w}%`, height: `${(100 * (y1 - y0)) / h}%` });
      }
    });
    view.querySelectorAll("img.mask").forEach((m) => tint(m));
    const panel = view.parentElement;
    const show = (i, on) => view.querySelectorAll(`[data-i="${i}"]`).forEach((el) => (el.hidden = !on));
    panel.querySelectorAll("input[data-mask]").forEach((box) => (box.onchange = () => show(box.dataset.mask, box.checked)));
    panel.parentElement.querySelectorAll("button[data-all]").forEach((b) => (b.onclick = () => {
      panel.querySelectorAll("input[data-mask]").forEach((box) => { box.checked = b.dataset.all === "1"; show(box.dataset.mask, box.checked); });
    }));
  });
}

const tinted = new Map();
function tint(el) {
  const src = el.dataset.src, color = el.dataset.color;
  const key = src + color;
  if (tinted.has(key)) { el.src = tinted.get(key); return; }
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const [r, g, b] = [1, 3, 5].map((k) => parseInt(color.slice(k, k + 2), 16));
    for (let p = 0; p < data.data.length; p += 4) {
      const on = data.data[p] > 127;
      data.data[p] = r; data.data[p + 1] = g; data.data[p + 2] = b; data.data[p + 3] = on ? 140 : 0;
    }
    ctx.putImageData(data, 0, 0);
    const url = canvas.toDataURL();
    tinted.set(key, url);
    el.src = url;
  };
  image.src = src;
}

// Grasp candidates drawn on the camera image when the call ran.
export function graspBlock(s) {
  return (s.grasp_overlays || []).map((g) => `<h3>Grasps <span class="muted small">#${s.index} ${esc(s.name)} · ${esc(g.camera)}</span></h3>
    <div class="overlay">${img(g.image, "grasp candidates", "big")}
      <table class="table compact"><thead><tr><th>#</th><th>score</th><th>width</th><th>position (m, base)</th></tr></thead><tbody>
      ${g.candidates.map((c) => `<tr class="${c.rank === 0 ? "top-pick" : ""}"><td>${c.rank + 1}</td><td><b>${num(c.score, 3)}</b></td><td>${num(c.width_m * 100, 1)} cm</td><td class="mono small">${c.position.map((v) => num(v, 3)).join(", ")}</td></tr>`).join("")}
      </tbody></table></div>`).join("");
}

// 3D results of a call and the calls inside it (point clouds, boxes, grasps), as chips.
export function geometryChips(spans) {
  const items = spans.flatMap((s) => (s.geometry || []).map((g) => ({ ...g, call: s })));
  if (!items.length) return "";
  const label = (g) => g.kind === "cloud" ? `point cloud · ${(g.points || 0).toLocaleString()} points`
    : g.kind === "box" ? `box · centre ${(g.center || []).map((v) => num(v, 3)).join(", ")} · size ${(g.extents || []).map((v) => num(v * 100, 1)).join(" × ")} cm`
    : `${g.kind === "grasp" ? "chosen grasp" : `${g.grasps} grasps`}`;
  return `<h3>3D</h3><div>${items.map((g) => `<span class="chip" title="#${g.call.index} ${esc(g.call.name)}">${esc(label(g))}${g.frame ? ` <span class="muted">${esc(g.frame)}</span>` : ""}</span>`).join(" ")}</div>
    <p class="muted small">See them with the rest of the scene in the <b>3D scene</b> tab.</p>`;
}
