// One episode as a replay: synchronized camera videos, a call timeline, the chain of calls, and
// for each call its keyframes, SAM3 candidates and full input / output / error.
import { api, esc, q, num, fileUrl, size, pretty, mmss } from "../lib/util.js";
import { badge, card, downloads, img, jsonView, tabs, bindTabs, bindZoom } from "../lib/ui.js";
import { lineChart, moveMarkers, gantt, movePlayhead } from "../lib/charts.js";

let state = null;

export async function render(app, params) {
  const path = params.get("path");
  const d = await api(`/api/episode?path=${q(path)}`);
  const m = d.meta, o = d.outcome || {};
  const spans = d.spans;
  const byId = Object.fromEntries(spans.map((s) => [s.id, s]));
  const children = {};
  spans.forEach((s) => s.parent && (children[s.parent] ||= []).push(s));
  const top = spans.filter((s) => !s.parent);
  const duration = Math.max(d.duration_s || 0, 0.1);
  const failed = spans.filter((s) => s.ok === false);
  state = { d, byId, children, top, duration, selected: null };

  const session = path.split("/").slice(0, 2).join("/");
  app.innerHTML = `
    <div class="crumbs"><a href="#/">Runs</a> / <a href="#/session?path=${q(session)}">${esc(session)}</a> / episode</div>
    <div class="page-head"><h1>${esc(m.suite)}:${esc(m.task_id)} <span class="muted">seed ${esc(m.seed)}</span> ${badge(o.task_success, { yes: "task solved", no: "task not solved" })}</h1>
      <div class="toolbar">${downloads(d.path, "Download episode")}</div></div>
    ${d.language ? `<div class="instruction">“${esc(d.language)}”</div>` : ""}
    <div class="cards">
      ${card("Program", badge(o.program_ok, { yes: "ran", no: "error" }), esc(o.termination_reason || "not finalized"))}
      ${card("Steps", `${o.steps_executed ?? "?"}`, `${num(o.simulator_duration_s)} s simulated`)}
      ${card("Wall time", `${num(o.wall_duration_s)} s`, o.timing ? `simulator steps ${num(((o.timing.buckets || {}).adapter_step || {}).seconds)} s` : "")}
      ${card("Calls", `${spans.length}`, failed.length ? `<span class="bad">${failed.length} failed</span>` : "none failed")}
      ${card("Task", `<span class="small">${esc(m.benchmark)}</span>`, esc(m.task_name || ""))}
    </div>
    ${d.program_error ? `<div class="alert bad"><b>Program error:</b> ${esc(d.program_error.message || JSON.stringify(d.program_error))}</div>` : ""}
    <div class="panel player">
      <div class="videos">${d.media.videos.map((v) => `<figure><video class="sync" muted playsinline preload="auto" src="${fileUrl(v)}"></video><figcaption>${esc(v.split("/").pop().replace(".mp4", ""))}</figcaption></figure>`).join("") || `<p class="muted">No videos were recorded for this episode.</p>`}</div>
      ${d.media.videos.length ? `<div class="controls">
        <button id="play" class="primary">Play</button>
        <button id="back" title="one frame back">◀︎</button><button id="fwd" title="one frame forward">▶︎</button>
        <select id="speed">${[0.25, 0.5, 1, 2, 4].map((r) => `<option value="${r}" ${r === 1 ? "selected" : ""}>${r}×</option>`).join("")}</select>
        <input type="range" id="scrub" min="0" max="${duration}" step="0.01" value="0" class="grow-input">
        <span id="clock" class="mono small">0:00.00 / ${mmss(duration)}</span>
      </div>` : ""}
      <div id="timeline">${gantt(
        spans.map((s) => ({ start: s.start, end: s.end, lane: s.category || "", key: s.id, label: s.name, cls: `${s.category} ${s.ok === false ? "failed" : ""}`, title: `#${s.index} ${s.name} (${s.category})\nsimulated ${num(s.start, 2)}–${num(s.end, 2)} s, wall ${num(s.wall_s, 3)} s${s.error ? "\n" + s.error : ""}` })),
        { total: duration, lanes: ["program", "api", "provider"].filter((l) => spans.some((s) => s.category === l)) },
      )}</div>
      <div class="muted small">Click a bar or a call to jump the videos there and see its details. Bars: <span class="chip program">program</span> your program's calls, <span class="chip api">api</span> stages inside the robot API, <span class="chip provider">provider</span> model services (SAM3, Contact-GraspNet, PyRoki, cuRobo); red = failed.</div>
    </div>
    <div class="split">
      <div class="panel chain"><div class="panel-head"><b>Chain of calls</b><label class="small"><input type="checkbox" id="only-failed"> failed only</label></div><div id="chain"></div></div>
      <div class="panel detail"><div id="detail"><p class="muted">Select a call.</p></div></div>
    </div>
    ${tabs("ep", ["Trajectories", "Perception (SAM3)", "Keyframes", "Program & output", ...(d.commands.length ? ["Console commands"] : []), "Files"], "Trajectories")}`;

  drawChain();
  document.getElementById("only-failed").onchange = drawChain;
  document.querySelectorAll("#timeline rect.bar").forEach((r) => (r.onclick = () => select(r.dataset.key, true)));
  setupPlayer();
  bindTabs("ep", renderTab);
  const firstFailure = failed.find((s) => !s.parent) || failed[0];
  select((firstFailure || top[0] || {}).id, false);
}

// ---------------------------------------------------------------------------- chain --
function drawChain() {
  const { top, children } = state;
  const onlyFailed = document.getElementById("only-failed").checked;
  const failedBelow = (s) => s.ok === false || (children[s.id] || []).some(failedBelow);
  const row = (s, depth) => {
    if (onlyFailed && !failedBelow(s)) return "";
    const kids = children[s.id] || [];
    const line = `<div class="call ${s.ok === false ? "failed" : ""}" data-id="${s.id}" style="padding-left:${8 + depth * 16}px">
      <span class="dot ${s.category}"></span><b>${esc(s.name)}</b>
      ${s.ok === false ? `<span class="badge bad">failed</span>` : ""}
      ${s.keyframes ? `<span class="chip small">frames</span>` : ""}${s.overlay ? `<span class="chip small">SAM3</span>` : ""}
      <span class="muted small right">${s.end > s.start ? `${num(s.start, 1)}–${num(s.end, 1)} s` : `@ ${num(s.start, 1)} s`} · ${num(s.wall_s, 2)} s wall</span>
      ${s.error ? `<div class="small bad err-line">${esc(s.error)}</div>` : ""}</div>`;
    if (!kids.length) return line;
    return `<details ${s.ok === false ? "open" : ""}><summary>${line}</summary>${kids.map((k) => row(k, depth + 1)).join("")}</details>`;
  };
  document.getElementById("chain").innerHTML = top.map((s) => row(s, 0)).join("") || `<p class="muted">No calls recorded.</p>`;
  document.querySelectorAll("#chain .call").forEach((el) => (el.onclick = (event) => {
    event.preventDefault();
    select(el.dataset.id, true);
  }));
}

// ---------------------------------------------------------------------------- detail --
async function select(id, seekVideo) {
  const s = state.byId[id];
  if (!s) return;
  state.selected = id;
  document.querySelectorAll("#chain .call").forEach((el) => el.classList.toggle("selected", el.dataset.id === id));
  document.querySelectorAll("#timeline rect.bar").forEach((r) => r.classList.toggle("selected", r.dataset.key === id));
  if (seekVideo) seek(s.start);
  const descendants = [];
  const walk = (x) => (state.children[x.id] || []).forEach((c) => (descendants.push(c), walk(c)));
  walk(s);
  const overlays = [s, ...descendants].filter((x) => x.overlay);
  const box = document.getElementById("detail");
  box.innerHTML = `
    <div class="panel-head"><b>#${s.index} ${esc(s.name)}</b> <span class="chip ${s.category}">${esc(s.category)}</span> ${badge(s.ok, { yes: "ok", no: "failed" })}
      <button class="small" id="jump">Jump video to ${num(s.start, 2)} s</button></div>
    <div class="small muted">simulated ${num(s.start, 2)}–${num(s.end, 2)} s (${num(s.end - s.start, 2)} s) · wall ${num(s.wall_s, 3)} s${s.started_at ? ` · started ${esc(s.started_at.replace("T", " ").replace("Z", ""))}` : ""}</div>
    ${s.error ? `<div class="alert bad">${esc(s.error)}</div>` : ""}
    ${s.keyframes ? `<h3>Keyframes</h3><div class="kf-pair"><div><div class="muted small">before</div><div class="grid">${(s.keyframes.before || []).map((p) => img(p, p.split("/").slice(-2).join("/"))).join("")}</div></div>
      <div><div class="muted small">after</div><div class="grid">${(s.keyframes.after || []).map((p) => img(p, p.split("/").slice(-2).join("/"))).join("")}</div></div></div>` : ""}
    ${overlays.map(overlayBlock).join("")}
    ${descendants.length ? `<h3>Inside this call</h3><div class="small">${descendants.map((c) => `<div class="sub ${c.ok === false ? "bad" : ""}" data-id="${c.id}">#${c.index} ${esc(c.name)} <span class="muted">${esc(c.category)} · ${num(c.wall_s, 3)} s</span>${c.error ? ` — ${esc(c.error)}` : ""}</div>`).join("")}</div>` : ""}
    <h3>Input · output · error</h3><div id="io" class="muted small">${s.dir ? "loading..." : "No recorded input/output for this call."}</div>`;
  document.getElementById("jump").onclick = () => seek(s.start);
  box.querySelectorAll(".sub[data-id]").forEach((el) => (el.onclick = () => select(el.dataset.id, true)));
  bindZoom(box);
  if (s.dir) {
    const c = await api(`/api/call?path=${q(s.dir)}`);
    if (state.selected !== id) return;
    document.getElementById("io").innerHTML = ["input", "output", "error"].filter((k) => c[k] !== undefined).map((k) => `<details ${k !== "output" || s.ok === false ? "open" : ""}><summary><b>${k}</b> <a class="small" target="_blank" href="${fileUrl(c.dir + "/" + k + ".json")}">raw</a></summary>${jsonView(c[k], { open: 1 })}</details>`).join("") +
      (c.other_files || []).map((f) => (/\.(png|jpe?g)$/.test(f) ? img(f, f.split("/").pop(), "inline") : `<div><a target="_blank" href="${fileUrl(f)}">${esc(f.split("/").pop())}</a></div>`)).join("");
    bindZoom(document.getElementById("io"));
  }
}

function overlayBlock(s) {
  const o = s.overlay;
  return `<h3>SAM3 result <span class="muted small">(#${s.index} ${esc(s.name)})</span></h3>
    <div class="overlay">${o.image ? img(o.image, "SAM3 overlay", "big") : ""}
      <table class="table compact"><thead><tr><th>#</th><th>label</th><th>score</th><th>box (x0,y0,x1,y1)</th><th>pixels</th></tr></thead><tbody>
      ${(o.candidates || []).map((c, i) => `<tr class="${i === 0 ? "top-pick" : ""}"><td>${c.index ?? i}</td><td>${esc(c.label)}</td><td><b>${num(c.score, 3)}</b></td><td class="mono small">${esc((c.box_xyxy || []).join(", "))}</td><td>${c.mask_pixels ?? ""}</td></tr>`).join("")}
      </tbody></table></div>`;
}

// ---------------------------------------------------------------------------- player --
function setupPlayer() {
  const videos = [...document.querySelectorAll("video.sync")];
  state.videos = videos;
  if (!videos.length) return;
  const master = videos[0];
  const fps = state.d.video_fps || 20;
  const play = document.getElementById("play");
  const scrub = document.getElementById("scrub");
  let raf = null;
  const tick = () => {
    const t = master.currentTime;
    for (const v of videos.slice(1)) if (Math.abs(v.currentTime - t) > 0.12) v.currentTime = t;
    update(t);
    if (!master.paused) raf = requestAnimationFrame(tick);
  };
  play.onclick = () => {
    if (master.paused) {
      videos.forEach((v) => v.play());
      play.textContent = "Pause";
      raf = requestAnimationFrame(tick);
    } else {
      videos.forEach((v) => v.pause());
      play.textContent = "Play";
      cancelAnimationFrame(raf);
      update(master.currentTime);
    }
  };
  master.onended = () => (play.textContent = "Play");
  document.getElementById("speed").onchange = (e) => videos.forEach((v) => (v.playbackRate = +e.target.value));
  document.getElementById("back").onclick = () => seek(master.currentTime - 1 / fps);
  document.getElementById("fwd").onclick = () => seek(master.currentTime + 1 / fps);
  scrub.oninput = () => seek(+scrub.value);
  master.addEventListener("loadedmetadata", () => {
    if (master.duration && Number.isFinite(master.duration)) scrub.max = master.duration;
  });
}

function seek(t) {
  if (!state.videos || !state.videos.length) return update(t);
  const max = state.videos[0].duration || state.duration;
  const clamped = Math.min(Math.max(t, 0), max);
  state.videos.forEach((v) => (v.currentTime = clamped));
  update(clamped);
}

function update(t) {
  const scrub = document.getElementById("scrub");
  if (scrub && document.activeElement !== scrub) scrub.value = t;
  const clock = document.getElementById("clock");
  if (clock) clock.textContent = `${mmss(t)} / ${mmss(state.duration)} · step ${Math.round(t * (state.d.video_fps || 20))}`;
  movePlayhead(document.querySelector("#timeline svg"), t);
  moveMarkers(document.getElementById("ep-body") || document, t / state.duration);
  const current = state.top.find((s) => s.end > s.start && t >= s.start && t <= s.end);
  document.querySelectorAll("#chain .call").forEach((el) => el.classList.toggle("now", !!current && el.dataset.id === current.id));
}

// ------------------------------------------------------------------------------ tabs --
function renderTab(tab) {
  const d = state.d;
  if (tab === "Trajectories") {
    const charts = Object.entries(d.charts);
    return charts.length ? charts.map(([title, series]) => lineChart(series, title, { height: 170, x: "time →", marker: 0 })).join("") + `<p class="muted small">The vertical line follows the video.</p>` : `<p class="muted">No steps recorded.</p>`;
  }
  if (tab === "Perception (SAM3)") {
    const withOverlay = d.spans.filter((s) => s.overlay);
    return withOverlay.length ? withOverlay.map(overlayBlock).join("") : `<p class="muted">No SAM3 calls in this episode.</p>`;
  }
  if (tab === "Keyframes") {
    const motions = d.spans.filter((s) => s.keyframes);
    return motions.length ? motions.map((s) => `<div class="kf-row"><div class="kf-title"><b>#${s.index} ${esc(s.name)}</b> ${badge(s.ok, { yes: "ok", no: "failed" })} <span class="muted small">${num(s.start, 1)}–${num(s.end, 1)} s</span></div>
      <div class="kf-pair"><div class="grid">${(s.keyframes.before || []).map((p) => img(p, "before " + p.split("/").pop())).join("")}</div><div class="grid">${(s.keyframes.after || []).map((p) => img(p, "after " + p.split("/").pop())).join("")}</div></div></div>`).join("")
      : `<div class="grid">${d.media.keyframes.map((p) => img(p, p)).join("")}</div>`;
  }
  if (tab === "Program & output") {
    return `${d.result !== null && d.result !== undefined ? `<h3>Program result</h3>${jsonView(d.result, { open: 2 })}` : ""}
      ${Object.entries(d.texts).map(([k, v]) => `<h3>${esc(k)}</h3><pre class="code">${esc(pretty(v))}</pre>`).join("")}`;
  }
  if (tab === "Console commands") {
    return d.commands.map((c, i) => `<div class="msg ${c.error ? "err" : "tool_use"}"><b>${i + 1}. ${esc(c.kind)} ${esc(c.name)}</b> ${badge(!c.error, { yes: c.status, no: c.status })} <span class="muted small">${num(c.elapsed_s, 2)} s · steps after ${c.steps_after ?? ""}</span>
      <pre class="code">${esc(c.code || pretty(c.arguments))}</pre>${c.error ? `<div class="alert bad">${esc(c.error)}</div>` : ""}<details><summary>result</summary>${jsonView(c.result)}</details></div>`).join("");
  }
  if (tab === "Files") {
    return `<table class="table"><tbody>${d.files.map((f) => `<tr><td class="mono small">${esc(f.name)}</td><td class="small">${size(f.size)}</td><td class="nowrap">${/\.(log|txt|jsonl|stdout|stderr|json|py)$/.test(f.name) ? `<a class="btn small" href="#/log?path=${q(f.path)}">View</a> ` : ""}<a class="btn small" href="${fileUrl(f.path)}?download=1">Download</a></td></tr>`).join("")}</tbody></table>
      <p class="muted small">Each call's input/output/error is under trace/calls (shown per call above, and in the zip).</p>`;
  }
  return "";
}
