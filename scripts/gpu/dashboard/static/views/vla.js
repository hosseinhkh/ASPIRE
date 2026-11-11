// VLA student: SmolVLA benchmark runs under outputs/vla, kept apart from ASPIRE's runs.
// List of runs and parity checks -> one run (progress, tasks, episodes, log) -> one episode
// (video synced with its action, hand-position and gripper charts).
import { api, esc, q, num, fileUrl, when, size, pct } from "../lib/util.js";
import { card, jsonView, rate, tabs, bindTabs } from "../lib/ui.js";
import { lineChart, moveMarkers } from "../lib/charts.js";

const STATUS_CLASS = { running: "ok", finished: "ok", interrupted: "warn", stopped: "bad", failed: "bad" };
const REFRESH_MS = 10000;
let runTab = "Tasks";
let episodeFilter = { task: "all", seed: "all", outcome: "all" };

const statusBadge = (s) => `<span class="badge ${STATUS_CLASS[s] || "muted"}">${esc(s || "-")}${s === "running" ? " ●" : ""}</span>`;
const minutes = (s) => (s === null || s === undefined ? "-" : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const percent = (v) => (v === null || v === undefined ? "-" : `${(100 * v).toFixed(1)}%`);
const zip = (path, media, label) => `<a class="btn" href="/download?path=${q(path)}&media=${media ? 1 : 0}">${esc(label)}</a>`;

function progressBar(done, planned) {
  const p = pct(done || 0, planned || 0);
  return `<span class="rate"><span class="bar"><span class="fill ok" style="width:${p}%"></span></span><b>${done ?? 0}/${planned ?? "?"}</b> <span class="muted">${p}%</span></span>`;
}

function schedule(ctx, fn) {
  ctx.timers.push(setTimeout(fn, REFRESH_MS));
}

// ------------------------------------------------------------------------- list --
export async function renderList(app, params, ctx) {
  const data = await api("/api/vla/runs");
  const running = data.runs.some((r) => r.progress.status === "running");
  app.innerHTML = `
    <div class="page-head"><h1>VLA student</h1></div>
    <p class="muted small">SmolVLA (LeRobot) running through ASPIRE's own LIBERO adapter in hand-movement (OSC_POSE) control.
      Runs live in <span class="mono">outputs/vla/</span>, apart from ASPIRE's runs. Start them with
      <span class="mono">scripts/gpu/vla.sh</span>; this page refreshes itself while a run is going.</p>
    <h2>Parity checks</h2>
    <p class="muted small">LeRobot's LIBERO env and ASPIRE's adapter on the same scene and actions (<span class="mono">vla.sh parity --policy</span>).</p>
    ${
      data.parity.length
        ? `<div class="cards">${data.parity
            .map((p) =>
              card(
                `${p.suite} task ${p.task} scene ${p.scene}`,
                `${p.environment ? '<span class="badge ok">environment PASS</span>' : '<span class="badge bad">environment FAIL</span>'} ${
                  p.policy === undefined || p.policy === null ? "" : p.policy ? '<span class="badge ok">policy PASS</span>' : '<span class="badge bad">policy FAIL</span>'
                }`,
                `${p.policy_detail ? `SmolVLA: ${esc(p.policy_detail.aspire.steps)} steps, ${p.policy_detail.aspire.success ? "success" : "fail"}; max action difference ${esc(p.policy_detail.max_action_difference)}<br>` : ""}${when(p.modified)}`,
              ),
            )
            .join("")}</div>`
        : `<p class="muted">No parity report yet: run <span class="mono">scripts/gpu/vla.sh parity --policy</span>.</p>`
    }
    <h2>Runs</h2>
    <div class="panel flush"><table class="table"><thead><tr><th>Run</th><th>Suite</th><th>Status</th><th>Progress</th><th>Success so far</th><th>Published</th><th>Speed / ETA</th><th>Updated</th><th>Download</th></tr></thead>
    <tbody>${
      data.runs
        .map((r) => {
          const p = r.progress;
          return `<tr class="click" data-href="#/vla/run?path=${q(r.path)}">
            <td><b>${esc(r.name)}</b><div class="muted small">${esc(r.policy || "")}${r.n_action_steps ? ` · ${esc(r.n_action_steps)} actions per chunk` : ""}</div></td>
            <td class="mono small">${esc(r.suite || "")}</td>
            <td>${statusBadge(p.status)}${(p.warnings || []).length ? ` <span class="badge warn" title="${esc(p.warnings.join("\n"))}">${p.warnings.length} warning(s)</span>` : ""}</td>
            <td>${progressBar(p.done, p.planned)}</td>
            <td>${rate(p.successes || 0, p.done || 0)}</td>
            <td class="small">${(p.notes || []).map(esc).join("<br>")}</td>
            <td class="small nowrap">${p.s_per_episode ? `${num(p.s_per_episode)} s/episode` : "-"}${p.status === "running" ? `<br>ETA ${minutes(p.eta_s)}` : ""}</td>
            <td class="small nowrap">${esc(p.updated || "")}</td>
            <td data-stop class="nowrap"><span class="btn-group">${zip(r.path, false, "Logs")}${zip(r.path, true, "All")}</span></td></tr>`;
        })
        .join("") || `<tr><td colspan="9" class="muted">No runs yet: <span class="mono">scripts/gpu/vla.sh sanity --quick</span></td></tr>`
    }</tbody></table></div>`;
  app.querySelectorAll("tr.click").forEach((tr) => {
    tr.onclick = (e) => !e.target.closest("[data-stop]") && (location.hash = tr.dataset.href);
  });
  if (running) schedule(ctx, () => renderList(app, params, ctx));
}

// -------------------------------------------------------------------------- run --
export async function renderRun(app, params, ctx) {
  const path = params.get("path");
  const d = await api(`/api/vla/run?path=${q(path)}`);
  const p = d.progress, r = d.results || {}, settings = (d.meta || {}).settings || {};
  const tasks = r.tasks || [];
  const seeds = Object.keys(r.per_seed_mean || {});
  app.innerHTML = `
    <div class="crumbs"><a href="#/vla">VLA student</a> / ${esc(d.name)}</div>
    <div class="page-head"><h1>${esc(p.title || d.name)} ${statusBadge(p.status)}</h1>
      <div class="toolbar"><span class="btn-group">${zip(d.path, false, "Download logs")}${zip(d.path, true, "Download all")}</span></div></div>
    <div class="cards">
      ${card("Progress", progressBar(p.done, p.planned), `${esc(p.resumed || 0)} resumed, ${esc(p.ran || 0)} run by this process`)}
      ${card("Success", r.episodes ? `<b>${percent(r.mean)}</b>` : "-", r.episodes ? `${r.successes}/${r.episodes} episodes · 95% ${percent((r.ci95 || [])[0])}–${percent((r.ci95 || [])[1])}${seeds.length > 1 ? ` · ±${(100 * r.std_across_seeds).toFixed(1)} across seeds` : ""}` : "")}
      ${card("Published", p.expected ? `${num(p.expected, 0)}%` : "-", (p.notes || []).map(esc).join("<br>"))}
      ${card("Speed", p.s_per_episode ? `${num(p.s_per_episode)} s/episode` : "-", p.status === "running" ? `ETA ${minutes(p.eta_s)}` : `elapsed ${minutes(p.elapsed_s)}`)}
      ${card("Where", `GPU ${esc(p.gpu ?? "-")}`, `pid ${esc(p.pid ?? "-")} on ${esc(p.host ?? "-")}<br>started ${esc(p.started || "")}`)}
      ${card("Now", p.current ? `task ${esc(p.current.task_id)}` : "-", p.current ? `seed ${esc(p.current.policy_seed_index)} scene ${esc(p.current.scene)}` : p.last ? `last: task ${esc(p.last.task_id)} scene ${esc(p.last.scene)} ${p.last.success ? "success" : "fail"}` : "")}
    </div>
    ${(p.warnings || []).map((w) => `<div class="alert warn">${esc(w)}</div>`).join("")}
    ${p.error ? `<div class="alert bad"><b>Error:</b> ${esc(p.error)}</div>` : ""}
    ${tabs("vla-run", ["Tasks", "Episodes", "Log", "Settings"], runTab)}`;

  const taskTable = () => `
    <table class="table"><thead><tr><th>Task</th><th>Success</th><th>95% interval</th>${seeds.map((s) => `<th>seed ${esc(s)}</th>`).join("")}<th>Mean steps (successes)</th></tr></thead><tbody>
    ${tasks
      .map(
        (t) => `<tr class="click" data-task="${esc(t.task_id)}"><td><b>${esc(t.suite)}:${esc(t.task_id)}</b><div class="muted small">${esc(d.instructions[String(t.task_id)] || "")}</div></td>
        <td>${rate(t.successes, t.episodes)}</td><td class="small">${percent(t.ci95[0])}–${percent(t.ci95[1])}</td>
        ${seeds.map((s) => `<td class="small">${t.per_seed[s] === undefined ? "-" : percent(t.per_seed[s])}</td>`).join("")}
        <td>${t.mean_steps_when_successful ? num(t.mean_steps_when_successful, 0) : "-"}</td></tr>`,
      )
      .join("") || `<tr><td colspan="${4 + seeds.length}" class="muted">No finished episode yet.</td></tr>`}
    </tbody></table>
    ${(r.references || []).length ? `<p class="muted small">Published numbers to compare: ${r.references.map(esc).join("; ")}.</p>` : ""}`;

  const episodeTable = () => {
    const taskIds = [...new Set(d.episodes.map((e) => e.task_id))];
    const seedIds = [...new Set(d.episodes.map((e) => e.seed))];
    const f = episodeFilter;
    const rows = d.episodes.filter(
      (e) => (f.task === "all" || String(e.task_id) === f.task) && (f.seed === "all" || String(e.seed) === f.seed) && (f.outcome === "all" || (f.outcome === "success") === !!e.success),
    );
    const option = (value, label, current) => `<option value="${esc(value)}" ${String(current) === String(value) ? "selected" : ""}>${esc(label)}</option>`;
    return `<div class="toolbar">
        <select id="f-task">${option("all", "all tasks", f.task)}${taskIds.map((t) => option(t, `task ${t}`, f.task)).join("")}</select>
        <select id="f-seed">${option("all", "all seeds", f.seed)}${seedIds.map((s) => option(s, `seed ${s}`, f.seed)).join("")}</select>
        <select id="f-outcome">${option("all", "all outcomes", f.outcome)}${option("success", "successes", f.outcome)}${option("fail", "failures", f.outcome)}</select>
        <span class="muted small">${rows.length} of ${d.episodes.length}</span></div>
      <table class="table compact"><thead><tr><th>Task</th><th>Seed</th><th>Scene</th><th>Outcome</th><th>Steps</th><th>Ended</th><th>Wall</th><th title="share of hand-motion values at the controller limit">Saturated</th><th>Hand path</th><th>Policy</th><th></th></tr></thead><tbody>
      ${rows
        .map(
          (e) => `<tr class="click" data-href="#/vla/episode?path=${q(e.path)}"><td>${esc(e.task_id)}</td><td>${esc(e.seed)}</td><td>${esc(e.scene)}</td>
          <td>${e.success ? '<span class="badge ok">success</span>' : '<span class="badge bad">fail</span>'}</td><td>${esc(e.steps)}</td><td class="small">${esc(e.ended || "")}</td>
          <td class="small">${num(e.wall_s)} s</td><td class="small">${e.saturated === undefined || e.saturated === null ? "-" : percent(e.saturated)}</td>
          <td class="small">${e.hand_path_m === undefined || e.hand_path_m === null ? "-" : num(e.hand_path_m, 2) + " m"}</td>
          <td class="small">${e.policy_ms_per_step ? num(e.policy_ms_per_step) + " ms/step" : "-"}</td><td>${e.video ? "▶" : ""}</td></tr>`,
        )
        .join("")}
      </tbody></table>`;
  };

  const logView = () =>
    `<p class="muted small">Last ${d.log.length} of ${d.log_lines} lines of <span class="mono">${esc(d.path)}/benchmark.log</span>.</p>
     <pre class="log">${d.log.map((line) => `<div class="${/WARNING|ERROR|failed|stopped/.test(line) ? "alert-line" : ""}">${esc(line)}</div>`).join("")}</pre>`;

  const bodies = {
    Tasks: taskTable,
    Episodes: episodeTable,
    Log: logView,
    Settings: () => jsonView(d.meta, { open: 2 }),
  };
  bindTabs("vla-run", (name) => bodies[name](), (name) => {
    runTab = name;
    const body = document.getElementById("vla-run-body");
    body.querySelectorAll("tr.click[data-href]").forEach((tr) => (tr.onclick = () => (location.hash = tr.dataset.href)));
    body.querySelectorAll("tr.click[data-task]").forEach((tr) => {
      tr.onclick = () => {
        episodeFilter = { ...episodeFilter, task: tr.dataset.task };
        document.querySelector('#vla-run button[data-tab="Episodes"]').click();
      };
    });
    for (const [id, key] of [["f-task", "task"], ["f-seed", "seed"], ["f-outcome", "outcome"]]) {
      const select = document.getElementById(id);
      if (select) select.onchange = () => {
        episodeFilter = { ...episodeFilter, [key]: select.value };
        document.querySelector('#vla-run button[data-tab="Episodes"]').click();
      };
    }
    if (name === "Log") body.querySelector("pre.log")?.scrollTo(0, 1e9);
  });
  if (p.status === "running") schedule(ctx, () => renderRun(app, params, ctx));
}

// ---------------------------------------------------------------------- episode --
export async function renderEpisode(app, params) {
  const path = params.get("path");
  const d = await api(`/api/vla/episode?path=${q(path)}`);
  const e = d.summary, diag = e.diagnostics || {}, s = d.series;
  const split = (series, keys) => Object.fromEntries(keys.filter((k) => series && series[k]).map((k) => [k, series[k]]));
  const charts = [
    s.action && lineChart(split(s.action, ["dx", "dy", "dz"]), "Hand position command (OSC, -1..1)", { x: "step" }),
    s.action && lineChart(split(s.action, ["drx", "dry", "drz"]), "Hand rotation command (OSC, -1..1)", { x: "step" }),
    s.action && lineChart(split(s.action, ["gripper"]), "Gripper command (-1 open, +1 close)", { x: "step", height: 120 }),
    s.eef_pos && lineChart(s.eef_pos, "Hand position (m, world)", { x: "step" }),
    s.gripper_qpos && lineChart(s.gripper_qpos, "Gripper fingers (m)", { x: "step", height: 140 }),
  ].filter(Boolean);
  app.innerHTML = `
    <div class="crumbs"><a href="#/vla">VLA student</a> / <a href="#/vla/run?path=${q(d.run)}">${esc(d.run_name)}</a> / episode</div>
    <div class="page-head"><h1>${esc(e.task_ref)} <span class="muted">seed ${esc(e.policy_seed_index)} · scene ${esc(e.scene)}</span>
      ${e.success ? '<span class="badge ok">success</span>' : '<span class="badge bad">fail</span>'}</h1>
      <div class="toolbar"><span class="btn-group">${zip(d.path, true, "Download episode")}</span></div></div>
    <div class="instruction">“${esc(e.instruction)}”</div>
    <div class="cards">
      ${card("Outcome", e.success ? "success" : "fail", `ended by ${esc(e.ended)} after ${esc(e.steps)} of ${esc(e.max_steps)} steps`)}
      ${card("Scene", `init state ${esc(e.init_state_index ?? "seeded")}`, `policy noise seed ${esc(e.policy_seed)}`)}
      ${card("Time", `${num(e.wall_s)} s`, `policy ${num(e.policy_s)} s · ${diag.policy_ms_per_step ? num(diag.policy_ms_per_step) + " ms/step" : ""}`)}
      ${card("Hand", diag.hand_path_m !== undefined ? `${num(diag.hand_path_m, 2)} m travelled` : "-", diag.hand_net_m !== undefined ? `${num(diag.hand_net_m, 2)} m from start` : "")}
      ${card("Commands", diag.action_saturated_fraction !== undefined ? `${percent(diag.action_saturated_fraction)} saturated` : "-", diag.gripper_switches !== undefined ? `gripper ${esc(diag.gripper_switches)} switches, closed ${percent(diag.gripper_closed_fraction)}` : "")}
    </div>
    <div class="player">${
      d.video
        ? `<div class="videos"><figure style="max-width:900px;flex-basis:640px"><video id="vla-video" controls muted playsinline src="${fileUrl(d.video)}"></video>
           <figcaption class="muted small">agentview | wrist camera, as the policy saw them (rotated 180°), 20 fps</figcaption></figure></div>`
        : `<p class="muted">No video was kept for this episode (--video failures or none).</p>`
    }</div>
    <div id="vla-charts" class="panel">${charts.join("") || '<p class="muted">No recorded steps.</p>'}</div>
    <div class="panel"><div class="panel-head"><b>Files</b></div>
      ${d.files.map((f) => `<div><a href="${fileUrl(f.path)}" target="_blank">${esc(f.name)}</a> <span class="muted small">${size(f.bytes)}</span></div>`).join("")}
      <details><summary class="muted">episode.json</summary>${jsonView(e, { open: 1 })}</details></div>`;
  const video = document.getElementById("vla-video");
  if (video) {
    const box = document.getElementById("vla-charts");
    video.addEventListener("timeupdate", () => moveMarkers(box, video.duration ? video.currentTime / video.duration : 0));
  }
}
