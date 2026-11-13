// One session / campaign / check folder: summary, stage timeline, GPU, tasks, episodes, agents, logs.
import { api, esc, q, num, fileUrl, size, mmss, zipUrl } from "../lib/util.js";
import { badge, card, downloads, rate, statusBadge, tabs, bindTabs } from "../lib/ui.js";
import { lineChart, gantt } from "../lib/charts.js";

export async function render(app, params) {
  const path = params.get("path");
  const d = await api(`/api/collection?path=${q(path)}`);
  const m = d.meta || {};
  const stages = d.stages || [];
  const done = d.episodes.filter((e) => e.finalized);
  const ok = done.filter((e) => e.task_success === true).length;
  const agentCost = d.agents.reduce((sum, a) => sum + (Number(a.cost_usd) || 0), 0);
  const timed = stages.filter((s) => s.start !== null && s.end > s.start && !String(s.stage).startsWith("preflight: "));
  const span = timed.length ? Math.max(...timed.map((s) => s.end)) : 0;
  const peak = d.gpu ? Math.max(...d.gpu.yours) : null;

  app.innerHTML = `
    <div class="crumbs"><a href="#/">Runs</a> / ${esc(path)}</div>
    <div class="page-head"><h1>${esc(m.name || path.split("/").pop())} ${m.status ? `<span class="chip">${esc(m.status)}</span>` : ""}</h1>
      <div class="toolbar">${downloads(d.path, "Download session")}${d.has_report ? `<a class="btn" target="_blank" href="${fileUrl(d.path + "/report/index.html")}">Static report</a>` : ""}<a class="btn" href="#/compare?path=${q(d.path)}">Compare with...</a></div></div>
    <div class="cards">
      ${card("Task success", rate(ok, done.length), `${d.episodes.length} episodes`)}
      ${card("Agents", d.agents.length ? `${d.agents.length}` : "-", agentCost ? `$${agentCost.toFixed(2)} LLM cost` : "")}
      ${card("Duration", span ? mmss(span) : "-", m.started ? esc(m.started.replace("T", " ").slice(0, 19)) : "")}
      ${card("Your GPU peak", peak !== null ? `${peak.toFixed(1)} GiB` : "-", d.gpu ? `GPU ${esc(d.gpu.target)}` : "")}
      ${card("Code", `<span class="mono">${esc(m.git_commit || "-")}</span>`, m.git_dirty === true || m.git_dirty === "true" ? "uncommitted changes" : "")}
      ${card("Model", `<span class="small">${esc(m.model || "-")}</span>`, m.profile ? `profile ${esc(m.profile)}` : "")}
    </div>
    ${timed.length ? `<h2>Timeline</h2><div class="panel">${gantt(
      timed.map((s) => ({ start: s.start, end: s.end, lane: laneOf(s.stage), label: s.stage.replace(/^episode |^agent /, ""), cls: (s.status || "").toLowerCase(), title: `#${s.n} ${s.stage}: ${s.status} (${(s.end - s.start).toFixed(0)} s)\n${s.detail}` })),
      { lanes: [...new Set(timed.map((s) => laneOf(s.stage)))] },
    )}</div>` : ""}
    ${tabs("sess", ["Tasks", ...(d.failures && d.failures.length ? [`Failures (${d.failures.reduce((n, f) => n + f.count, 0)})`] : []), "Episodes", "Agents", "Stages", "GPU", "Logs & environment", "Session details"], "Tasks")}`;

  bindTabs("sess", (tab) => {
    if (tab === "Tasks") return tasksTable(d.tasks);
    if (tab.startsWith("Failures")) return failuresView(d.failures);
    if (tab === "Episodes") return episodesView(d.episodes);
    if (tab === "Agents") return agentsTable(d.agents);
    if (tab === "Stages") return stagesTable(stages);
    if (tab === "GPU") return gpuView(d.gpu, timed);
    if (tab === "Logs & environment") return filesView(d.files);
    if (tab === "Session details") return `<table class="table kv">${Object.entries(m).map(([k, v]) => `<tr><th>${esc(k)}</th><td class="mono small">${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table>`;
    return "";
  }, (tab) => {
    if (tab === "Episodes") bindEpisodeFilters(d.episodes);
    if (tab === "Tasks") bindTasks(d.tasks);
  });
}

const laneOf = (stage) => (stage.startsWith("episode") ? "episodes" : stage.startsWith("agent") ? "agents" : "setup");

// One row per task and phase. Seeds link to their replays; a row opens to every run, with
// what went wrong and downloads; ticked rows can be compared (e.g. two program versions).
function tasksTable(tasks) {
  if (!tasks.length) return `<p class="muted">No episodes.</p>`;
  return `<div class="toolbar"><button class="primary" id="task-compare" disabled>Compare selected</button>
      <button id="task-expand">Open all</button><span class="muted small">Click a row for its runs and downloads. Tick two rows (for example two program versions) to compare them.</span></div>
    <table class="table tasks"><thead><tr><th></th><th>Task</th><th>Agent / phase</th><th>Success</th><th>Program ran</th><th>Seeds</th><th>Mean steps</th><th>Mean wall s</th><th>Download</th></tr></thead><tbody>
    ${tasks.map((t, i) => `<tr class="task-row" data-i="${i}"><td data-stop><input type="checkbox" data-folder="${esc(t.folder)}"></td>
      <td><span class="caret">▸</span> ${esc(t.benchmark)} <b>${esc(t.suite)}:${esc(t.task_id)}</b><div class="muted small">${esc(t.task_name || "")}</div></td>
      <td>${t.agent ? `<span class="chip">${esc(t.agent)}</span> ` : ""}<span class="chip">${esc(t.phase)}</span></td>
      <td>${rate(t.successes, t.episodes)}</td><td>${t.program_ok}/${t.episodes}</td>
      <td data-stop>${t.runs.map(seedChip).join("")}</td>
      <td>${num(t.mean_steps, 0)}</td><td>${num(t.mean_wall_s)}</td>
      <td class="nowrap" data-stop><a class="btn small" href="${zipUrl(t.folder, false, groupName(t))}" title="Logs, traces and JSON of every run in this row">Logs</a> <a class="btn small" href="${zipUrl(t.folder, true, groupName(t))}" title="Everything, with videos and images">Full</a></td></tr>
      <tr class="task-detail" data-for="${i}" hidden><td></td><td colspan="8">${taskDetail(t)}</td></tr>`).join("")}</tbody></table>`;
}

const groupName = (t) => [t.suite, t.task_id, t.agent, t.phase].filter((x) => x !== null && x !== undefined && x !== "").join("-");

const seedChip = (r) => `<a class="seed ${r.task_success === true ? "ok" : r.finalized ? "bad" : ""}" href="#/episode?path=${q(r.path)}" title="${esc(r.task_success === true ? "solved" : r.error || r.termination || "not finished")}">${esc(r.seed ?? "?")}</a>`;

function taskDetail(t) {
  return `<table class="table compact runs"><thead><tr><th>Seed</th><th>Result</th><th>How it ended</th><th>Steps</th><th>Wall s</th><th>Error</th><th></th></tr></thead><tbody>
    ${t.runs.map((r) => `<tr><td><b>${esc(r.seed ?? "?")}</b></td>
      <td>${r.finalized ? badge(r.task_success, { yes: "solved", no: "not solved" }) : '<span class="badge warn">not finished</span>'}${r.program_ok === false ? ' <span class="badge bad">program error</span>' : ""}</td>
      <td class="small">${esc(r.termination || "")}</td><td>${r.steps ?? ""}</td><td>${num(r.wall_s)}</td>
      <td class="small err-cell">${r.error ? esc(r.error) : ""}</td>
      <td class="nowrap"><a class="btn small primary" href="#/episode?path=${q(r.path)}">Replay</a> <a class="btn small" href="${zipUrl(r.path, false)}">Logs</a> <a class="btn small" href="${zipUrl(r.path, true)}">Full</a>${r.program ? ` <a class="btn small" href="#/log?path=${q(r.program)}">Program</a>` : ""}</td></tr>`).join("")}
    </tbody></table>
    <div class="toolbar small"><span class="muted mono">${esc(t.folder)}</span>
      ${t.runs.some((r) => r.task_success !== true) ? `<a class="btn small" href="${zipUrl(t.runs.filter((r) => r.task_success !== true).map((r) => r.path), false, groupName(t) + "-unsolved")}">Logs of unsolved runs only</a>` : ""}</div>`;
}

function bindTasks(tasks) {
  // A session with no episode yet (just started) has no task table and no toolbar.
  if (!document.getElementById("task-expand")) return;
  const rows = [...document.querySelectorAll("tr.task-row")];
  const open = (row, on) => {
    const detail = document.querySelector(`tr.task-detail[data-for="${row.dataset.i}"]`);
    detail.hidden = on === undefined ? !detail.hidden : !on;
    row.classList.toggle("open", !detail.hidden);
  };
  rows.forEach((row) => (row.onclick = (e) => { if (!e.target.closest("[data-stop]")) open(row); }));
  const expand = document.getElementById("task-expand");
  expand.onclick = () => {
    const on = expand.textContent === "Open all";
    rows.forEach((row) => open(row, on));
    expand.textContent = on ? "Close all" : "Open all";
  };
  const compare = document.getElementById("task-compare");
  const boxes = [...document.querySelectorAll("tr.task-row input[data-folder]")];
  const picked = () => boxes.filter((b) => b.checked).map((b) => b.dataset.folder);
  boxes.forEach((b) => (b.onchange = () => {
    const n = picked().length;
    compare.disabled = n < 2;
    compare.textContent = n ? `Compare selected (${n})` : "Compare selected";
  }));
  compare.onclick = () => (location.hash = `#/compare?${picked().map((p) => `path=${q(p)}`).join("&")}`);
  void tasks;
}

// Every unsolved episode, grouped by how it failed (numbers and paths blanked so the same
// failure on different seeds lands in one row).
function failuresView(failures) {
  if (!failures || !failures.length) return `<p class="muted">Every finished episode solved its task.</p>`;
  return `<p class="muted small">Unsolved episodes grouped by how they failed, most common first. Click a seed to replay it.</p>
    <table class="table"><thead><tr><th>How it failed</th><th>Episodes</th><th>Which</th><th>Download</th></tr></thead><tbody>
    ${failures.map((f) => `<tr><td><b>${esc(f.reason)}</b>${f.example ? `<details><summary class="small muted">example error</summary><pre class="code small">${esc(f.example)}</pre></details>` : ""}</td>
      <td><b>${f.count}</b></td>
      <td>${f.episodes.map((e) => `<a class="seed bad" href="#/episode?path=${q(e.path)}" title="${esc([e.agent, e.phase].filter(Boolean).join(" · "))}">${esc(e.suite)}:${esc(e.task_id)} · ${esc(e.seed)}${e.phase && e.phase !== "episode" ? ` · ${esc(e.phase)}` : ""}</a>`).join("")}</td>
      <td class="nowrap"><a class="btn small" href="${zipUrl(f.episodes.map((e) => e.path), false, "failures")}">Logs</a></td></tr>`).join("")}</tbody></table>`;
}

function episodesView(episodes) {
  const phases = [...new Set(episodes.map((e) => e.phase))];
  return `<div class="toolbar"><select id="ep-result"><option value="all">all results</option><option value="ok">succeeded</option><option value="fail">not solved</option></select>
    <select id="ep-phase"><option value="all">all phases</option>${phases.map((p) => `<option>${esc(p)}</option>`).join("")}</select>
    <input id="ep-text" placeholder="filter by task, agent or seed" class="grow-input"></div><div id="ep-grid" class="ep-grid"></div>`;
}
function bindEpisodeFilters(episodes) {
  const draw = () => {
    const result = document.getElementById("ep-result").value;
    const phase = document.getElementById("ep-phase").value;
    const text = document.getElementById("ep-text").value.toLowerCase();
    const shown = episodes.filter((e) =>
      (result === "all" || (result === "ok" ? e.task_success === true : e.task_success !== true)) &&
      (phase === "all" || e.phase === phase) &&
      (!text || `${e.suite}:${e.task_id} ${e.task_name} ${e.agent || ""} seed ${e.seed}`.toLowerCase().includes(text)));
    document.getElementById("ep-grid").innerHTML = shown.map((e) => `
      <a class="ep-card ${e.task_success === true ? "is-ok" : e.finalized ? "is-bad" : ""}" href="#/episode?path=${q(e.path)}">
        ${e.thumb ? `<img loading="lazy" src="${fileUrl(e.thumb)}" alt="">` : `<div class="thumb-empty">no keyframes</div>`}
        <div class="ep-meta"><b>${esc(e.suite)}:${esc(e.task_id)}</b> seed ${esc(e.seed)} ${badge(e.task_success, { yes: "solved", no: "not solved" })}
          <div class="muted small">${esc(e.task_name || "")}</div>
          <div class="small">${e.agent ? `<span class="chip">${esc(e.agent)}</span>` : ""}<span class="chip">${esc(e.phase)}</span> ${e.steps ?? "?"} steps · ${num(e.wall_s)} s · ${e.calls} calls</div>
          <div class="muted small">${esc(e.termination || (e.finalized ? "" : "not finalized"))}</div></div></a>`).join("") || `<p class="muted">No matching episodes.</p>`;
  };
  ["ep-result", "ep-phase"].forEach((id) => (document.getElementById(id).onchange = draw));
  document.getElementById("ep-text").oninput = draw;
  draw();
}

function agentsTable(agents) {
  if (!agents.length) return `<p class="muted">No agent conversations in this run.</p>`;
  return `<table class="table"><thead><tr><th>Conversation</th><th>Result</th><th>Turns</th><th>Tool calls</th><th>Errors</th><th>Minutes</th><th>Cost</th></tr></thead><tbody>
    ${agents.map((a) => `<tr class="click" onclick="location.hash='#/agent?path=${q(a.path)}'"><td class="mono small">${esc(a.title)}</td><td>${esc(a.result || "")}</td><td>${a.turns ?? ""}</td><td>${a.tool_calls}</td><td>${a.tool_errors}</td><td>${a.minutes ?? ""}</td><td>${a.cost_usd != null ? "$" + num(a.cost_usd, 2) : ""}</td></tr>`).join("")}</tbody></table>`;
}

function stagesTable(stages) {
  if (!stages.length) return `<p class="muted">No stages.tsv in this folder.</p>`;
  return `<table class="table"><thead><tr><th>#</th><th>Stage</th><th>Status</th><th>Detail</th><th>Start s</th><th>Duration s</th><th>Your GPU peak GiB</th></tr></thead><tbody>
    ${stages.map((s) => `<tr><td>${s.n}</td><td>${esc(s.stage)}</td><td>${statusBadge(s.status)}</td><td class="small">${esc(s.detail)}</td><td>${num(s.start, 0)}</td><td>${s.start === null ? "" : num(s.end - s.start, 0)}</td><td>${num(s.peak)}</td></tr>`).join("")}</tbody></table>`;
}

function gpuView(gpu, timed) {
  if (!gpu) return `<p class="muted">No GPU timeline recorded.</p>`;
  const t = gpu.t;
  const index = (v) => { const i = t.findIndex((x) => x >= v); return i < 0 ? t.length - 1 : i; };
  const bands = timed.map((s) => ({ start: index(s.start), end: index(s.end), label: String(s.n), title: `#${s.n} ${s.stage}: ${s.status}` }));
  return lineChart({ "GPU total used (GiB)": gpu.used, "your services and simulators (GiB)": gpu.yours, "your reservation (GiB)": gpu.reserved }, `GPU ${gpu.target} memory (bands are stages; hover for names)`, { bands, x: `${t[t.length - 1].toFixed(0)} s` }) +
    lineChart({ "utilization % (all users)": gpu.util }, "GPU utilization", { height: 140, bands });
}

function filesView(files) {
  if (!files.length) return `<p class="muted">No log files.</p>`;
  return `<table class="table"><thead><tr><th>File</th><th>Size</th><th></th></tr></thead><tbody>
    ${files.map((f) => `<tr><td class="mono small">${esc(f.name)}</td><td class="small">${size(f.size)}</td>
      <td class="nowrap"><a class="btn small" href="#/log?path=${q(f.path)}">View</a> <a class="btn small" href="${fileUrl(f.path)}?download=1">Download</a></td></tr>`).join("")}</tbody></table>`;
}
