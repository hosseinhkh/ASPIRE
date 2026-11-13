// Trace2VLA: teacher (ASPIRE) -> demonstrations -> dataset -> fine-tuned student -> comparison.
// Experiments live in outputs/trace2vla/<name>/ and are written by scripts/gpu/t2v.sh.
import { api, esc, q, num, fileUrl, when, pct } from "../lib/util.js";
import { card, jsonView, rate, tabs, bindTabs } from "../lib/ui.js";
import { lineChart, moveMarkers, stepChart } from "../lib/charts.js";

const REFRESH_MS = 10000;
const FAMILY = {
  success: ["ok", "solved"],
  semantic: ["bad", "wrong object, placed"],
  physical: ["warn", "right object, not placed"],
  both: ["bad", "wrong object, not placed"],
  no_attempt: ["muted", "never closed on an object"],
  unknown: ["muted", "not graded"],
};
let expTab = null; // the first visit opens Pipeline while a collection exists, else Compare
let teacherSet = null;
let demoSet = null;

const percent = (v) => (v === null || v === undefined ? "-" : `${(100 * v).toFixed(1)}%`);
const minutes = (s) => (s === null || s === undefined ? "-" : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const zip = (path, media, label) => `<a class="btn" href="/download?path=${q(path)}&media=${media ? 1 : 0}">${esc(label)}</a>`;
const statusBadge = (s) => `<span class="badge ${{ running: "ok", finished: "ok", waiting: "warn", interrupted: "warn", stopped: "bad", failed: "bad", died: "bad" }[s] || "muted"}">${esc(s || "-")}${s === "running" ? " ●" : ""}</span>`;
// t2v.sh collect: the state of each (scene set, task), its background conversion, each task's program.
const CELL = { recording: ["ok", "recording ●"], recorded: ["ok", "recorded"], pending: ["muted", "to record"], incomplete: ["bad", "incomplete"], busy: ["warn", "another process has it"] };
const PREP = { queued: ["muted", "conversion queued"], running: ["ok", "converting ●"], finished: ["ok", "converted"], problems: ["warn", "conversion: see log"] };
const PROGRAM = {
  registered: ["ok", "registered"], taken: ["ok", "from its coding agent"], waiting: ["muted", "waiting"], agent_running: ["warn", "agent working ●"],
  no_program: ["bad", "agent wrote none"], agent_stopped: ["bad", "agent stopped early"], weak: ["bad", "too weak, not used"], unchecked: ["warn", "not checked on held-out scenes"],
};
const cardRate = (k, n) => {
  if (!n) return '<span class="muted">-</span>';
  const p = pct(k, n), cls = p >= 80 ? "ok" : p >= 40 ? "warn" : "bad";
  return `${k}/${n} <span class="muted" style="font-size:13px">${p}%</span><div class="rate"><span class="bar" style="width:120px"><span class="fill ${cls}" style="width:${p}%"></span></span></div>`;
};
const tag = (map, key) => (key && map[key] ? `<span class="badge ${map[key][0]}">${esc(map[key][1])}</span>` : "");
const collectNow = (c) => {
  const n = (c && c.now) || {};
  if (n.stage === "teacher") return `teacher on ${n.scene_set}, task ${n.task}: ${n.done ?? 0}/${n.total ?? "?"}`;
  return n.detail || (c && c.status) || "";
};
export const familyBadge = (f) => (f ? `<span class="badge ${(FAMILY[f] || ["muted"])[0]}" title="${esc((FAMILY[f] || ["", f])[1])}">${esc(f)}</span>` : "");
// Runs recorded before pi0.5 support name no student: they are SmolVLA's.
export const studentBadge = (label) => `<span class="chip" title="the student policy">${esc(label || "SmolVLA")}</span>`;
export const families = (counts) =>
  Object.entries(counts || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, n]) => `${familyBadge(name)} ${n}`)
    .join(" &nbsp;");
const span = (s) => (s ? `${s.min} / ${Math.round(s.mean)} / ${s.max}` : "-");

// ------------------------------------------------------------------------- list --
export async function renderList(app) {
  const data = await api("/api/t2v/experiments");
  app.innerHTML = `
    <div class="page-head"><h1>Trace2VLA</h1></div>
    <p class="muted small">ASPIRE as the teacher, SmolVLA or pi0.5 as the student (each evaluation, dataset and training run shows which). Each experiment keeps the teacher's programs and episodes,
      the demonstrations converted to the student's control, the datasets, the training runs and the student's evaluations,
      with a ledger of what produced what. Start one with <span class="mono">scripts/gpu/t2v.sh init</span>.</p>
    <div class="panel flush"><table class="table"><thead><tr><th>Experiment</th><th>Suite</th><th>Teacher (official scenes)</th><th>Collection</th><th>Demos</th><th>Datasets</th><th>Training</th><th>Student evaluations</th><th>Last event</th></tr></thead><tbody>
    ${
      data.items
        .map(
          (e) => `<tr class="click" data-href="#/t2v/exp?path=${q(e.path)}">
          <td><b>${esc(e.name)}</b><div class="muted small">tasks ${esc((e.tasks || []).join(", "))}</div></td>
          <td class="mono small">${esc(e.suite || "")}</td>
          <td>${e.teacher_official ? rate(e.teacher_official.successes, e.teacher_official.episodes) : '<span class="muted">not run</span>'}</td>
          <td class="small">${(() => {
            const c = e.collection || {}, tot = c.totals || {};
            const working = (e.agents_working || []).length ? `<div>coding agent at work: task ${esc(e.agents_working.join(", "))} <span class="badge warn">●</span></div>` : "";
            if (!tot.planned && !c.status) return working || '<span class="muted">-</span>';
            return `${rate(tot.recorded || 0, tot.planned || 0)}<div class="muted">${tot.kept || 0} demos kept · programs ${c.programs ?? 0}/${(e.tasks || []).length}</div>${c.status ? `<div>collect ${statusBadge(c.status)} <span class="muted">${esc(["running", "waiting"].includes(c.status) ? collectNow(c) : "")}</span></div>` : ""}${working}`;
          })()}</td>
          <td>${esc(e.demos)}</td><td>${esc(e.datasets)}</td>
          <td class="small">${e.training.map((t) => `${esc(t.name)} <span class="muted">${esc(t.student_label || "")}</span> ${statusBadge(t.status)}`).join("<br>")}</td>
          <td class="small">${e.evals.map((v) => `${esc(v.name)} <span class="muted">${esc(v.student_label || "")}</span>: <b>${percent(v.mean)}</b> ${v.status === "finished" ? "" : statusBadge(v.status)}`).join("<br>")}</td>
          <td class="small">${e.last_event ? `${esc(e.last_event.event)}<div class="muted">${esc(e.last_event.time)}</div>` : ""}</td></tr>`,
        )
        .join("") || `<tr><td colspan="9" class="muted">No experiment yet: <span class="mono">scripts/gpu/t2v.sh init preflight --suite libero_object_swap --tasks 0,1</span></td></tr>`
    }</tbody></table></div>`;
  app.querySelectorAll("tr.click").forEach((tr) => (tr.onclick = () => (location.hash = tr.dataset.href)));
}

// ------------------------------------------------------------------- experiment --
export async function renderExperiment(app, params, ctx) {
  const path = params.get("path");
  const d = await api(`/api/t2v/experiment?path=${q(path)}`);
  const suite = d.settings.suite, tasks = d.settings.tasks || [];
  const official = d.teacher.official;
  // Next to the teacher: full evaluations of the experiment's suite on the official scenes (a
  // chosen training step evaluated in full included). Short probes of each saved checkpoint
  // (t2v.sh eval-steps) appear only on their training run's curve (Training tab).
  const isStep = (e) => e.checkpoint_step !== null && e.checkpoint_step !== undefined;
  const isOfficial = (e) => e.suite === suite && (e.scene_set || "official") === "official";
  const main = d.evals.filter((e) => isOfficial(e) && !e.probe);
  const other = d.evals.filter((e) => !isOfficial(e) && !isStep(e) && !e.probe);
  // Before and after training, per student (SmolVLA and pi0.5 are compared separately).
  const studentLabels = [...new Set(main.map((e) => e.student_label || "SmolVLA"))];
  const progressCards = studentLabels.length
    ? studentLabels
        .map((label) => {
          const own = main.filter((e) => (e.student_label || "SmolVLA") === label);
          const before = own.find((e) => (e.checkpoint || "").startsWith("base"));
          const after = [...own].reverse().find((e) => !(e.checkpoint || "").startsWith("base"));
          return card(
            `${esc(label)}: before → after`,
            `<b>${before ? percent(before.mean) : "-"}</b> → <b>${after ? percent(after.mean) : "-"}</b>`,
            `${before ? `${before.successes}/${before.episodes} · ${esc(before.name)}` : "t2v.sh eval --checkpoint base"}<br>${after ? `${after.successes}/${after.episodes} · ${esc(after.name)} (${esc(after.checkpoint)})` : "after training: t2v.sh eval --checkpoint RUN"}`,
          );
        })
        .join("")
    : card("Student before → after", "-", "t2v.sh eval --checkpoint base");
  const demos = d.demos.reduce((n, s) => n + (s.accepted || 0), 0);
  const pipe = d.pipeline || { collect: {}, scene_sets: [], programs: {}, tasks: [], totals: {}, background: [], claims: [] };
  const collecting = ["running", "waiting", "starting"].includes((pipe.collect || {}).status);
  const agents = d.agents || [];
  const agentsWorking = agents.filter((a) => a.state === "working");
  const running = collecting || agentsWorking.length > 0 || pipe.background.length > 0 || d.training.some((t) => t.progress.status === "running") || d.evals.some((e) => e.progress.status === "running");
  if (!expTab) expTab = (pipe.collect || {}).status || agents.length || pipe.scene_sets.some((s) => s.in_collection) ? "Pipeline" : "Compare";
  const nowLines = [
    ...(collecting ? [`collect: ${esc(collectNow(pipe.collect))}`] : []),
    ...agentsWorking.map((a) => {
      const last = a.versions[a.versions.length - 1];
      return `coding agent, task ${esc(a.task)}: ${last ? `${esc(last.name)} ${last.running ? "running" : `${last.passes}/${last.seeds}`}` : "exploring the scene"}`;
    }),
    ...d.training.filter((t) => t.progress.status === "running").map((t) => `train ${esc(t.name)}: step ${t.progress.step ?? 0}/${t.progress.steps ?? "?"}${t.progress.eta_s ? `, ${minutes(t.progress.eta_s)} left` : ""}`),
    ...d.evals.filter((e) => e.progress.status === "running").map((e) => `eval ${esc(e.name)}: ${e.progress.done ?? 0}/${e.progress.planned ?? "?"}`),
    ...pipe.background.filter((b) => !["collect", "train", "eval", "eval-steps"].includes(b.stage)).map((b) => `${esc(b.stage)} (pid ${esc(b.pid)})`),
  ];
  teacherSet = teacherSet && d.teacher[teacherSet] ? teacherSet : Object.keys(d.teacher)[0] || null;
  demoSet = demoSet && d.demos.some((s) => s.name === demoSet) ? demoSet : (d.demos[0] || {}).name || null;

  app.innerHTML = `
    <div class="crumbs"><a href="#/t2v">Trace2VLA</a> / ${esc(d.name)}</div>
    <div class="page-head"><h1>${esc(d.name)} <span class="muted">${esc(suite)}</span></h1>
      <div class="toolbar"><span class="btn-group">${zip(d.path, false, "Download logs")}${zip(d.path, true, "Download all")}</span></div></div>
    <div class="cards">
      ${card("Now running", nowLines.length ? `${nowLines.length}` : "nothing", nowLines.join("<br>"))}
      ${card("Teacher on official scenes", official ? `<b>${percent(official.mean)}</b>` : "not run", official ? `${official.successes}/${official.episodes} episodes` : "t2v.sh teacher-run --scenes official")}
      ${progressCards}
      ${card("Demonstrations", `${demos}`, d.demos.map((s) => `${esc(s.name)}: ${s.accepted} kept, ${s.rejected} rejected`).join("<br>"))}
      ${card("Datasets", `${d.datasets.length}`, d.datasets.map((s) => `${esc(s.name)} <span class="muted">${esc(s.student_label || "")}</span>: ${s.episodes} episodes, ${s.frames} frames`).join("<br>"))}
      ${card("Training", `${d.training.length}`, d.training.map((t) => `${esc(t.name)} <span class="muted">${esc(t.student_label || "")}</span> ${statusBadge(t.progress.status)} ${t.progress.step ?? 0}/${t.progress.steps ?? "?"}`).join("<br>"))}
    </div>
    ${tabs("t2v-exp", ["Pipeline", "Compare", "Teacher", "Demos", "Training", "Evaluations", "Datasets", "Ledger"], expTab)}`;

  const taskName = (t) => `<b>${esc(suite)}:${esc(t)}</b><div class="muted small">${esc((d.settings.instructions || {})[String(t)] || "")}</div>`;
  const cell = (t) => (t ? rate(t.successes, t.episodes) : '<span class="muted">-</span>');

  // The coding agents writing this experiment's programs: each version's result on every layout.
  const layoutName = (suite) => {
    const alt = suite.match(/_alt(\d+)$/);
    return alt ? `alt${alt[1]}` : "official";
  };
  const batchCell = (b) => {
    if (!b) return '<span class="muted">-</span>';
    if (b.running) return '<span class="badge warn">running ●</span>';
    const per = Object.entries(b.layouts || {});
    const parts = per.map(([name, r]) => `${esc(layoutName(name))} <b>${r.error ? "?" : `${r.passes}/${r.seeds}`}</b>${!r.error && r.passes < r.seeds ? " ✗" : ""}`);
    return `${rate(b.passes || 0, b.seeds || 0)}${parts.length ? `<div class="small">${parts.join(" · ")}</div>` : ""}`;
  };
  const AGENT = { working: ["warn", "working ●"], finished: ["ok", "finished"], stopped: ["bad", "stopped"] };
  const agentsView = () =>
    agents.length
      ? `<h3>Coding agents</h3>
        <p class="muted small">Each agent writes one task's program. Every version runs on each layout (development seeds); the best one then runs once on held-out seeds. The collection needs a version that solves every layout.</p>
        <table class="table compact"><thead><tr><th>Task</th><th>Agent</th><th>Versions (development)</th><th>Held-out</th><th></th></tr></thead><tbody>
        ${agents
          .map(
            (a) => `<tr><td>${taskName(a.task)}</td>
          <td>${tag(AGENT, a.state)}<div class="small muted">${a.state === "working" ? `last activity ${esc(when(a.updated))}` : esc(a.outcome || "")}${a.cost_usd != null ? ` · $${num(a.cost_usd, 2)}` : ""}${a.turns ? ` · ${esc(a.turns)} turns` : ""}</div></td>
          <td>${a.versions.length ? a.versions.map((v) => `<div style="margin-bottom:4px"><span class="mono small">${esc(v.name)}</span> ${batchCell(v)}</div>`).join("") : '<span class="muted small">exploring the scene</span>'}</td>
          <td>${batchCell(a.heldout)}${a.program ? "" : '<div class="small muted">no program_best.py yet</div>'}</td>
          <td class="small nowrap">${a.transcript ? `<a href="#/agent?path=${q(a.transcript)}">conversation</a><br>` : ""}${a.session ? `<a href="#/session?path=${q(a.session)}">session</a>` : ""}</td></tr>`,
          )
          .join("")}
        </tbody></table>`
      : "";

  // Data collection: per task and scene set, what the teacher recorded and solved, what was graded
  // and what became demonstrations; what t2v.sh collect is doing; what runs right now.
  const pipeline = () => {
    const c = pipe.collect || {}, tot = pipe.totals || {}, now = c.now || {};
    const order = c.scene_sets || [];
    const rank = (s) => (order.includes(s.name) ? order.indexOf(s.name) : order.length + pipe.scene_sets.indexOf(s));
    const sets = [...pipe.scene_sets].sort((a, b) => rank(a) - rank(b));
    const hasProgram = (task) => (pipe.programs[String(task)] || {}).registered;
    const agentsBlock = agentsView();
    const start = `<pre class="mono small">scripts/gpu/t2v.sh collect --detach ${esc(d.name)} --scenes alt1-train,alt2-train,alt3-train,alt4-train,train --workers 2 --prepare-workers 4</pre>`;
    if (!sets.length && !c.status && agents.length) return agentsBlock;
    if (!sets.length && !c.status)
      return `<p class="muted">Nothing collected yet. One command records the teacher on every scene set, task by task, and grades and converts each finished one in the background (add <span class="mono">--programs-from SESSION --follow</span> to take each task's program from its coding agent as it finishes):</p>${start}`;
    const banner = c.status
      ? `<div class="panel"><div class="panel-head"><b>t2v.sh collect</b> ${statusBadge(c.status)} <span class="muted small">started ${esc(when(c.started))} · updated ${esc(when(c.updated))}${c.programs_from ? ` · programs from ${c.session_path ? `<a href="#/session?path=${q(c.session_path)}">${esc(c.session_path.split("/").pop())}</a>` : `<span class="mono">${esc(c.programs_from)}</span>`}` : ""}${c.follow ? " · following" : ""}${c.log_path ? ` · <a href="#/log?path=${q(c.log_path)}">full log</a>` : ""}</span></div>
        ${now.stage === "teacher" ? `<p>Recording <b>${esc(now.scene_set)}</b>, task <b>${esc(now.task)}</b>: ${rate(now.done || 0, now.total || 0)}</p>` : now.detail ? `<p>${esc(now.detail)}</p>` : ""}
        ${c.status === "failed" ? `<div class="alert bad">${esc(c.error || "failed")}. Run the same command again: it continues where it stopped.</div>` : ""}
        ${c.status === "died" ? `<div class="alert bad">Its process is gone without a word (killed, or the machine restarted). Run the same command again: it continues where it stopped.</div>` : ""}
        ${c.summary && (c.summary.incomplete || []).length ? `<div class="alert warn">Incomplete: ${esc(c.summary.incomplete.join(", "))} (hover the cell for why).</div>` : ""}
        <p class="small">${[
          c.episodes_left ? `${esc(c.episodes_left)} teacher episodes left for the tasks that have a program` : "",
          c.seconds_per_episode ? `${num(c.seconds_per_episode, 0)} s per episode so far` : "",
          c.eta_s && collecting ? `about ${minutes(c.eta_s)} of recording to go` : "",
          (c.prepare_running || []).length ? `converting ${esc(c.prepare_running.join(", "))}` : "",
          (c.prepare_queued || []).length ? `${c.prepare_queued.length} conversions queued` : "",
        ].filter(Boolean).join(" · ")}</p></div>`
      : "";
    const cards = `<div class="cards">
      ${card("Teacher episodes", cardRate(tot.recorded || 0, tot.planned || 0), "recorded of planned (training and dev scenes)")}
      ${card("Solved by the teacher", cardRate(tot.solved || 0, tot.recorded || 0), "only these can become demonstrations")}
      ${card("Graded", cardRate(tot.graded || 0, tot.recorded || 0), "replayed in joint control; failure labels")}
      ${card("Demonstrations kept", cardRate(tot.kept || 0, tot.solved || 0), `of the solved episodes; ${tot.rejected || 0} rejected at conversion`)}
      ${card("Tasks with a program", `${Object.values(pipe.programs).filter((x) => x.registered).length}/${pipe.tasks.length}`, Object.entries(pipe.programs).filter(([, x]) => !x.registered).map(([k]) => `task ${esc(k)}`).join(", ") || "all")}
    </div>`;
    const cellHtml = (s, cell) => {
      if (!cell) return '<td class="muted">-</td>';
      const live = now.stage === "teacher" && now.scene_set === s.name && Number(now.task) === cell.task;
      const recorded = Math.min(cell.planned, cell.recorded + (live ? now.done || 0 : 0));
      const log = cell.prepare_log ? ` <a class="small" href="#/log?path=${q(`${d.path}/${cell.prepare_log}`)}">log</a>` : "";
      if (!recorded && !live && cell.state !== "incomplete")
        return `<td class="muted small" title="${esc(cell.note || "")}">0/${esc(cell.planned)} · ${hasProgram(cell.task) ? (cell.state === "busy" ? "another process has it" : "to record") : "no program yet"}</td>`;
      const made = s.role === "test" ? "" : ` · <b>${cell.kept}</b> kept${cell.rejected ? ` · ${cell.rejected} rejected` : ""}`;
      return `<td title="${esc(cell.note || "")}">${rate(recorded, cell.planned)}
        <div class="small">${cell.solved} solved · ${cell.graded} graded${made}</div>
        <div>${tag(CELL, live ? "recording" : cell.state)} ${tag(PREP, cell.prepare)}${log}</div></td>`;
    };
    const sum = (s, key) => s.cells.reduce((n, cell) => n + (cell[key] || 0), 0);
    const matrix = `<h3>Per task and scene set</h3>
      <p class="muted small">Each cell: teacher episodes recorded of planned; how many the teacher solved, how many were graded, and how many became demonstrations (kept). Hover a cell for a note.</p>
      <div style="overflow-x:auto"><table class="table compact"><thead><tr><th>Task</th>${sets.map((s) => `<th>${esc(s.name)}${c.status && !s.in_collection ? ' <span class="muted small">(not in this collection)</span>' : ""}<div class="muted small">${esc(s.role)} · ${esc(s.suite || "")}<br>${esc(s.init_mode || "")} ${esc(s.seeds || "")}</div></th>`).join("")}</tr></thead><tbody>
      ${pipe.tasks.map((t, i) => `<tr><td>${taskName(t)}</td>${sets.map((s) => cellHtml(s, s.cells[i])).join("")}</tr>`).join("")}
      <tr><td><b>all tasks</b></td>${sets.map((s) => `<td>${rate(Math.min(sum(s, "recorded"), sum(s, "planned")), sum(s, "planned"))}<div class="small">${s.role === "test" ? "test scenes: graded, never converted" : `<b>${sum(s, "kept")}</b> demos kept${s.demo_sets.length ? ` in <span class="mono">${esc(s.demo_sets.join(", "))}</span>` : ""}`}</div></td>`).join("")}</tr>
      </tbody></table></div>`;
    const programs = `<h3>Teacher programs</h3><table class="table compact"><thead><tr><th>Task</th><th>Program</th><th>Note</th></tr></thead><tbody>
      ${pipe.tasks.map((t) => {
        const g = pipe.programs[String(t)] || {};
        const state = g.registered ? (g.state === "taken" ? "taken" : "registered") : g.state || "waiting";
        return `<tr><td>${taskName(t)}</td><td>${tag(PROGRAM, state)} <span class="mono small">${esc((g.sha256 || "").slice(0, 10))}</span>${g.agent_path ? ` <a class="small" href="#/agent?path=${q(g.agent_path)}">agent's conversation</a>` : ""}</td><td class="small">${esc(g.note && g.note !== "registered" ? g.note : g.registered ? g.source || "" : "t2v.sh teacher-add, or collect --programs-from SESSION")}</td></tr>`;
      }).join("")}</tbody></table>`;
    const active = `<h3>Running now</h3>${
      pipe.background.length || pipe.claims.length
        ? `<table class="table compact"><thead><tr><th>What</th><th>Detail</th></tr></thead><tbody>
          ${pipe.background.map((b) => `<tr><td><span class="chip">${esc(b.stage)}</span></td><td class="small">background stage (pid ${esc(b.pid)}); stop: <span class="mono">t2v.sh stop ${esc(d.name)} ${esc(b.stage)}</span></td></tr>`).join("")}
          ${pipe.claims.map((x) => `<tr><td class="mono small">${esc(x.key)}</td><td class="small">${esc(x.holder)}</td></tr>`).join("")}</tbody></table>`
        : '<p class="muted small">No background stage of this experiment is running.</p>'
    }`;
    const log = (c.log || []).length ? `<details ${collecting ? "open" : ""}><summary class="muted">collect: latest lines</summary><pre class="log">${c.log.map((l) => `<div>${esc(l)}</div>`).join("")}</pre></details>` : "";
    return banner + agentsBlock + cards + matrix + programs + active + log + (c.status ? "" : start);
  };

  const compare = () => {
    const teacherTasks = Object.fromEntries(((official || {}).tasks || []).map((t) => [t.task_id, t]));
    return `
    <p class="muted small">Every column is measured on the same scenes of each task. The teacher runs in joint control with up to 1,000 steps;
      the student runs in hand-movement control with LIBERO's step limit.</p>
    <table class="table"><thead><tr><th>Task</th><th>Teacher (official)</th>${main.map((e) => `<th><a href="#/vla/run?path=${q(e.path)}">${esc(e.name)}</a><div class="small">${studentBadge(e.student_label)}</div><div class="muted small">${esc(e.checkpoint || "")}</div></th>`).join("")}</tr></thead><tbody>
      ${tasks.map((t) => `<tr><td>${taskName(t)}</td><td>${cell(teacherTasks[t])}</td>${main.map((e) => `<td>${cell(e.tasks.find((x) => x.task_id === t))}</td>`).join("")}</tr>`).join("")}
      <tr><td><b>mean</b></td><td><b>${official ? percent(official.mean) : "-"}</b></td>${main.map((e) => `<td><b>${percent(e.mean)}</b> ${e.progress.status === "finished" ? "" : statusBadge(e.progress.status)}</td>`).join("")}</tr>
      <tr><td class="muted">student failures</td><td></td>${main.map((e) => `<td class="small">${families(e.failure_families) || '<span class="muted">-</span>'}</td>`).join("")}</tr>
    </tbody></table>
    ${
      other.length
        ? `<h3>Retention and other scene sets</h3><table class="table compact"><thead><tr><th>Evaluation</th><th>Student</th><th>Suite</th><th>Scene set</th><th>Checkpoint</th><th>Success</th><th>Failures</th></tr></thead><tbody>
        ${other.map((e) => `<tr class="click" data-href="#/vla/run?path=${q(e.path)}"><td>${esc(e.name)}</td><td>${studentBadge(e.student_label)}</td><td class="mono small">${esc(e.suite)}</td><td class="small">${esc(e.scene_set || "")}</td><td class="small">${esc(e.checkpoint || "")}</td><td>${rate(e.successes || 0, e.episodes || 0)}</td><td class="small">${families(e.failure_families)}</td></tr>`).join("")}</tbody></table>`
        : ""
    }
    <h3>Same scene, side by side</h3>
    <div class="toolbar"><select id="sbs-task">${tasks.map((t) => `<option value="${t}">task ${t}</option>`).join("")}</select>
      <label class="small">official scene <input id="sbs-scene" type="number" min="1" max="50" value="1" style="width:5em"></label>
      <button class="primary" id="sbs-go">Show</button></div>
    <div id="sbs" class="panel"><p class="muted small">Pick a task and a scene to see the teacher's and the student's videos for it.</p></div>
    <p class="muted small">Failure families (offline grading from simulator state): ${Object.entries(FAMILY).filter(([k]) => !["success", "unknown"].includes(k)).map(([k, v]) => `${familyBadge(k)} ${esc(v[1])}`).join("; ")}.</p>`;
  };

  const teacher = () => {
    const names = Object.keys(d.teacher);
    if (!names.length) return `<p class="muted">The teacher has not run yet: <span class="mono">scripts/gpu/t2v.sh teacher-run ${esc(d.name)} --scenes official</span></p>`;
    const s = d.teacher[teacherSet];
    const sets = d.settings.scene_sets || {};
    return `<div class="toolbar"><select id="t-set">${names.map((n) => `<option ${n === teacherSet ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>
        <span class="muted small">${esc((sets[teacherSet] || {}).role || "")} scenes: ${esc((sets[teacherSet] || {}).init_mode || "")} seeds ${esc((sets[teacherSet] || {}).seeds || "")}</span></div>
      <table class="table"><thead><tr><th>Task</th><th>Success</th><th>Steps when solved (min / mean / max)</th><th>Unsolved episodes ended as</th><th>Failure families</th><th>Graded</th><th>Program</th></tr></thead><tbody>
      ${s.tasks.map((t) => `<tr><td>${taskName(t.task_id)}</td><td>${rate(t.successes, t.episodes)}</td><td>${span(t.steps_when_successful)}</td>
        <td class="small">${Object.entries(t.terminations || {}).map(([k, v]) => `${esc(k)} ${v}`).join(", ") || "-"}</td><td class="small">${families(t.failure_families) || "-"}</td>
        <td class="small">${t.graded}/${t.episodes}${(t.not_reproduced || []).length ? ` <span class="badge bad" title="the recording did not reproduce in replay">${t.not_reproduced.length} not reproduced</span>` : ""}</td>
        <td class="mono small">${esc(((d.programs[String(t.task_id)] || {}).sha256 || "").slice(0, 10))}</td></tr>`).join("")}</tbody></table>
      <h3>Episodes</h3><p class="muted small">Replay opens ASPIRE's recording: both camera videos, every tool call, SAM3 masks, grasps and point clouds.</p>
      <table class="table compact"><thead><tr><th>Task</th><th>Scene (seed)</th><th>Outcome</th><th>Steps</th><th>Ended as</th><th>Grading</th><th></th></tr></thead><tbody>
      ${s.tasks.flatMap((t) => Object.entries(t.seeds || {}).map(([seed, e]) => `<tr class="click" data-href="#/episode?path=${q(e.run)}"><td>${esc(t.task_id)}</td><td>${esc(seed)}</td>
        <td>${e.success ? '<span class="badge ok">solved</span>' : '<span class="badge bad">not solved</span>'}</td><td>${esc(e.steps ?? "")}</td><td class="small">${esc(e.termination || "")}</td><td>${familyBadge(e.family)}</td><td class="small">Replay ▶</td></tr>`)).join("")}</tbody></table>`;
  };

  const demosView = () => {
    if (!d.demos.length) return `<p class="muted">No demonstrations yet: <span class="mono">scripts/gpu/t2v.sh convert ${esc(d.name)} --scenes train</span></p>`;
    const s = d.demos.find((x) => x.name === demoSet);
    const st = s.settings || {};
    return `<div class="toolbar"><select id="d-set">${d.demos.map((x) => `<option ${x.name === demoSet ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select>
        <span class="muted small">speed-up ${esc(((st.track || {}).speed ?? "?"))}×, idle frames ${st.drop_idle === false ? "kept" : "dropped when the replay still solves the task"}</span></div>
      <table class="table"><thead><tr><th>Task</th><th>Kept</th><th>Rejected</th><th>Steps (min / mean / max)</th><th>Teacher steps (mean)</th><th>Idle frames dropped (mean)</th><th>Kept all idle frames</th><th>Over the student's limit</th></tr></thead><tbody>
      ${(s.tasks || []).map((t) => `<tr><td>${taskName(t.task_id)}</td><td>${esc(t.accepted)}</td><td>${esc(t.rejected)} <span class="muted small">${esc((t.rejected_reasons || []).join(", "))}</span></td><td>${span(t.steps)}</td>
        <td>${num(t.teacher_steps_mean, 0)}</td><td>${num(t.idle_dropped_mean, 1)}</td><td>${esc(t.kept_all_idle)}</td><td>${esc(t.over_student_limit)}</td></tr>`).join("")}</tbody></table>
      <h3>Episodes</h3><div id="demo-list"><p class="muted">Loading...</p></div>`;
  };

  // Evaluations of the checkpoints a run saved during training (t2v.sh eval-steps): one column
  // and one line per probe (suite and scene set), one row per step; step 0 is where it started.
  const stepEvals = (run) => {
    const rows = d.evals.filter((e) => e.training_run === run && e.checkpoint_step !== null && e.checkpoint_step !== undefined);
    if (!rows.length) return "";
    const probeOf = (e) => `${e.suite} · ${e.scene_set || "?"}${e.scene_list ? ` (${e.scene_list.length} scenes)` : ""}${e.probe ? "" : " · full"}`;
    const probes = [...new Set(rows.map(probeOf))].sort();
    const steps = [...new Set(rows.map((e) => e.checkpoint_step))].sort((a, b) => a - b);
    const at = (probe, step) => rows.find((e) => probeOf(e) === probe && e.checkpoint_step === step);
    const series = Object.fromEntries(
      probes.map((p) => [p, rows.filter((e) => probeOf(e) === p && e.episodes).map((e) => [e.checkpoint_step, 100 * (e.mean || 0), `${e.successes}/${e.episodes}`])]),
    );
    const cell = (e) =>
      e
        ? `<td class="click-cell"><a href="#/vla/run?path=${q(e.path)}">${rate(e.successes || 0, e.episodes || 0)}</a> ${e.progress.status === "finished" ? "" : statusBadge(e.progress.status)}<div class="small">${families(e.failure_families)}</div></td>`
        : '<td class="muted">-</td>';
    return `<h3>Checkpoints during training</h3>
      <p class="muted small">Each saved checkpoint on short probes (<span class="mono">t2v.sh eval-steps</span>); step 0 is the checkpoint the run started from. Rising on the swap suite and falling on the standard one is the trade-off between learning the new layouts and forgetting the old skill. Choose a step here, then evaluate it in full on the official scenes.</p>
      ${stepChart(series, "Success by training step")}
      <table class="table compact"><thead><tr><th>Step</th>${probes.map((p) => `<th>${esc(p)}</th>`).join("")}</tr></thead><tbody>
      ${steps.map((step) => `<tr><td><b>${step}</b>${step === 0 ? ' <span class="muted small">start</span>' : ""}</td>${probes.map((p) => cell(at(p, step))).join("")}</tr>`).join("")}</tbody></table>`;
  };
  const training = () =>
    d.training.length
      ? d.training
          .map((t) => {
            const r = t.record, s = r.settings || {}, p = t.progress;
            return `<div class="panel"><div class="panel-head"><b>${esc(t.name)}</b> ${studentBadge(t.student_label)} ${statusBadge(p.status)}
              <span class="muted small">${esc(s.method)} · ${esc(s.steps)} steps · batch ${esc(s.batch_size)} · lr ${esc(s.lr)} · dataset ${esc(String(r.dataset || "").split("/").pop())} (${esc(r.dataset_episodes)} episodes, ${esc(r.dataset_frames)} frames)</span></div>
              <div class="cards">${card("Step", `${p.step ?? 0}/${p.steps ?? s.steps}`, p.status === "running" ? `ETA ${minutes(p.eta_s)}` : `took ${minutes(r.wall_s ?? p.elapsed_s)}`)}
                ${card("Loss", num(p.loss ?? r.final_loss, 3), t.loss.length ? `first logged ${num(t.loss[0], 3)}` : "")}
                ${card("GPU", esc(r.gpu ?? "-"), `started ${esc(r.started || "")}`)}
                ${card("Checkpoint", r.checkpoint ? "saved" : "-", r.checkpoint ? `<span class="mono small">${esc(r.checkpoint.split("/").slice(-5).join("/"))}</span>` : esc(r.error || ""))}</div>
              ${lineChart({ loss: t.loss }, "Training loss", { x: `logged every ${s.log_freq} steps`, height: 170 })}
              ${lineChart({ "learning rate": t.lr }, "Learning rate", { height: 110 })}
              ${stepEvals(t.name)}
              ${t.log ? `<p><a href="#/log?path=${q(t.log)}">Open train.log</a></p>` : ""}
              <details><summary class="muted">last log lines</summary><pre class="log">${t.log_tail.map((l) => `<div>${esc(l)}</div>`).join("")}</pre></details>
              <details><summary class="muted">settings and command</summary>${jsonView(r, { open: 1 })}</details></div>`;
          })
          .join("")
      : `<p class="muted">No training run yet: <span class="mono">scripts/gpu/t2v.sh train ${esc(d.name)} --dataset NAME --name RUN</span></p>`;

  const evals = () =>
    d.evals.length
      ? `<table class="table"><thead><tr><th>Evaluation</th><th>Student</th><th>Checkpoint</th><th>Suite</th><th>Scenes</th><th>Status</th><th>Success</th><th>Failures</th></tr></thead><tbody>
        ${d.evals.map((e) => `<tr class="click" data-href="#/vla/run?path=${q(e.path)}"><td><b>${esc(e.name)}</b></td><td>${studentBadge(e.student_label)}</td><td class="small">${esc(e.checkpoint || "")}</td><td class="mono small">${esc(e.suite)}</td><td class="small">${esc(e.scene_set || "")}</td>
          <td>${statusBadge(e.progress.status)} <span class="small">${e.progress.done ?? 0}/${e.progress.planned ?? "?"}</span></td><td>${rate(e.successes || 0, e.episodes || 0)}</td><td class="small">${families(e.failure_families)}</td></tr>`).join("")}</tbody></table>
        <p class="muted small">Click a row for the per-task table, every episode's video and its charts.</p>`
      : `<p class="muted">No evaluation yet: <span class="mono">scripts/gpu/t2v.sh eval ${esc(d.name)} --name before --checkpoint base</span></p>`;

  const datasets = () =>
    d.datasets.length
      ? d.datasets
          .map((s) => {
            const f = ((s.statistics || {}).frozen || {}).action;
            return `<div class="panel"><div class="panel-head"><b>${esc(s.name)}</b> ${studentBadge(s.student_label)}${s.derived_from ? ` <span class="muted small">frames of ${esc(s.derived_from)}</span>` : ""}${(s.demo_sets || []).length ? ` <span class="muted small">from ${esc(s.demo_sets.join(", "))}</span>` : ""} <span class="muted small">${esc(s.episodes)} episodes · ${esc(s.frames)} frames · per task ${esc(JSON.stringify(s.per_task))} · ${esc(s.rejected)} rejected at build · ${esc(s.created || "")}</span></div>
              <p class="small">Normalisation statistics: <span class="mono">${esc((s.statistics || {}).source || "")}</span></p>
              ${f ? `<table class="table compact"><thead><tr><th>Action</th>${["dx", "dy", "dz", "drx", "dry", "drz", "gripper"].map((n) => `<th>${n}</th>`).join("")}</tr></thead><tbody>
                <tr><td>std used (checkpoint)</td>${f.checkpoint_std.map((v) => `<td>${num(v, 3)}</td>`).join("")}</tr>
                <tr><td>std of this data</td>${(f.data_std || []).map((v) => `<td>${num(v, 3)}</td>`).join("")}</tr></tbody></table>` : ""}
              <p class="mono small">${esc(s.path)}</p></div>`;
          })
          .join("")
      : `<p class="muted">No dataset yet: <span class="mono">scripts/gpu/t2v.sh dataset ${esc(d.name)} --demos NAME --name NAME</span></p>`;

  const ledger = () =>
    `<p class="muted small">Every stage appends one event: what ran, on what, and what it produced. ${d.report ? `<a href="${fileUrl(d.report)}" target="_blank">Open the report</a>.` : ""}
      ${d.logs.length ? "Background logs: " + d.logs.map((l) => `<a href="#/log?path=${q(l.path)}">${esc(l.name)}</a>`).join(", ") : ""}</p>
     <table class="table compact"><thead><tr><th>Time</th><th>Event</th><th>Details</th></tr></thead><tbody>
     ${[...d.ledger].reverse().map((e) => {
       const { time, event, host, pid, ...rest } = e;
       return `<tr><td class="small nowrap">${esc(time)}</td><td><span class="chip">${esc(event)}</span></td><td class="small mono">${esc(JSON.stringify(rest).slice(0, 400))}</td></tr>`;
     }).join("")}</tbody></table>`;

  const bodies = { Pipeline: pipeline, Compare: compare, Teacher: teacher, Demos: demosView, Training: training, Evaluations: evals, Datasets: datasets, Ledger: ledger };
  const reopen = (name) => document.querySelector(`#t2v-exp button[data-tab="${name}"]`).click();
  bindTabs("t2v-exp", (name) => bodies[name](), async (name) => {
    expTab = name;
    const body = document.getElementById("t2v-exp-body");
    body.querySelectorAll("tr.click[data-href]").forEach((tr) => (tr.onclick = () => (location.hash = tr.dataset.href)));
    const tset = document.getElementById("t-set");
    if (tset) tset.onchange = () => { teacherSet = tset.value; reopen("Teacher"); };
    const dset = document.getElementById("d-set");
    if (dset) dset.onchange = () => { demoSet = dset.value; reopen("Demos"); };
    const go = document.getElementById("sbs-go");
    if (go) go.onclick = () => sideBySide(d.path, document.getElementById("sbs-task").value, document.getElementById("sbs-scene").value);
    if (name === "Demos" && demoSet) {
      const list = await api(`/api/t2v/demos?path=${q(d.demos.find((x) => x.name === demoSet).path)}`);
      const box = document.getElementById("demo-list");
      if (!box) return;
      box.innerHTML = `<table class="table compact"><thead><tr><th>Task</th><th>Seed</th><th>Result</th><th>Teacher steps</th><th>Re-executed</th><th>Stored</th><th>Idle dropped</th><th>Dwell kept</th><th>Tracking lag</th><th></th></tr></thead><tbody>
        ${list.episodes.map((e) => `<tr ${e.accepted ? `class="click" data-href="#/t2v/demo?path=${q(e.path)}"` : ""}><td>${esc(e.task_id)}</td><td>${esc(e.seed)}</td>
          <td>${e.accepted ? '<span class="badge ok">kept</span>' : `<span class="badge bad">rejected</span> <span class="small">${esc(e.rejected || "")}</span>`}</td>
          <td>${esc(e.teacher_steps ?? "")}</td><td>${esc(e.tracked_steps ?? "")}</td><td>${esc(e.steps ?? "")}${e.within_student_limit === false ? ' <span class="badge warn">over limit</span>' : ""}</td>
          <td>${e.accepted ? esc(e.tracked_steps - e.steps) : ""}</td><td>${esc(e.dwell ?? "")}</td><td class="small">${e.lag_mean_mm ? num(e.lag_mean_mm) + " mm" : ""}</td><td>${e.accepted ? "▶" : ""}</td></tr>`).join("")}</tbody></table>`;
      box.querySelectorAll("tr.click").forEach((tr) => (tr.onclick = () => (location.hash = tr.dataset.href)));
    }
  });
  if (running) ctx.timers.push(setTimeout(() => renderExperiment(app, params, ctx), REFRESH_MS));
}

async function sideBySide(path, task, scene) {
  const box = document.getElementById("sbs");
  box.innerHTML = `<p class="muted">Loading...</p>`;
  const s = await api(`/api/t2v/scene?path=${q(path)}&task=${q(task)}&scene=${q(scene)}`);
  const video = (src, caption) => `<figure style="flex:1 1 320px;max-width:520px;margin:0"><video controls muted playsinline src="${fileUrl(src)}" style="width:100%;border-radius:8px;background:#000"></video><figcaption class="small">${caption}</figcaption></figure>`;
  const teacher = s.teacher
    ? (s.teacher.videos.length ? s.teacher.videos.slice(0, 1) : [null]).map((v) =>
        v ? video(v, `<b>Teacher</b> ${s.teacher.success ? '<span class="badge ok">solved</span>' : '<span class="badge bad">not solved</span>'} ${esc(s.teacher.steps ?? "")} steps · <a href="#/episode?path=${q(s.teacher.run)}">full replay</a>`)
          : `<p class="muted">The teacher's run has no video (<a href="#/episode?path=${q(s.teacher.run)}">replay</a>).</p>`).join("")
    : `<p class="muted">The teacher has not run on official scene ${esc(scene)} of task ${esc(task)}.</p>`;
  const students = s.students
    .map((e) => (e.video ? video(e.video, `<b>${esc(e.eval)}</b> ${esc(e.student_label || "")} seed ${esc(e.seed)} ${e.success ? '<span class="badge ok">solved</span>' : '<span class="badge bad">failed</span>'} ${familyBadge(e.family)} ${e.chosen ? `took ${esc(e.chosen)}` : ""} · ${esc(e.steps)} steps · <a href="#/vla/episode?path=${q(e.path)}">episode</a>`) : ""))
    .join("");
  box.innerHTML = `<div style="display:flex;gap:12px;flex-wrap:wrap">${teacher}${students || '<p class="muted">No student evaluation has this scene.</p>'}</div>`;
}

// ------------------------------------------------------------------------- demo --
export async function renderDemo(app, params) {
  const d = await api(`/api/t2v/demo?path=${q(params.get("path"))}`);
  const e = d.summary, s = d.series, tr = e.tracking || {};
  const pick = (series, keys) => Object.fromEntries(keys.filter((k) => series && series[k]).map((k) => [k, series[k]]));
  const charts = [
    s.action && lineChart(pick(s.action, ["dx", "dy", "dz"]), "Hand position command (stored actions, -1..1)", { x: "step" }),
    s.action && lineChart(pick(s.action, ["drx", "dry", "drz"]), "Hand rotation command", { x: "step" }),
    s.action && lineChart(pick(s.action, ["gripper"]), "Gripper command (-1 open, +1 close)", { x: "step", height: 120 }),
    s.eef_pos && lineChart(s.eef_pos, "Hand position (m, world)", { x: "step" }),
  ].filter(Boolean);
  app.innerHTML = `
    <div class="crumbs"><a href="#/t2v">Trace2VLA</a> / <a href="#/t2v/exp?path=${q(d.experiment)}">${esc(d.experiment.split("/").pop())}</a> / demo ${esc(d.demo_set)}</div>
    <div class="page-head"><h1>task ${esc(e.scene.task_id)} <span class="muted">scene ${esc(e.scene.init_mode)} ${esc(e.scene.seed)}</span> <span class="badge ok">kept</span></h1></div>
    <div class="instruction">“${esc(e.instruction)}”</div>
    <div class="cards">
      ${card("Steps", `${e.teacher_steps} → ${e.steps}`, `teacher → stored; re-executed in ${e.tracked_steps} at ${esc(((e.settings || {}).track || {}).speed)}×`)}
      ${card("Idle frames", `${e.tracked_steps - e.steps} dropped`, `${e.idle_frames} idle found; dwell kept: ${esc(e.dwell)}${(e.idle_tried || []).length ? "<br>tried: " + e.idle_tried.map((t) => `dwell ${t.dwell} ${t.success ? "✓" : "✗"}`).join(", ") : ""}`)}
      ${card("Student limit", e.within_student_limit ? "within" : "over", `${e.steps} of ${e.student_step_limit} steps`)}
      ${card("Tracking", tr.lag_mean_mm !== undefined ? `${num(tr.lag_mean_mm)} mm lag` : "-", `max ${num(tr.lag_max_mm)} mm · start gap ${num(tr.start_gap_mm, 2)} mm · ${percent(tr.saturated_fraction)} saturated`)}
      ${card("Grading", (e.labels || {}).failure_family || "-", (e.labels || {}).chosen ? `took ${esc(e.labels.chosen)}, placed ${esc((e.labels.placed || []).join(", "))}` : "")}
    </div>
    <div class="player"><div class="videos">
      ${d.video ? `<figure style="flex-basis:520px;max-width:760px"><video id="demo-video" controls muted playsinline src="${fileUrl(d.video)}"></video><figcaption class="muted small"><b>Stored demonstration</b>: agentview | wrist, as the student sees them, in the student's control</figcaption></figure>` : ""}
      ${d.teacher && d.teacher.videos.length ? `<figure style="flex-basis:420px;max-width:620px"><video controls muted playsinline src="${fileUrl(d.teacher.videos[0])}"></video><figcaption class="muted small"><b>Teacher's own run</b> (joint control, ${esc(e.teacher_steps)} steps) · <a href="#/episode?path=${q(d.teacher.run)}">full replay with tool calls and SAM3 masks</a></figcaption></figure>` : ""}
    </div></div>
    <div id="demo-charts" class="panel">${charts.join("")}</div>
    <div class="panel"><div class="panel-head"><b>Record</b></div><p class="mono small">source: ${esc(e.source_run)}</p>${jsonView(e, { open: 1 })}</div>`;
  const video = document.getElementById("demo-video");
  if (video) video.addEventListener("timeupdate", () => moveMarkers(document.getElementById("demo-charts"), video.duration ? video.currentTime / video.duration : 0));
}
