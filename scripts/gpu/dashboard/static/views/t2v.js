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
let expTab = "Compare";
let teacherSet = null;
let demoSet = null;

const percent = (v) => (v === null || v === undefined ? "-" : `${(100 * v).toFixed(1)}%`);
const minutes = (s) => (s === null || s === undefined ? "-" : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const zip = (path, media, label) => `<a class="btn" href="/download?path=${q(path)}&media=${media ? 1 : 0}">${esc(label)}</a>`;
const statusBadge = (s) => `<span class="badge ${{ running: "ok", finished: "ok", interrupted: "warn", stopped: "bad", failed: "bad" }[s] || "muted"}">${esc(s || "-")}${s === "running" ? " ●" : ""}</span>`;
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
    <div class="panel flush"><table class="table"><thead><tr><th>Experiment</th><th>Suite</th><th>Teacher (official scenes)</th><th>Demos</th><th>Datasets</th><th>Training</th><th>Student evaluations</th><th>Last event</th></tr></thead><tbody>
    ${
      data.items
        .map(
          (e) => `<tr class="click" data-href="#/t2v/exp?path=${q(e.path)}">
          <td><b>${esc(e.name)}</b><div class="muted small">tasks ${esc((e.tasks || []).join(", "))}</div></td>
          <td class="mono small">${esc(e.suite || "")}</td>
          <td>${e.teacher_official ? rate(e.teacher_official.successes, e.teacher_official.episodes) : '<span class="muted">not run</span>'}</td>
          <td>${esc(e.demos)}</td><td>${esc(e.datasets)}</td>
          <td class="small">${e.training.map((t) => `${esc(t.name)} <span class="muted">${esc(t.student_label || "")}</span> ${statusBadge(t.status)}`).join("<br>")}</td>
          <td class="small">${e.evals.map((v) => `${esc(v.name)} <span class="muted">${esc(v.student_label || "")}</span>: <b>${percent(v.mean)}</b> ${v.status === "finished" ? "" : statusBadge(v.status)}`).join("<br>")}</td>
          <td class="small">${e.last_event ? `${esc(e.last_event.event)}<div class="muted">${esc(e.last_event.time)}</div>` : ""}</td></tr>`,
        )
        .join("") || `<tr><td colspan="8" class="muted">No experiment yet: <span class="mono">scripts/gpu/t2v.sh init preflight --suite libero_object_swap --tasks 0,1</span></td></tr>`
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
  const running = d.training.some((t) => t.progress.status === "running") || d.evals.some((e) => e.progress.status === "running");
  teacherSet = teacherSet && d.teacher[teacherSet] ? teacherSet : Object.keys(d.teacher)[0] || null;
  demoSet = demoSet && d.demos.some((s) => s.name === demoSet) ? demoSet : (d.demos[0] || {}).name || null;

  app.innerHTML = `
    <div class="crumbs"><a href="#/t2v">Trace2VLA</a> / ${esc(d.name)}</div>
    <div class="page-head"><h1>${esc(d.name)} <span class="muted">${esc(suite)}</span></h1>
      <div class="toolbar"><span class="btn-group">${zip(d.path, false, "Download logs")}${zip(d.path, true, "Download all")}</span></div></div>
    <div class="cards">
      ${card("Teacher on official scenes", official ? `<b>${percent(official.mean)}</b>` : "not run", official ? `${official.successes}/${official.episodes} episodes` : "t2v.sh teacher-run --scenes official")}
      ${progressCards}
      ${card("Demonstrations", `${demos}`, d.demos.map((s) => `${esc(s.name)}: ${s.accepted} kept, ${s.rejected} rejected`).join("<br>"))}
      ${card("Datasets", `${d.datasets.length}`, d.datasets.map((s) => `${esc(s.name)} <span class="muted">${esc(s.student_label || "")}</span>: ${s.episodes} episodes, ${s.frames} frames`).join("<br>"))}
      ${card("Training", `${d.training.length}`, d.training.map((t) => `${esc(t.name)} <span class="muted">${esc(t.student_label || "")}</span> ${statusBadge(t.progress.status)} ${t.progress.step ?? 0}/${t.progress.steps ?? "?"}`).join("<br>"))}
    </div>
    ${tabs("t2v-exp", ["Compare", "Teacher", "Demos", "Training", "Evaluations", "Datasets", "Ledger"], expTab)}`;

  const taskName = (t) => `<b>${esc(suite)}:${esc(t)}</b><div class="muted small">${esc((d.settings.instructions || {})[String(t)] || "")}</div>`;
  const cell = (t) => (t ? rate(t.successes, t.episodes) : '<span class="muted">-</span>');

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
            return `<div class="panel"><div class="panel-head"><b>${esc(s.name)}</b> ${studentBadge(s.student_label)}${s.derived_from ? ` <span class="muted small">frames of ${esc(s.derived_from)}</span>` : ""} <span class="muted small">${esc(s.episodes)} episodes · ${esc(s.frames)} frames · per task ${esc(JSON.stringify(s.per_task))} · ${esc(s.rejected)} rejected at build · ${esc(s.created || "")}</span></div>
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

  const bodies = { Compare: compare, Teacher: teacher, Demos: demosView, Training: training, Evaluations: evals, Datasets: datasets, Ledger: ledger };
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
