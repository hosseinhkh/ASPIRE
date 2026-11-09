// Benchmark comparison: runs side by side (code versions, GPUs, simulators, real robots).
import { api, esc, q, num, pct } from "../lib/util.js";
import { rate } from "../lib/ui.js";

export async function render(app, params) {
  const paths = params.getAll("path");
  const all = await api("/api/collections");
  const picker = `<div class="toolbar"><select id="add"><option value="">Add a run to compare...</option>${all.items.filter((i) => !paths.includes(i.path)).map((i) => `<option value="${esc(i.path)}">${esc(i.name)} — ${esc(i.path)}</option>`).join("")}</select></div>`;
  if (!paths.length) {
    app.innerHTML = `<div class="page-head"><h1>Compare runs</h1></div><p class="muted">Pick runs to put side by side: task success per task and phase, model-service latency, simulator speed and LLM cost. Use it to compare code versions, GPUs, simulators or a real robot.</p>${picker}`;
    bindPicker(paths);
    return;
  }
  const b = await api(`/api/benchmark?${paths.map((p) => `path=${q(p)}`).join("&")}`);
  const cols = b.columns;
  const key = (t) => [t.agent || "", t.phase, t.benchmark, t.suite, t.task_id].join("|");
  const cell = (col, row) => col.tasks.find((t) => key(t) === key(row));
  const two = cols.length === 2;
  const providers = [...new Set(cols.flatMap((c) => Object.keys(c.providers)))].sort();
  const sims = [...new Set(cols.flatMap((c) => Object.keys(c.simulator_ms)))].sort();
  app.innerHTML = `
    <div class="page-head"><h1>Compare runs</h1>${picker}</div>
    <div class="panel flush"><table class="table compare"><thead><tr><th></th>${cols.map((c, i) => `<th><a href="#/session?path=${q(c.path)}">${esc(c.name)}</a> <button class="small" data-remove="${i}" title="remove">✕</button><div class="muted small mono">${esc(c.path)}</div></th>`).join("")}${two ? "<th>Change</th>" : ""}</tr></thead>
    <tbody>
      <tr><th>Code</th>${cols.map((c) => `<td class="mono">${esc(c.git_commit || "-")}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
      <tr><th>GPU</th>${cols.map((c) => `<td>${esc(c.gpu ?? "-")}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
      <tr><th>Agent</th>${cols.map((c) => `<td class="small">${agentModels(c)}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
      <tr><th>Agent effort</th>${cols.map((c) => `<td>${esc(c.claude_effort ?? "-")}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
      <tr><th>Motion timing</th>${cols.map((c) => `<td>${c.motion_retiming == null ? "-" : c.motion_retiming === "1" ? "re-timed" : "published"}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
      <tr><th>All episodes</th>${cols.map((c) => `<td>${rate(c.successes, c.episodes)}</td>`).join("")}${two ? `<td>${delta(cols[0].successes, cols[0].episodes, cols[1].successes, cols[1].episodes)}</td>` : ""}</tr>
      <tr><th>LLM cost</th>${cols.map((c) => `<td>${c.llm_cost_usd != null ? "$" + num(c.llm_cost_usd, 2) : "-"}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
      <tr><th>Motion speed</th>${cols.map((c) => `<td>${c.realtime_factor ? num(c.realtime_factor, 2) + " s per simulated s" : "-"}</td>`).join("")}${two ? "<td></td>" : ""}</tr>
    </tbody></table></div>
    <h2>Task success</h2>
    <div class="panel flush"><table class="table compare"><thead><tr><th>Task</th>${cols.map((c) => `<th>${esc(c.name)}</th>`).join("")}${two ? "<th>Change</th>" : ""}</tr></thead><tbody>
      ${b.rows.map((r) => {
        const cells = cols.map((c) => cell(c, r));
        return `<tr><td>${esc(r.benchmark)} <b>${esc(r.suite)}:${esc(r.task_id)}</b> ${r.agent ? `<span class="chip">${esc(r.agent)}</span>` : ""}<span class="chip">${esc(r.phase)}</span><div class="muted small">${esc(r.task_name || "")}</div></td>
          ${cells.map((t) => `<td>${t ? `${rate(t.successes, t.episodes)}<div class="muted small">${num(t.mean_steps, 0)} steps · ${num(t.mean_wall_s)} s</div>` : `<span class="muted">-</span>`}</td>`).join("")}
          ${two ? `<td>${cells[0] && cells[1] ? delta(cells[0].successes, cells[0].episodes, cells[1].successes, cells[1].episodes) : ""}</td>` : ""}</tr>`;
      }).join("") || `<tr><td colspan="${cols.length + 2}" class="muted">No episodes.</td></tr>`}
    </tbody></table></div>
    <h2>Model services: median seconds per call (p90)</h2>
    <div class="panel flush"><table class="table compare"><thead><tr><th>Service call</th>${cols.map((c) => `<th>${esc(c.name)}</th>`).join("")}</tr></thead><tbody>
      ${providers.map((p) => `<tr><td class="mono small">${esc(p.replace("provider: ", ""))}</td>${cols.map((c) => { const v = c.providers[p]; return `<td>${v ? `<b>${num(v.median_s, 3)}</b> <span class="muted small">(${num(v.p90_s, 3)}) · ${v.count} calls</span>` : "-"}</td>`; }).join("")}</tr>`).join("") || `<tr><td class="muted">No service calls.</td></tr>`}
    </tbody></table></div>
    <h2>Simulator: median milliseconds per call</h2>
    <div class="panel flush"><table class="table compare"><thead><tr><th></th>${cols.map((c) => `<th>${esc(c.name)}</th>`).join("")}</tr></thead><tbody>
      ${sims.map((sname) => `<tr><td class="mono small">${esc(sname)}</td>${cols.map((c) => `<td>${c.simulator_ms[sname] != null ? num(c.simulator_ms[sname], 1) : "-"}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`;
  bindPicker(paths);
  document.querySelectorAll("[data-remove]").forEach((btn) => (btn.onclick = () => {
    const rest = paths.filter((_, i) => i !== +btn.dataset.remove);
    location.hash = `#/compare?${rest.map((p) => `path=${q(p)}`).join("&")}`;
  }));
}

function bindPicker(paths) {
  const add = document.getElementById("add");
  if (add) add.onchange = () => add.value && (location.hash = `#/compare?${[...paths, add.value].map((p) => `path=${q(p)}`).join("&")}`);
}

function delta(k1, n1, k2, n2) {
  if (!n1 || !n2) return "";
  const d = pct(k2, n2) - pct(k1, n1);
  return `<span class="${d > 0 ? "ok" : d < 0 ? "bad" : "muted"}">${d > 0 ? "+" : ""}${d} pts</span>`;
}

// The model(s) the agents ran, with the context window and thinking Claude Code reported.
function agentModels(c) {
  const rows = Object.entries(c.agent_models || {}).map(([model, u]) => {
    const window = u.context_window ? (u.context_window >= 1e6 ? `${u.context_window / 1e6}M` : `${Math.round(u.context_window / 1000)}k`) : "?";
    return `${esc(model)}<div class="muted">${window} window · ${num(u.thinking_tokens || 0, 0)} thinking tokens</div>`;
  });
  return rows.join("") || "-";
}
