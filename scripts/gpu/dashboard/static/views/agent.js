// One Claude Code conversation: prompt, report, programs, per-version results, every turn.
import { api, esc, q, num, pretty } from "../lib/util.js";
import { card, jsonView, rate, bindZoom } from "../lib/ui.js";

export async function render(app, params) {
  const d = await api(`/api/agent?path=${q(params.get("path"))}`);
  const image = (ref) => `/api/agent-image?path=${q(d.path)}&ref=${ref}`;
  const session = d.path.split("/").slice(0, 2).join("/");
  app.innerHTML = `
    <div class="crumbs"><a href="#/">Runs</a> / <a href="#/session?path=${q(session)}">${esc(session)}</a> / agent</div>
    <div class="page-head"><h1>Agent conversation</h1><span class="muted small mono">${esc(d.path)}</span></div>
    <div class="cards">
      ${card("Result", esc(d.result || "-"))}${card("Turns", d.turns ?? "-")}${card("Tool calls", `${d.tool_calls}`, d.tool_errors ? `<span class="bad">${d.tool_errors} errors</span>` : "")}
      ${card("Minutes", d.minutes ?? "-")}${card("Cost", d.cost_usd != null ? "$" + num(d.cost_usd, 2) : "-")}
    </div>
    ${d.versions.length ? `<h2>Program versions</h2><div class="panel"><table class="table compact"><thead><tr><th>Phase</th><th>Task success</th></tr></thead><tbody>${d.versions.map((v) => `<tr><td class="mono">${esc(v.phase)}</td><td>${rate(v.passes || 0, v.seeds || 0)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    ${Object.entries(d.texts).map(([k, v]) => `<details class="panel" ${/REPORT|SUMMARY/.test(k) ? "open" : ""}><summary><b>${esc(k)}</b></summary><pre class="code">${esc(v)}</pre></details>`).join("")}
    ${Object.keys(d.programs).length ? `<details class="panel"><summary><b>Programs written (${Object.keys(d.programs).length})</b></summary>${Object.entries(d.programs).map(([k, v]) => `<h3>${esc(k)}</h3><pre class="code">${esc(v)}</pre>`).join("")}</details>` : ""}
    <h2>Conversation</h2>
    <div class="toolbar"><input id="find" placeholder="search the conversation" class="grow-input">
      <label><input type="checkbox" id="hide-think"> hide thinking</label><label><input type="checkbox" id="only-err"> tool errors only</label><label><input type="checkbox" id="only-tools"> tool calls only</label></div>
    <div id="events"></div>`;
  const view = (e) => {
    if (e.kind === "init") return `<div class="msg init"><b>session start</b> · ${esc(e.model)} <div class="muted small">${esc((e.tools || []).join(", "))}</div></div>`;
    if (e.kind === "text") return `<div class="msg text"><b>${esc(e.role)}</b><pre class="prose">${esc(e.text)}</pre></div>`;
    if (e.kind === "thinking") return `<details class="msg thinking"><summary><b>thinking</b> <span class="muted small">${esc(e.text.slice(0, 160))}...</span></summary><pre class="prose">${esc(e.text)}</pre></details>`;
    if (e.kind === "tool_use") return `<div class="msg tool_use"><b>${esc(e.name)}</b>${e.input && e.input.command ? `<pre class="code">${esc(e.input.command)}</pre>` : jsonView(e.input, { open: 1 })}</div>`;
    if (e.kind === "tool_result") return `<details class="msg ${e.is_error ? "err" : "tool_result"}" ${e.is_error ? "open" : ""}><summary><b>${esc(e.name)} result</b> ${e.is_error ? '<span class="badge bad">error</span>' : ""}</summary>${e.parts.map((p) => (p.type === "image" ? `<img data-zoom class="inline" src="${image(p.ref)}" alt="image shown to the agent">` : `<pre class="code">${esc(pretty(p.text, 20000))}</pre>`)).join("")}</details>`;
    if (e.kind === "result") return `<div class="msg result"><b>finished: ${esc(e.subtype)}</b> · ${e.turns} turns · ${e.minutes} min · $${num(e.cost_usd, 2)}${e.text ? `<pre class="prose">${esc(e.text)}</pre>` : ""}</div>`;
    return "";
  };
  const text = (e) => [e.text, e.name, e.input && JSON.stringify(e.input), ...(e.parts || []).map((p) => p.text)].filter(Boolean).join(" ").toLowerCase();
  const draw = () => {
    const find = document.getElementById("find").value.toLowerCase();
    const hide = document.getElementById("hide-think").checked;
    const errors = document.getElementById("only-err").checked;
    const tools = document.getElementById("only-tools").checked;
    const shown = d.events.filter((e) => (!hide || e.kind !== "thinking") && (!errors || (e.kind === "tool_result" && e.is_error)) && (!tools || e.kind === "tool_use" || e.kind === "tool_result") && (!find || text(e).includes(find)));
    document.getElementById("events").innerHTML = shown.map(view).join("") || `<p class="muted">Nothing matches.</p>`;
    bindZoom(document.getElementById("events"));
  };
  ["hide-think", "only-err", "only-tools"].forEach((id) => (document.getElementById(id).onchange = draw));
  document.getElementById("find").oninput = draw;
  draw();
}
