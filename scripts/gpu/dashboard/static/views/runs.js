// Runs: every session, live run, check, demo and campaign, with compare and downloads.
import { api, esc, q, when } from "../lib/util.js";
import { downloads, rate } from "../lib/ui.js";

export async function render(app) {
  const data = await api("/api/collections");
  const kinds = ["all", ...new Set(data.items.map((i) => i.kind))];
  app.innerHTML = `
    <div class="page-head"><h1>Runs</h1>
      <div class="toolbar">
        <input id="filter" placeholder="Filter by name or path" class="grow-input">
        <select id="kind">${kinds.map((k) => `<option>${k}</option>`).join("")}</select>
        <button class="primary" id="compare" disabled>Compare selected</button>
      </div></div>
    <p class="muted small">Sessions (<span class="mono">scripts/gpu/session.sh</span>), live-console runs, checks, demos and campaigns under <span class="mono">${esc(data.outputs)}</span>. Tick two or more to compare them task by task.</p>
    <div class="panel flush"><table class="table"><thead><tr><th></th><th>Run</th><th>Kind</th><th>Modified</th><th>Status</th><th>Task success</th><th>LLM cost</th><th>Commit</th><th>GPU</th><th>Download</th></tr></thead><tbody id="rows"></tbody></table></div>`;
  const selected = new Set();
  const updateCompare = () => {
    const button = document.getElementById("compare");
    button.disabled = selected.size < 1;
    button.textContent = selected.size ? `Compare selected (${selected.size})` : "Compare selected";
  };
  const draw = () => {
    const filter = document.getElementById("filter").value.toLowerCase();
    const kind = document.getElementById("kind").value;
    const rows = data.items.filter((i) => (kind === "all" || i.kind === kind) && (!filter || (i.name + i.path).toLowerCase().includes(filter)));
    document.getElementById("rows").innerHTML =
      rows
        .map((i) => {
          const target = i.kind === "live" ? `#/episode?path=${q(i.path)}` : `#/session?path=${q(i.path)}`;
          return `<tr class="click" data-href="${target}">
            <td class="nowrap" data-stop><input type="checkbox" data-path="${esc(i.path)}" ${selected.has(i.path) ? "checked" : ""}></td>
            <td><b>${esc(i.name)}</b><div class="muted small mono">${esc(i.path)}</div></td>
            <td><span class="chip">${esc(i.kind)}</span></td>
            <td class="small nowrap">${when(i.modified)}</td>
            <td class="small">${esc(i.status || "")}${i.stage_counts ? `<div class="muted">${esc(i.stage_counts)}</div>` : ""}</td>
            <td>${rate(i.successes, i.episodes)}</td>
            <td>${i.llm_cost_usd ? "$" + esc(i.llm_cost_usd) : ""}</td>
            <td class="mono small">${esc(i.git_commit || "")}</td>
            <td>${esc(i.gpu ?? "")}</td>
            <td data-stop class="nowrap">${downloads(i.path, "Zip")}</td></tr>`;
        })
        .join("") || `<tr><td colspan="10" class="muted">No runs yet. Start one: <span class="mono">scripts/gpu/session.sh run --tasks quick</span></td></tr>`;
    document.querySelectorAll("#rows tr.click").forEach((tr) => {
      tr.onclick = (event) => {
        if (event.target.closest("[data-stop]")) return;
        location.hash = tr.dataset.href;
      };
    });
    document.querySelectorAll("#rows input[type=checkbox]").forEach((box) => {
      box.onchange = () => {
        box.checked ? selected.add(box.dataset.path) : selected.delete(box.dataset.path);
        updateCompare();
      };
    });
  };
  document.getElementById("filter").oninput = draw;
  document.getElementById("kind").onchange = draw;
  document.getElementById("compare").onclick = () => {
    location.hash = `#/compare?${[...selected].map((p) => `path=${q(p)}`).join("&")}`;
  };
  draw();
}
