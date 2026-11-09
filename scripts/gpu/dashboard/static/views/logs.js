// Log viewer: search, errors-and-warnings only, tail, auto-refresh for logs still being written.
import { api, esc, q, fileUrl, size } from "../lib/util.js";

export async function render(app, params, timers) {
  const path = params.get("path");
  app.innerHTML = `
    <div class="crumbs"><a href="javascript:history.back()">&larr; back</a> / ${esc(path)}</div>
    <div class="page-head"><h1>${esc(path.split("/").pop())}</h1><div class="toolbar"><a class="btn" href="${fileUrl(path)}?download=1">Download</a></div></div>
    <div class="toolbar"><input id="lq" placeholder="search (case-insensitive)" class="grow-input">
      <label><input type="checkbox" id="lalerts"> errors and warnings only</label>
      <select id="ltail"><option value="1000">last 1000 lines</option><option value="5000" selected>last 5000</option><option value="50000">last 50000</option></select>
      <label><input type="checkbox" id="lauto"> follow (refresh every 3 s)</label></div>
    <div id="lmeta" class="muted small"></div><div id="lbody" class="log"></div>`;
  const load = async () => {
    const qs = `path=${q(path)}&q=${q(document.getElementById("lq").value)}&alerts=${document.getElementById("lalerts").checked ? 1 : 0}&tail=${document.getElementById("ltail").value}`;
    const d = await api(`/api/log?${qs}`);
    document.getElementById("lmeta").textContent = `${d.matched} of ${d.total_lines} lines · ${size(d.size)}`;
    const body = document.getElementById("lbody");
    const atBottom = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
    body.innerHTML = d.lines.map((l) => `<div class="${l.alert ? "alert-line" : ""}"><span class="ln">${l.n}</span>${esc(l.text)}</div>`).join("") || `<div class="muted">No matching lines.</div>`;
    if (atBottom || document.getElementById("lauto").checked) body.scrollTop = body.scrollHeight;
  };
  let pending = null;
  document.getElementById("lq").oninput = () => { clearTimeout(pending); pending = setTimeout(load, 250); };
  ["lalerts", "ltail"].forEach((id) => (document.getElementById(id).onchange = load));
  const follow = () => {
    if (!location.hash.startsWith("#/log")) return;
    if (document.getElementById("lauto")?.checked) load();
    timers.push(setTimeout(follow, 3000));
  };
  await load();
  follow();
}
