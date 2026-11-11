// Router and shell. Each page lives in views/<name>.js and exports render(app, params, ctx).
import { api, esc, params } from "./lib/util.js";
import * as runs from "./views/runs.js";
import * as session from "./views/session.js";
import * as episode from "./views/episode.js";
import * as agent from "./views/agent.js";
import * as compare from "./views/compare.js";
import * as logs from "./views/logs.js";
import * as live from "./views/live.js";
import * as about from "./views/about.js";
import * as vla from "./views/vla.js";

const app = document.getElementById("app");
const ctx = { outputs: "", timers: [] };

const routes = [
  [/^\/?$/, () => runs.render(app, params(), ctx)],
  [/^\/session$/, () => session.render(app, params(), ctx)],
  [/^\/episode$/, () => episode.render(app, params(), ctx)],
  [/^\/agent$/, () => agent.render(app, params(), ctx)],
  [/^\/compare$/, () => compare.render(app, params(), ctx)],
  [/^\/log$/, () => logs.render(app, params(), ctx.timers)],
  [/^\/live$/, () => live.renderList(app, params(), ctx)],
  [/^\/live\/(.+)$/, (m) => live.renderConsole(app, m[1], ctx)],
  [/^\/about$/, () => about.render(app)],
  [/^\/vla$/, () => vla.renderList(app, params(), ctx)],
  [/^\/vla\/run$/, () => vla.renderRun(app, params(), ctx)],
  [/^\/vla\/episode$/, () => vla.renderEpisode(app, params(), ctx)],
  // Links from older versions of the dashboard.
  [/^\/c$/, () => session.render(app, params(), ctx)],
  [/^\/e$/, () => episode.render(app, params(), ctx)],
  [/^\/a$/, () => agent.render(app, params(), ctx)],
];

async function route() {
  ctx.timers.splice(0).forEach(clearTimeout);
  document.querySelectorAll("video").forEach((v) => v.pause());
  const path = (location.hash.slice(1).split("?")[0] || "/").replace(/\/$/, "") || "/";
  const others = ["/compare", "/live", "/about", "/vla"];
  document.querySelectorAll("header nav a").forEach((a) => {
    const section = a.dataset.section;
    const on = section === "/" ? !others.some((p) => path.startsWith(p)) : path.startsWith(section);
    a.classList.toggle("on", on);
  });
  app.innerHTML = `<p class="muted">Loading...</p>`;
  try {
    if (!ctx.outputs) {
      try {
        ctx.outputs = (await api("/api/collections")).outputs;
        document.getElementById("where").textContent = ctx.outputs;
      } catch {}
    }
    for (const [pattern, handler] of routes) {
      const match = path.match(pattern);
      if (match) {
        await handler(match);
        return;
      }
    }
    app.innerHTML = `<p>Page not found. <a href="#/">Back to runs</a></p>`;
  } catch (error) {
    app.innerHTML = `<div class="alert bad">${esc(error.message)}</div><p><a href="#/">Back to runs</a></p>`;
  }
}

document.getElementById("lightbox").onclick = () => document.getElementById("lightbox").classList.remove("on");
document.addEventListener("keydown", (e) => e.key === "Escape" && document.getElementById("lightbox").classList.remove("on"));
window.addEventListener("hashchange", () => {
  route();
  window.scrollTo(0, 0);
});
route();
