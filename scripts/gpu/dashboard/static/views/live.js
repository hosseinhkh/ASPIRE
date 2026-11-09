// Live robots: start a simulator console, see cameras and state, call tools, run programs.
import { api, esc, q, num, fileUrl, pretty } from "../lib/util.js";
import { badge, jsonView } from "../lib/ui.js";

const EXAMPLES = [
  ["robosuite", "cube_lifting", 0], ["robosuite", "two_arm_lift", 0], ["robosuite", "two_arm_handover", 0],
  ["libero-pro", "libero_object_swap", 0], ["libero-pro", "libero_90", 33], ["libero-pro", "libero_goal_swap", 0],
  ["libero-pro", "libero_10_swap", 3],
];

export async function renderList(app) {
  const d = await api("/api/live");
  app.innerHTML = `
    <div class="page-head"><h1>Live robots</h1></div>
    <p class="muted">A live console keeps one robot running (a simulator here; a real robot uses the same API). You can call any of its tools (cameras, state, SAM3, Contact-GraspNet, PyRoki, cuRobo, motions) or run Code-as-Policy programs, one at a time. Everything is recorded like an episode and appears under Runs when you close it. The model services must be running (<span class="mono">scripts/gpu/services.sh start</span>).</p>
    <div class="panel"><h3>Start a simulator</h3><div class="toolbar">
      <label>benchmark <select id="b"><option>robosuite</option><option>libero-pro</option></select></label>
      <label>suite <input id="s" list="suites" value="cube_lifting" style="width:170px"></label><datalist id="suites">${[...new Set(EXAMPLES.map((e) => e[1]))].map((s) => `<option>${s}</option>`).join("")}</datalist>
      <label>task <input id="t" type="number" value="0" style="width:64px"></label>
      <label>seed <input id="seed" type="number" value="1" style="width:64px"></label>
      <label>camera <input id="w" type="number" value="512" style="width:64px">×<input id="h" type="number" value="512" style="width:64px"></label>
      <label>LIBERO init <select id="im"><option>saved</option><option>seeded</option></select></label>
      <button class="primary" id="start">Start</button></div>
      <div class="small muted">Examples: ${EXAMPLES.map((e) => `<a href="javascript:void 0" data-ex='${JSON.stringify(e)}'>${e[0]} ${e[1]}:${e[2]}</a>`).join(" · ")}</div>
      <div id="start-msg" class="small"></div></div>
    <h2>Consoles</h2><div class="panel flush"><table class="table"><thead><tr><th>Console</th><th>State</th><th>Ready</th><th>Log</th><th></th></tr></thead><tbody>
      ${d.consoles.map((c) => `<tr><td><a href="#/live/${esc(c.id)}">${esc(c.label)}</a><div class="muted small">${esc(c.url)}</div></td><td>${esc(c.state)}</td><td>${badge(c.ready)}</td><td>${c.log ? `<a href="#/log?path=${q(c.log)}">console log</a>` : ""}</td><td>${c.external ? "" : `<button data-stop="${esc(c.id)}">Close & save</button>`}</td></tr>`).join("") || `<tr><td colspan="5" class="muted">None running.</td></tr>`}
    </tbody></table></div>
    <p class="muted small">From a terminal instead: <span class="mono">.venv-robosuite/bin/python -m cap_harness.live --benchmark robosuite --suite cube_lifting</span> prints its address; start the dashboard with <span class="mono">--live &lt;that address&gt;</span>.</p>`;
  document.querySelectorAll("[data-ex]").forEach((a) => (a.onclick = () => {
    const [b, s, t] = JSON.parse(a.dataset.ex);
    document.getElementById("b").value = b;
    document.getElementById("s").value = s;
    document.getElementById("t").value = t;
  }));
  document.querySelectorAll("[data-stop]").forEach((b) => (b.onclick = async () => {
    b.disabled = true;
    b.textContent = "closing...";
    await api(`/api/live/${b.dataset.stop}/stop`, { method: "POST" });
    renderList(app);
  }));
  document.getElementById("start").onclick = async () => {
    const msg = document.getElementById("start-msg");
    msg.textContent = "Starting (the simulator takes 10–60 s to load)...";
    const v = (k) => document.getElementById(k).value;
    try {
      const c = await api("/api/live/start", { method: "POST", body: JSON.stringify({ benchmark: v("b"), suite: v("s"), task_id: +v("t"), seed: +v("seed"), camera_width: +v("w"), camera_height: +v("h"), init_mode: v("im") }) });
      location.hash = `#/live/${c.id}`;
    } catch (e) {
      msg.innerHTML = `<span class="bad">${esc(e.message)}</span>`;
    }
  };
}

export async function renderConsole(app, id, ctx) {
  const base = `/live/${id}`;
  app.innerHTML = `<div class="crumbs"><a href="#/live">Live robots</a> / ${esc(id)}</div><h1 id="ctitle">Connecting...</h1><div id="cbody" class="muted">Waiting for the console to load the robot (the first start can take a minute).</div>`;
  let info = null;
  for (let i = 0; i < 180 && !info; i++) {
    try {
      info = await api(`${base}/api/info`);
    } catch {
      await new Promise((r) => setTimeout(r, 1000));
    }
    if (!location.hash.startsWith(`#/live/${id}`)) return;
  }
  if (!info) {
    document.getElementById("cbody").innerHTML = `<span class="bad">The console did not come up; check its log on the Live robots page.</span>`;
    return;
  }
  const tools = info.tools.filter((t) => t.json_callable);
  const outputs = ctx.outputs || "";
  const relRun = outputs && info.run_dir.startsWith(outputs) ? info.run_dir.slice(outputs.length + 1) : null;
  document.getElementById("ctitle").innerHTML = `${esc(info.benchmark)} ${esc(info.suite)}:${info.task_id} <span class="muted">seed ${info.seed}</span> <span class="chip">${esc(info.embodiment)}</span>`;
  const body = document.getElementById("cbody");
  body.className = "";
  body.innerHTML = `
    ${info.language ? `<div class="instruction">“${esc(info.language)}”</div>` : ""}
    <div class="split wide-left"><div class="panel"><div class="cams">${info.cameras.map((c) => `<figure><img data-cam="${esc(c)}" alt="${esc(c)}"><figcaption>${esc(c)}</figcaption></figure>`).join("")}</div></div>
      <div class="panel"><div class="panel-head"><b>Robot</b><span id="phase" class="small"></span></div><div id="rstate"></div>
        <div class="toolbar"><button id="close">Close & save recording</button>${relRun ? `<a class="btn" href="#/episode?path=${q(relRun)}">Recording so far</a>` : ""}</div></div></div>
    <div class="split"><div class="panel"><h3>Call a tool</h3>
        <select id="tool">${tools.map((t) => `<option value="${esc(t.name)}">${esc(t.name)}</option>`).join("")}</select>
        <div class="small mono muted" id="sig"></div><div class="small" id="doc"></div>
        <textarea id="args" rows="9" class="code"></textarea><div class="toolbar"><button class="primary" id="run-tool">Run tool</button><button id="fill">Fill from current state</button></div></div>
      <div class="panel"><h3>Run a Code-as-Policy program</h3><textarea id="code" rows="12" class="code">state = get_robot_state()
masks = segment_text("${esc(info.default_camera)}", "cube")
print("masks:", len(masks.segmentations) if masks.ok else masks.error)
result = {"ee": [float(v) for v in state.end_effector_poses["primary"].position]}</textarea>
        <div class="toolbar"><input id="label" placeholder="label"><button class="primary" id="run-prog">Run program</button></div></div></div>
    <div class="split"><div class="panel"><h3>Current job</h3><div id="job" class="small muted">idle</div></div>
      <div class="panel"><h3>Events</h3><div id="events" class="small mono scroll"></div></div></div>
    <h2>History</h2><div id="history"></div>`;
  let state = null, after = 0, lastJob = null;
  const example = (t) => {
    const props = t.schema.properties || {}, req = new Set(t.schema.required || []), out = {};
    for (const [k, p] of Object.entries(props)) {
      if ("default" in p) { out[k] = p.default; continue; }
      if (!req.has(k)) continue;
      const ee = state?.robot_state?.end_effector_poses?.primary;
      if (k.includes("camera")) out[k] = info.default_camera;
      else if (k === "text" || k === "query") out[k] = "cube";
      else if ((p.properties && p.properties.position) || k.endsWith("pose") || k === "goal") out[k] = ee ? { position: ee.position, quaternion_wxyz: ee.quaternion_wxyz, frame: ee.frame } : { position: [0.5, 0, 0.3], rpy_deg: [180, 0, 0], frame: "robot0_base" };
      else if (k === "target" || k.includes("joints")) out[k] = state?.robot_state?.joint_positions?.primary || [];
      else if (k === "targets") out[k] = Object.fromEntries(Object.entries(state?.robot_state?.end_effector_poses || {}).map(([a, e]) => [a, { position: e.position, quaternion_wxyz: e.quaternion_wxyz, frame: e.frame }]));
      else if (p.type === "number" || p.type === "integer") out[k] = 0;
      else if (p.type === "string") out[k] = "";
      else out[k] = null;
    }
    return out;
  };
  const pick = () => {
    const t = tools.find((x) => x.name === document.getElementById("tool").value);
    document.getElementById("sig").textContent = t.signature;
    document.getElementById("doc").textContent = t.doc || "";
    document.getElementById("args").value = JSON.stringify(example(t), null, 2);
  };
  document.getElementById("tool").onchange = pick;
  document.getElementById("fill").onclick = pick;
  const submit = async (url, payload) => {
    try {
      const r = await api(base + url, { method: "POST", body: JSON.stringify(payload) });
      lastJob = r.job_id;
      poll();
    } catch (e) {
      document.getElementById("job").innerHTML = `<span class="bad">${esc(e.message)}</span>`;
    }
  };
  document.getElementById("run-tool").onclick = () => {
    let args;
    try { args = JSON.parse(document.getElementById("args").value || "{}"); } catch (e) {
      document.getElementById("job").innerHTML = `<span class="bad">arguments are not valid JSON: ${esc(e.message)}</span>`;
      return;
    }
    submit("/api/tool", { name: document.getElementById("tool").value, arguments: args });
  };
  document.getElementById("run-prog").onclick = () => submit("/api/program", { code: document.getElementById("code").value, label: document.getElementById("label").value || "console program" });
  document.getElementById("close").onclick = async () => {
    const button = document.getElementById("close");
    button.disabled = true;
    button.textContent = "saving...";
    try { await api(`/api/live/${id}/stop`, { method: "POST" }); } catch {}
    location.hash = relRun ? `#/episode?path=${q(relRun)}` : "#/live";
  };
  const refreshCams = () => document.querySelectorAll("[data-cam]").forEach((img) => {
    const next = new Image();
    next.onload = () => (img.src = next.src);
    next.src = `${base}/api/camera/${q(img.dataset.cam)}.jpg?t=${Date.now()}`;
  });
  const poll = async () => {
    if (!location.hash.startsWith(`#/live/${id}`)) return;
    try {
      state = await api(`${base}/api/state?after=${after}`);
      const rs = state.robot_state || {};
      document.getElementById("phase").innerHTML = `<span class="chip">${esc(state.phase)}</span> ${state.steps} steps`;
      document.getElementById("rstate").innerHTML = jsonView({ end_effector_poses: rs.end_effector_poses, gripper_positions: rs.gripper_positions, joint_positions: rs.joint_positions }, { open: 2 });
      const ev = document.getElementById("events");
      state.events.forEach((e) => {
        after = Math.max(after, e.seq);
        ev.insertAdjacentHTML("afterbegin", `<div>${new Date(e.timestamp_s * 1000).toLocaleTimeString()} [${esc(e.role)}] ${esc(e.text)}</div>`);
      });
      const job = state.job;
      document.getElementById("job").innerHTML = job ? `<b>${esc(job.label)}</b> ${job.active ? `<span class="badge warn">running</span>` : job.error ? `<span class="badge bad">error</span>` : `<span class="badge ok">done</span>`} ${num(job.elapsed_s, 2)} s${job.error ? `<div class="alert bad">${esc(job.error)}</div>` : ""}${!job.active && job.result ? `<details ${job.id === lastJob ? "open" : ""}><summary>result</summary>${jsonView(job.result)}</details>` : ""}` : "idle";
      if (!job || !job.active) {
        const h = await api(`${base}/api/history`);
        document.getElementById("history").innerHTML = h.history.slice().reverse().map((c, i) => `<div class="msg ${c.error ? "err" : "tool_use"}"><b>${h.history.length - i}. ${esc(c.kind)} ${esc(c.name)}</b> ${badge(c.status === "done", { yes: "done", no: c.status })} <span class="muted small">${num(c.elapsed_s, 2)} s</span>
          <details><summary>request and result</summary><pre class="code">${esc(c.code || pretty(c.arguments))}</pre>${c.error ? `<div class="alert bad">${esc(c.error)}</div>` : ""}${jsonView(c.result)}</details></div>`).join("");
      }
      refreshCams();
    } catch (e) {
      document.getElementById("phase").innerHTML = `<span class="bad">disconnected: ${esc(e.message)}</span>`;
    }
    ctx.timers.push(setTimeout(poll, state && state.job && state.job.active ? 400 : 1200));
  };
  await poll();
  pick();
}
