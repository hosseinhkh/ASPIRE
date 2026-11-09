// How the pieces fit together, and where to extend them.
export function render(app) {
  app.innerHTML = `
    <div class="page-head"><h1>How it works</h1></div>
    <div class="panel"><p>Every robot, simulated or real, sits behind one interface. Everything above it does not depend on the robot.</p>
<pre class="code">Agent (Claude Code + Opus)   writes Code-as-Policy programs, reads these recordings, debugs, keeps skills
      │  batch runs (cap-harness run, session.sh) · live console (HTTP, this dashboard) · MCP (real robot)
ProgramExecutor (sandbox: only registered tools)
      │
ToolRegistry ─► CapApi   perception: segment_text, localize_object, mask_to_point_cloud, generate_grasps
      │                  motion: solve_ik, plan_motion, move_to_pose, move_synchronized, grippers, step
      ├─► EnvironmentAdapter   robosuite | libero | behavior | yam_real | your robot
      ├─► Model services       SAM3 :8114 · Contact-GraspNet :8115 · PyRoki :8116 · cuRobo :8118 · yours
      └─► RunRecorder          run.json, outcome.json, trace/ (every call), steps, videos, keyframes, overlays
                                 └─► this dashboard: replay, compare, logs, downloads</pre>
    <h3>Reading an episode</h3>
    <ul>
      <li><b>program</b> calls are what the generated program called; <b>api</b> calls are the stages inside the robot API (IK, planning, execution); <b>provider</b> calls are requests to the model services.</li>
      <li>Time on the timeline is simulated time; the camera videos run on the same clock, so clicking a call jumps every camera to that moment.</li>
      <li>"final trajectory waypoint did not converge" means the arm did not reach its goal. The error details give the joint error; settling continues while the arm is still getting closer.</li>
    </ul>
    <h3>Extending it</h3>
    <ul>
      <li><b>A new robot</b> (simulated or physical): implement an <span class="mono">EnvironmentAdapter</span>, a task registry and kinematics. The recordings, this dashboard, the live console and the agents work unchanged.</li>
      <li><b>A new model or tool</b>: an HTTP service plus a provider client, registered in <span class="mono">CapApi</span> and wrapped in <span class="mono">TracedProvider</span>. Its calls then appear here automatically.</li>
      <li><b>Comparing ideas</b>: record each variant as a session (<span class="mono">scripts/gpu/session.sh run</span>), then tick them under Runs and press Compare.</li>
    </ul>
    <p class="muted">Full write-up: <span class="mono">docs/big-picture.md</span> in the repository.</p></div>`;
}
