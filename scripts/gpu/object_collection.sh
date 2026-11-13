#!/usr/bin/env bash
# The libero_object_swap data collection, one command per step. Each step prints every command
# before running it, stops at the first error, and keeps its log in the experiment's logs/.
#
#   scripts/gpu/object_collection.sh agent-test TASK
#                                               one coding agent writes task TASK's program against
#                                               all five layouts (in this terminal, ~45-75 min, one
#                                               agent's LLM cost), then the program records 3 new
#                                               scenes of each layout (experiment object-agent-test)
#                                               and a verdict says whether it is good enough
#   scripts/gpu/object_collection.sh agent-fix TASK [PROGRAM]
#                                               the same, but the agent repairs an existing program
#                                               (default: TASK's program from the last agent-test or
#                                               agent-fix): it runs it first, then fixes only what
#                                               fails, keeping every layout it solves
#   scripts/gpu/object_collection.sh pilot      pipeline test (~20-30 min): tasks 0-1 x 5 layouts x
#                                               3 scenes with the preflight's programs (experiment
#                                               object-pilot), a dataset, a verdict
#   scripts/gpu/object_collection.sh setup      the experiment object-full: 10 tasks, 25 scenes on
#                                               each of the 5 layouts, then t2v.sh check
#   scripts/gpu/object_collection.sh start      in the background: the coding agents for the tasks
#                                               without a program (one session; each agent develops
#                                               against all five layouts), and t2v.sh collect, which
#                                               takes each program as its agent finishes
#   scripts/gpu/object_collection.sh status     where it stands (also: the dashboard's Pipeline tab)
#   scripts/gpu/object_collection.sh stop       stop both; finished work is kept, start continues
#   scripts/gpu/object_collection.sh datasets   after the collection: main-25 (alt1-3), control-25
#                                               (the official partner's layout), their pi0.5 copies
#   scripts/gpu/object_collection.sh evaluate CHECKPOINT [STUDENT]
#                                               in the background: CHECKPOINT (a training run, RUN@STEP,
#                                               or base with STUDENT smolvla|pi05) on every layout:
#                                               the official swap scenes, the standard layout, and the
#                                               dev scenes of the official and each sibling layout
#
# Settings (environment variables): EXP=object-full TASKS=0-9 PILOT=object-pilot SCENES=25
# WORKERS=2 PREPARE_WORKERS=6 DATASET_WORKERS=4 PROGRAMS (folders of task_NN.py, first one
# wins per task, for setup; default: object-agent-test's tested programs; tasks with none get an
# agent) PILOT_PROGRAMS (the same for pilot; default object-agent-test's, then preflight2's)
# AGENT_TEST=object-agent-test AGENT_LAYOUTS=all ("" = the official layout only)
# LLM_BUDGET=25 LLM_TOTAL_BUDGET=250 AGENTS_NAME=objagents; COLLECT_FLAGS="--no-videos" adds flags
# to t2v.sh collect; SKIP_CHECK=1 skips t2v.sh check;
# SESSION_SH replaces scripts/gpu/session.sh (tests).
# The model services must be up first (scripts/gpu/services.sh start). Started by you, they stay
# up when the agents' session ends; a session that starts them stops them at its end.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
EXP="${EXP:-object-full}"
TASKS="${TASKS:-0-9}"
PILOT="${PILOT:-object-pilot}"
SCENES="${SCENES:-25}"
WORKERS="${WORKERS:-2}"
PREPARE_WORKERS="${PREPARE_WORKERS:-6}"
DATASET_WORKERS="${DATASET_WORKERS:-4}"
PROGRAMS="${PROGRAMS-outputs/trace2vla/object-agent-test/teacher/programs}"
PILOT_PROGRAMS="${PILOT_PROGRAMS:-outputs/trace2vla/object-agent-test/teacher/programs outputs/trace2vla/preflight2/teacher/programs}"
AGENT_TEST="${AGENT_TEST:-object-agent-test}"
AGENT_LAYOUTS="${AGENT_LAYOUTS-all}"
LLM_BUDGET="${LLM_BUDGET:-25}"
LLM_TOTAL_BUDGET="${LLM_TOTAL_BUDGET:-250}"
#: Minutes one agent may need (the Bedrock key must outlive them all).
AGENT_MINUTES=75
AGENTS_NAME="${AGENTS_NAME:-objagents}"
SESSION_SH="${SESSION_SH:-scripts/gpu/session.sh}"
read -r -a COLLECT_EXTRA <<< "${COLLECT_FLAGS:-}"
SUITE=libero_object_swap
LAYOUTS="alt1-train,alt2-train,alt3-train,alt4-train,train"
T2V="${T2V_SH:-scripts/gpu/t2v.sh}"
OUT=outputs/trace2vla
PY=.venv-libero/bin/python

usage() { sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
run() { printf '+ %s\n' "$*"; "$@"; }
die() { echo "error: $*" >&2; exit 1; }

log_to() {  # log_to EXP STEP: everything this step prints also goes to the experiment's logs/
  local dir="$OUT/$1/logs"
  mkdir -p "$dir"
  local log
  log="$dir/$(date +%Y%m%d-%H%M%S)-object_collection-$2.log"
  exec > >(tee -a "$log") 2>&1
  echo "this step's log: $log"
}

services_up() {
  local port
  for port in 8114 8115 8116; do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || return 1; done
}
need_services() {
  services_up || die "the model services do not answer (ports 8114-8116): scripts/gpu/services.sh start"
}

missing_programs() {  # the experiment's tasks without a registered program, space separated
  "$PY" - "$OUT/$1" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
tasks = json.loads((root / "experiment.json").read_text()).get("tasks", [])
path = root / "teacher" / "programs" / "programs.json"
have = json.loads(path.read_text()) if path.is_file() else {}
print(" ".join(str(t) for t in tasks if str(t) not in have or not (path.parent / f"task_{int(t):02d}.py").is_file()))
PY
}

collect_alive() {  # true while a collection of the experiment runs (or has just been started)
  "$PY" - "$OUT/$1" <<'PY'
import sys
from pathlib import Path
from cap_harness.trace2vla.progress import background, collect_status
root = Path(sys.argv[1])
running = collect_status(root).get("status") in ("running", "waiting", "starting")
sys.exit(0 if running or any(b["stage"] == "collect" for b in background(root)) else 1)
PY
}

agents_pid() {  # the agents' session pid, if it still runs
  local file="$OUT/$EXP/logs/agents.pid" pid
  [[ -s "$file" ]] || return 1
  pid="$(cat "$file")"
  [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "session.sh" || return 1
  printf '%s\n' "$pid"
}

make_scenes() {  # make_scenes EXP FIRST-LAST: the five training layouts, that many scenes each
  local k
  for k in 1 2 3 4; do
    run $T2V scenes "$1" --add "alt$k-train" --replace --suite "${SUITE}_alt$k" --seeds "$2" --role train > /dev/null
  done
  run $T2V scenes "$1" --add train --replace --seeds "$2" --role train
}

add_programs() {  # add_programs EXP "FOLDER...": each task's program from the first folder with one
  local exp="$1" folders=($2) tasks task folder file registered found=0
  ((${#folders[@]})) || { echo "no programs registered here: each task's program comes from its coding agent"; return; }
  tasks="$("$PY" -c 'import json,sys; print(" ".join(str(t) for t in json.load(open(sys.argv[1]))["tasks"]))' "$OUT/$exp/experiment.json")"
  for task in $tasks; do
    for folder in "${folders[@]}"; do
      file="$folder/task_$(printf %02d "$task").py"
      [[ -f "$file" ]] || continue
      registered="$OUT/$exp/teacher/programs/task_$(printf %02d "$task").py"
      if cmp -s "$file" "$registered"; then
        echo "task $task: $file (registered already)"
      else
        run $T2V teacher-add "$exp" --program "$file" --task "$task" --force
      fi
      found=$((found + 1))
      break
    done
  done
  echo "$found task(s) with a program from ${folders[*]}; the others get a coding agent"
}

make_eval_scenes() {  # make_eval_scenes EXP: dev scenes (2001-2010) on each sibling layout
  local k
  for k in 1 2 3 4; do
    run $T2V scenes "$1" --add "alt$k-dev" --replace --suite "${SUITE}_alt$k" --seeds 2001-2010 --role dev > /dev/null
  done
}

bedrock_ready() {  # bedrock_ready AGENTS: credentials set, and the key outlives that many agents
  [[ -n "${ANTHROPIC_MODEL:-}" && "${CLAUDE_CODE_USE_BEDROCK:-}" == 1 ]] \
    || die "Bedrock is not configured: scripts/gpu/set_credentials.sh, then conda deactivate && conda activate aspire"
  if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
    local need=$((AGENT_MINUTES * $1)) status=0
    "$PY" scripts/gpu/bedrock_key_info.py --min-minutes "$need" || status=$?
    ((status != 2)) || die "the Bedrock key expires before the agents finish (~$need min): scripts/gpu/set_credentials.sh --api-key"
  fi
}

layouts_args() {  # the session's --llm-layouts, from AGENT_LAYOUTS
  [[ -z "$AGENT_LAYOUTS" ]] || printf '%s\n' --llm-layouts "$AGENT_LAYOUTS"
}

check() {  # check EXP [programs-later]: t2v.sh check; stop on any FAIL (programs may come later)
  if [[ "${SKIP_CHECK:-0}" == 1 ]]; then echo "t2v.sh check skipped (SKIP_CHECK=1)"; return; fi
  local out status=0
  printf '+ %s\n' "$T2V check $1"
  out="$($T2V check "$1" 2>&1)" || status=$?
  printf '%s\n' "$out" | grep -E '^\s+\[|checks? (passed|failed)'
  local failed
  failed="$(grep '\[FAIL\]' <<< "$out" || true)"
  [[ -n "${2:-}" ]] && failed="$(grep -v 'teacher programs' <<< "$failed" || true)"
  [[ -z "$failed" ]] || die "a check failed (above)"
}

demo_sets() {  # "a,b" -> "a-3x,b-3x"
  local out="" name
  for name in ${1//,/ }; do out+="${out:+,}$name-3x"; done
  printf '%s\n' "$out"
}

verdict() {  # verdict EXP: what the collection gave, and whether the pilot passed
  "$PY" - "$OUT/$1" <<'PY'
import sys
from pathlib import Path
from cap_harness.trace2vla.progress import overview
view = overview(Path(sys.argv[1]))
tot, status = view["totals"], view["collect"]
print(f"teacher episodes {tot['recorded']}/{tot['planned']}, solved {tot['solved']}, graded "
      f"{tot['graded']}, demos kept {tot['kept']} (rejected {tot['rejected']}), all tasks")
problems = []
wanted = set(status.get("tasks") or view["tasks"])
for s in view["scene_sets"]:
    if not s["in_collection"]:
        continue
    cells = [c for c in s["cells"] if c["task"] in wanted]
    recorded = sum(c["recorded"] for c in cells)
    solved = sum(c["solved"] for c in cells)
    kept = sum(c["kept"] for c in cells)
    print(f"  {s['name']:<12} recorded {recorded:>3}, solved {solved:>3}, kept {kept:>3}")
    for c in cells:
        if c["recorded"] < c["planned"]:
            problems.append(f"{s['name']} task {c['task']}: {c['recorded']}/{c['planned']} recorded {c.get('note') or ''}")
        elif c["solved"] < 0.6 * c["recorded"]:
            problems.append(f"{s['name']} task {c['task']}: the teacher solved only {c['solved']}/{c['recorded']}: does its program work on this layout?")
        elif c["kept"] < c["solved"]:
            print(f"    task {c['task']}: {c['solved'] - c['kept']} solved episode(s) not kept (conversion)")
problems += [f"conversion problem: {k} (its log in logs/collect/)" for k in (status.get("summary") or {}).get("prepare_problems", [])]
for p in problems:
    print("  PROBLEM:", p)
kept_here = sum(c["kept"] for s in view["scene_sets"] if s["in_collection"] for c in s["cells"] if c["task"] in wanted)
print("RESULT: passed" if not problems and kept_here else "RESULT: look at the problems above")
sys.exit(0 if not problems and kept_here else 1)
PY
}

# ---------------------------------------------------------------------------------- steps --
cmd_pilot() {
  local exp="$PILOT"
  log_to "$exp" pilot
  need_services
  step "Experiment $exp: tasks 0 and 1, scenes 1001-1003 on each of the 5 layouts"
  [[ -f "$OUT/$exp/experiment.json" ]] || run $T2V init "$exp" --suite "$SUITE" --tasks 0,1 --train-seeds 1001-1003
  make_scenes "$exp" 1001-1003
  add_programs "$exp" "$PILOT_PROGRAMS"
  step "Checks"
  check "$exp"
  step "Collection: the teacher, then grading and conversion in the background (30 episodes)"
  run $T2V collect "$exp" --scenes "$LAYOUTS" --tasks 0,1 --workers "$WORKERS" --prepare-workers 4 \
    "${COLLECT_EXTRA[@]+"${COLLECT_EXTRA[@]}"}" || true
  step "A dataset from all five layouts"
  if [[ -f "$OUT/$exp/datasets/pilot/manifest.json" ]]; then
    echo "dataset 'pilot' exists"
  else
    run $T2V dataset "$exp" --demos "$(demo_sets "$LAYOUTS")" --name pilot --workers 2 || true
  fi
  step "Verdict"
  verdict "$exp"
}

cmd_agent_test() {  # cmd_agent_test TASK [START_PROGRAM]: a new agent, or a repair of START_PROGRAM
  local task="${1:-}" start="${2:-}" kind=test
  [[ "$task" =~ ^[0-9]$ ]] || die "agent-test needs a task id 0-9 (scripts/gpu/object_collection.sh agent-test 0)"
  local exp="$AGENT_TEST"
  [[ -z "$start" ]] || kind=fix
  if [[ "$start" == registered ]]; then
    start="$OUT/$exp/teacher/programs/task_$(printf %02d "$task").py"
    [[ -f "$start" ]] || die "task $task has no program in $exp yet: run agent-test $task first, or name one (agent-fix $task FILE)"
  fi
  if [[ -n "$start" ]]; then
    [[ -f "$start" ]] || die "no such program: $start"
    start="$(cd "$(dirname "$start")" && pwd)/$(basename "$start")"
    # The registered copy is replaced when the repair is registered: keep the one repaired.
    mkdir -p "$OUT/$exp/logs"
    cp "$start" "$OUT/$exp/logs/agent-fix-task$task-start-$(date +%Y%m%d-%H%M%S).py"
  fi
  log_to "$exp" "agent-$kind-task$task"
  need_services
  bedrock_ready 1
  step "Experiment $exp: all 10 tasks, scenes 1001-1003 on each of the 5 layouts (made once)"
  [[ -f "$OUT/$exp/experiment.json" ]] || run $T2V init "$exp" --suite "$SUITE" --tasks 0-9 --train-seeds 1001-1003
  make_scenes "$exp" 1001-1003
  local name="agent-$kind-task$task" marker="$OUT/$exp/logs/.agent-test-start" layouts=() repair=()
  [[ -z "$start" ]] || repair=(--llm-start-from "$start")
  mapfile -t layouts < <(layouts_args)
  touch "$marker"
  step "Coding agent for task $task${start:+, repairing $start}, developing against ${AGENT_LAYOUTS:-the official layout only} (this terminal; ~30-75 min)"
  run "$SESSION_SH" run --name "$name" --tasks quick --no-preflight --llm "libero-pro:$SUITE:$task" \
    "${layouts[@]+"${layouts[@]}"}" "${repair[@]+"${repair[@]}"}" --llm-budget "$LLM_BUDGET" || true
  local session program
  session="$(find outputs/sessions -maxdepth 1 -type d -name "*-$name" -newer "$marker" 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "$session" ]] || die "the agent's session made no folder (its output above)"
  program="$session/agents/llm-libero-pro-$SUITE-$task/program_best.py"
  [[ -f "$program" ]] || die "the agent wrote no program_best.py ($session; scripts/gpu/session.sh why $session)"
  step "Its program on 3 new scenes of each layout (scenes it never saw)"
  run $T2V teacher-add "$exp" --program "$program" --task "$task" --force
  run $T2V collect "$exp" --scenes "$LAYOUTS" --tasks "$task" --workers "$WORKERS" --prepare-workers 4 \
    "${COLLECT_EXTRA[@]+"${COLLECT_EXTRA[@]}"}" || true
  step "Verdict for task $task"
  echo "agent session: $session"
  verdict "$exp"
}

cmd_setup() {
  log_to "$EXP" setup
  local last=$((1000 + SCENES))
  step "Experiment $EXP: tasks $TASKS, scenes 1001-$last on each of the 5 layouts"
  [[ -f "$OUT/$EXP/experiment.json" ]] || run $T2V init "$EXP" --suite "$SUITE" --tasks "$TASKS" --train-seeds "1001-$last"
  make_scenes "$EXP" "1001-$last"
  make_eval_scenes "$EXP"
  add_programs "$EXP" "$PROGRAMS"
  step "Checks (tasks without a program get theirs from the coding agents: start)"
  check "$EXP" programs-later
  echo
  echo "setup done; next: scripts/gpu/object_collection.sh start"
}

cmd_start() {
  [[ -f "$OUT/$EXP/experiment.json" ]] || die "no experiment $EXP: run setup first"
  log_to "$EXP" start
  need_services
  ! collect_alive "$EXP" || die "a collection of $EXP is running already ($0 status; $0 stop)"
  local missing session="" collect_args=()
  missing="$(missing_programs "$EXP")"
  if [[ -n "$missing" ]]; then
    local pid
    if pid="$(agents_pid)"; then
      session="$(cat "$OUT/$EXP/logs/agents.session" 2>/dev/null || true)"
      [[ -n "$session" ]] || die "the agents' session (pid $pid) is still starting; run start again in a minute"
      echo "the agents' session is running already (pid $pid): $session"
    else
      local agents=() task
      for task in $missing; do agents+=(--llm "libero-pro:$SUITE:$task"); done
      step "Coding agents for tasks $missing (\$$LLM_BUDGET each, \$$LLM_TOTAL_BUDGET in all; layouts: ${AGENT_LAYOUTS:-official only}), in the background"
      bedrock_ready "$(wc -w <<< "$missing")"
      local layouts=()
      mapfile -t layouts < <(layouts_args)
      local marker="$OUT/$EXP/logs/.agents-start" log
      touch "$marker"
      log="$OUT/$EXP/logs/$(date +%Y%m%d-%H%M%S)-agents.log"
      printf '+ %s\n' "$SESSION_SH run --name $AGENTS_NAME --tasks quick --no-preflight ${agents[*]} ${layouts[*]} --llm-budget $LLM_BUDGET --llm-total-budget $LLM_TOTAL_BUDGET"
      setsid nohup "$SESSION_SH" run --name "$AGENTS_NAME" --tasks quick --no-preflight \
        "${agents[@]}" "${layouts[@]+"${layouts[@]}"}" --llm-budget "$LLM_BUDGET" \
        --llm-total-budget "$LLM_TOTAL_BUDGET" > "$log" 2>&1 < /dev/null &
      pid=$!
      echo "$pid" > "$OUT/$EXP/logs/agents.pid"
      echo "agents' session started (pid $pid); its terminal output: $log"
      local i
      for i in $(seq 60); do
        session="$(find outputs/sessions -maxdepth 1 -type d -name "*-$AGENTS_NAME" -newer "$marker" 2>/dev/null | sort | tail -n 1 || true)"
        [[ -n "$session" ]] && break
        kill -0 "$pid" 2>/dev/null || { tail -n 20 "$log"; die "the agents' session stopped at once (its log above)"; }
        sleep 2
      done
      [[ -n "$session" ]] || die "the agents' session made no folder in 2 minutes; see $log"
      echo "$session" > "$OUT/$EXP/logs/agents.session"
      echo "agents' session folder: $session"
    fi
    collect_args=(--programs-from "$session" --follow)
  else
    echo "every task has a program: no coding agents needed"
  fi
  step "The collection, in the background"
  run $T2V collect --detach "$EXP" --scenes "$LAYOUTS" --workers "$WORKERS" \
    --prepare-workers "$PREPARE_WORKERS" "${collect_args[@]+"${collect_args[@]}"}" \
    "${COLLECT_EXTRA[@]+"${COLLECT_EXTRA[@]}"}"
  echo
  echo "watch: $0 status   (or the dashboard: Trace2VLA > $EXP > Pipeline)"
  echo "stop:  $0 stop     (finished work is kept; start continues it)"
}

cmd_status() {
  [[ -f "$OUT/$EXP/experiment.json" ]] || die "no experiment $EXP (setup first)"
  $T2V status "$EXP" 2>/dev/null | sed -n '1,/^last events/p' | sed '$d'
  local pid session
  session="$(cat "$OUT/$EXP/logs/agents.session" 2>/dev/null || true)"
  if pid="$(agents_pid)"; then echo "agents' session: running (pid $pid), $session"
  elif [[ -n "$session" ]]; then echo "agents' session: not running ($session)"; fi
  if [[ -n "$session" && -d "$session/agents" ]]; then
    local ws
    for ws in "$session"/agents/llm-*; do
      [[ -d "$ws" ]] || continue
      if [[ -f "$ws/result.json" ]]; then
        echo "  ${ws##*/}: finished$( [[ -f "$ws/heldout/batch.json" ]] && "$PY" -c 'import json,sys; b=json.load(open(sys.argv[1])); print(", held-out %s/%s" % (b.get("passes"), b.get("seeds")))' "$ws/heldout/batch.json")"
      else
        echo "  ${ws##*/}: working"
      fi
    done
  fi
}

cmd_stop() {
  $T2V stop "$EXP" collect || true
  local pid
  if pid="$(agents_pid)"; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    echo "stopping the agents' session (pid $pid); a finished agent's program is kept"
  else
    echo "no agents' session of $EXP is running"
  fi
}

cmd_datasets() {
  [[ -f "$OUT/$EXP/experiment.json" ]] || die "no experiment $EXP"
  if collect_alive "$EXP" && [[ "${FORCE:-0}" != 1 ]]; then
    die "the collection is still running: a dataset holds only the demos converted so far (FORCE=1 to build anyway)"
  fi
  log_to "$EXP" datasets
  build() {  # build NAME DEMO_SETS
    if [[ -f "$OUT/$EXP/datasets/$1/manifest.json" ]]; then echo "dataset '$1' exists"; return; fi
    run $T2V dataset "$EXP" --demos "$2" --name "$1" --per-task "$SCENES" --workers "$DATASET_WORKERS"
  }
  copy() {  # copy NAME: its pi0.5 copy (same frames, pi0.5's statistics)
    if [[ -f "$OUT/$EXP/datasets/$1-pi05/manifest.json" ]]; then echo "dataset '$1-pi05' exists"; return; fi
    run $T2V dataset "$EXP" --from-dataset "$1" --student pi05 --name "$1-pi05"
  }
  step "main-$SCENES: the new layouts alt1-3 (alt4, the official swap and the standard layout stay held out)"
  build "main-$SCENES" "$(demo_sets alt1-train,alt2-train,alt3-train)"
  copy "main-$SCENES"
  step "control-$SCENES: the official partner's layout (the preflight's setting)"
  build "control-$SCENES" "$(demo_sets train)"
  copy "control-$SCENES"
}

cmd_evaluate() {  # cmd_evaluate CHECKPOINT [STUDENT]
  local checkpoint="${1:-}" student="${2:-}"
  [[ -n "$checkpoint" ]] || die "evaluate needs a checkpoint: a training run, RUN@STEP, or base (with smolvla|pi05)"
  [[ -f "$OUT/$EXP/experiment.json" ]] || die "no experiment $EXP"
  log_to "$EXP" "evaluate"
  local tag="${checkpoint//@/-step}" extra=() set
  if [[ "$checkpoint" == base ]]; then
    [[ -n "$student" ]] || die "evaluate base needs the student: smolvla or pi05"
    tag="base-$student"
    extra=(--student "$student")
  fi
  step "$checkpoint on every layout, in the background (one evaluation per scene set; seed 0)"
  run $T2V eval --detach "$EXP" --name "$tag-official" --checkpoint "$checkpoint" --scenes official --seeds 0 "${extra[@]+"${extra[@]}"}"
  run $T2V eval --detach "$EXP" --name "$tag-standard" --checkpoint "$checkpoint" --suite libero_object --scenes official --seeds 0 "${extra[@]+"${extra[@]}"}"
  for set in dev alt1-dev alt2-dev alt3-dev alt4-dev; do
    run $T2V eval --detach "$EXP" --name "$tag-$set" --checkpoint "$checkpoint" --scenes "$set" --seeds 0 "${extra[@]+"${extra[@]}"}"
  done
  echo
  echo "seven evaluations started; results: scripts/gpu/t2v.sh status $EXP, or the dashboard (Evaluations)"
}

case "${1:-}" in
  agent-test) shift; cmd_agent_test "${1:-}" ;;
  agent-fix) shift; cmd_agent_test "${1:-}" "${2:-registered}" ;;
  pilot) cmd_pilot ;;
  setup) cmd_setup ;;
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  datasets) cmd_datasets ;;
  evaluate) shift; cmd_evaluate "${1:-}" "${2:-}" ;;
  -h|--help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
