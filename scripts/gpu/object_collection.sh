#!/usr/bin/env bash
# The libero_object_swap data collection, one command per step. Each step prints every command
# before running it, stops at the first error, and keeps its log in the experiment's logs/.
#
#   scripts/gpu/object_collection.sh pilot      light test (~20-30 min, in this terminal): tasks 0-1
#                                               x 5 layouts x 3 scenes through the whole pipeline
#                                               (experiment object-pilot), a dataset, a verdict
#   scripts/gpu/object_collection.sh setup      the experiment object-full: 10 tasks, 25 scenes on
#                                               each of the 5 layouts, the programs of tasks 0-1,
#                                               then t2v.sh check
#   scripts/gpu/object_collection.sh start      in the background: the coding agents for the tasks
#                                               without a program (one session), and t2v.sh collect,
#                                               which takes each program as its agent finishes
#   scripts/gpu/object_collection.sh status     where it stands (also: the dashboard's Pipeline tab)
#   scripts/gpu/object_collection.sh stop       stop both; finished work is kept, start continues
#   scripts/gpu/object_collection.sh datasets   after the collection: main-25 (alt1-3), control-25
#                                               (the official partner's layout), their pi0.5 copies
#
# Settings (environment variables): EXP=object-full TASKS=0-9 PILOT=object-pilot SCENES=25
# WORKERS=2 PREPARE_WORKERS=6 DATASET_WORKERS=4 PROGRAMS=outputs/trace2vla/preflight2/teacher/programs
# LLM_BUDGET=25 LLM_TOTAL_BUDGET=200 AGENTS_NAME=objagents; COLLECT_FLAGS="--no-videos" adds flags
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
PROGRAMS="${PROGRAMS:-outputs/trace2vla/preflight2/teacher/programs}"
LLM_BUDGET="${LLM_BUDGET:-25}"
LLM_TOTAL_BUDGET="${LLM_TOTAL_BUDGET:-200}"
AGENTS_NAME="${AGENTS_NAME:-objagents}"
SESSION_SH="${SESSION_SH:-scripts/gpu/session.sh}"
read -r -a COLLECT_EXTRA <<< "${COLLECT_FLAGS:-}"
SUITE=libero_object_swap
LAYOUTS="alt1-train,alt2-train,alt3-train,alt4-train,train"
T2V=scripts/gpu/t2v.sh
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

add_programs() {  # tasks 0 and 1 from the preflight, unless they have one already
  local missing
  missing=" $(missing_programs "$1") "
  if [[ "$missing" == *" 0 "* || "$missing" == *" 1 "* ]]; then
    [[ -d "$PROGRAMS" ]] || die "no programs folder $PROGRAMS (set PROGRAMS=...)"
    run $T2V teacher-add "$1" --from "$PROGRAMS"
  else
    echo "tasks 0 and 1 have their programs"
  fi
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
      f"{tot['graded']}, demos kept {tot['kept']} (rejected {tot['rejected']})")
problems = []
for s in view["scene_sets"]:
    if not s["in_collection"]:
        continue
    recorded = sum(c["recorded"] for c in s["cells"])
    solved = sum(c["solved"] for c in s["cells"])
    kept = sum(c["kept"] for c in s["cells"])
    print(f"  {s['name']:<12} recorded {recorded:>3}, solved {solved:>3}, kept {kept:>3}")
    for c in s["cells"]:
        if c["recorded"] < c["planned"]:
            problems.append(f"{s['name']} task {c['task']}: {c['recorded']}/{c['planned']} recorded {c.get('note') or ''}")
        elif c["solved"] < 0.6 * c["recorded"]:
            problems.append(f"{s['name']} task {c['task']}: the teacher solved only {c['solved']}/{c['recorded']}: does its program work on this layout?")
        elif c["kept"] < c["solved"]:
            print(f"    task {c['task']}: {c['solved'] - c['kept']} solved episode(s) not kept (conversion)")
problems += [f"conversion problem: {k} (its log in logs/collect/)" for k in (status.get("summary") or {}).get("prepare_problems", [])]
for p in problems:
    print("  PROBLEM:", p)
print("RESULT: passed" if not problems and tot["kept"] else "RESULT: look at the problems above")
sys.exit(0 if not problems and tot["kept"] else 1)
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
  add_programs "$exp"
  step "Checks"
  check "$exp"
  step "Collection: the teacher, then grading and conversion in the background (30 episodes)"
  run $T2V collect "$exp" --scenes "$LAYOUTS" --workers "$WORKERS" --prepare-workers 4 \
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

cmd_setup() {
  log_to "$EXP" setup
  local last=$((1000 + SCENES))
  step "Experiment $EXP: tasks $TASKS, scenes 1001-$last on each of the 5 layouts"
  [[ -f "$OUT/$EXP/experiment.json" ]] || run $T2V init "$EXP" --suite "$SUITE" --tasks "$TASKS" --train-seeds "1001-$last"
  make_scenes "$EXP" "1001-$last"
  add_programs "$EXP"
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
      step "Coding agents for tasks $missing (\$$LLM_BUDGET each, \$$LLM_TOTAL_BUDGET in all), in the background"
      [[ -n "${ANTHROPIC_MODEL:-}" && "${CLAUDE_CODE_USE_BEDROCK:-}" == 1 ]] \
        || die "Bedrock is not configured: scripts/gpu/set_credentials.sh, then conda deactivate && conda activate aspire"
      if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
        # Agents run one after another, up to ~45 min each: the key must outlive them all.
        local need=$((45 * $(wc -w <<< "$missing"))) status=0
        "$PY" scripts/gpu/bedrock_key_info.py --min-minutes "$need" || status=$?
        ((status != 2)) || die "the Bedrock key expires before the agents finish (~$need min): scripts/gpu/set_credentials.sh --api-key"
      fi
      local marker="$OUT/$EXP/logs/.agents-start" log
      touch "$marker"
      log="$OUT/$EXP/logs/$(date +%Y%m%d-%H%M%S)-agents.log"
      printf '+ %s\n' "$SESSION_SH run --name $AGENTS_NAME --tasks quick --no-preflight ${agents[*]} --llm-budget $LLM_BUDGET --llm-total-budget $LLM_TOTAL_BUDGET"
      setsid nohup "$SESSION_SH" run --name "$AGENTS_NAME" --tasks quick --no-preflight \
        "${agents[@]}" --llm-budget "$LLM_BUDGET" --llm-total-budget "$LLM_TOTAL_BUDGET" \
        > "$log" 2>&1 < /dev/null &
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

case "${1:-}" in
  pilot) cmd_pilot ;;
  setup) cmd_setup ;;
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  datasets) cmd_datasets ;;
  -h|--help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
