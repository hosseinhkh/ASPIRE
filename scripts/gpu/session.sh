#!/usr/bin/env bash
# Recorded test sessions: run simulator tasks (and optionally Claude Opus 4.6 coding agents) with
# everything recorded in one directory, then an HTML report and Weights & Biases runs.
#
#   scripts/gpu/session.sh run [--name NAME] [--tasks quick|standard|full|FILE] [--seeds "1 2"]
#                              [--llm BENCHMARK:SUITE:TASK_ID]... [--llm-tasks FILE]
#                              [--llm-budget USD] [--llm-total-budget USD]
#                              [--llm-max-turns N] [--llm-dev-seeds "51 52"]
#                              [--llm-heldout-seeds "1 2 3"] [--no-preflight] [--no-curobo]
#                              [--keep-services] [--graspnet-cpu] [--reserve GIB]
#                              [--wandb offline|online|off] [--gpu N]
#   scripts/gpu/session.sh report SESSION_DIR [--wandb offline|online|off]
#   scripts/gpu/session.sh claude-logs SESSION_DIR [--all]
#   scripts/gpu/session.sh why SESSION_DIR
#   scripts/gpu/session.sh tasks
#
# run          preflight checks, environment snapshot, model services, a GPU timeline, each task
#              (one episode per seed: videos, trajectories, every tool / model call), the agents,
#              then stops the services it started and re-holds --reserve GiB on the GPU.
# report       rebuild the HTML report (and W&B runs) for a session directory.
# claude-logs  copy Claude Code transcripts (with subagents) from ~/.claude/projects into the
#              session; use after an interactive campaign (scripts/gpu/demo.sh campaign).
# why          for each episode that did not succeed: how it ended, failed calls, the
#              program's result and its last printed lines (plain text, easy to share).
# tasks        print the task presets.
#
# Sessions land in outputs/sessions/<date>-<time>-<name>/ (report/index.html is the entry point).
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${CAP_HARNESS_PYTHON:-python3}"
GPU="${ASPIRE_GPU:-0}"
SERVICE_LOG_DIR="${CAP_HARNESS_SERVICE_LOG_DIR:-$REPO_ROOT/validation-artifacts/service-logs}"
TOOLS_VENV="${ASPIRE_TOOLS_VENV:-$REPO_ROOT/.venv-tools}"
declare -A PORTS=([sam3]=8114 [contact_graspnet]=8115 [pyroki]=8116 [curobo]=8118)

usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
port_up() { curl --fail --silent --max-time 2 "http://127.0.0.1:$1/openapi.json" >/dev/null; }
now() { printf '%s' "${EPOCHREALTIME:-$(date +%s.%N)}"; }
since_t0() { awk -v a="$1" -v b="$T0" 'BEGIN { printf "%.1f", a - b }'; }

# label | benchmark | suite | task id | program | max steps | camera WxH | services it needs
PRESET_QUICK="
robosuite-cube-lift|robosuite|cube_lifting|0|examples/robosuite/cube_lifting_seed1.py|3000|512x512|sam3,contact_graspnet,pyroki
libero-close-microwave|libero-pro|libero_90|33|examples/libero/close_microwave.py|2000|800x512|pyroki
libero-object-swap|libero-pro|libero_object_swap|0|examples/libero/object_swap_seed1.py|2000|800x512|sam3,contact_graspnet,pyroki"
PRESET_STANDARD="$PRESET_QUICK
robosuite-cube-lift-curobo|robosuite|cube_lifting|0|examples/robosuite/cube_lifting_curobo_seed1.py|3000|512x512|sam3,contact_graspnet,pyroki,curobo
robosuite-two-arm-lift|robosuite|two_arm_lift|0|examples/robosuite/two_arm_lift_seed1.py|3000|512x512|sam3,pyroki
robosuite-two-arm-handover|robosuite|two_arm_handover|0|examples/robosuite/two_arm_handover_seed1.py|7000|512x512|sam3,contact_graspnet,pyroki
libero-open-middle-drawer|libero-pro|libero_goal_swap|0|examples/libero/open_middle_drawer.py|3000|800x512|sam3,pyroki"
PRESET_FULL="$PRESET_STANDARD
robosuite-bimanual-control|robosuite|two_arm_lift|0|examples/robosuite/bimanual_control_smoke.py|2000|512x512|
libero-object-swap-curobo|libero-pro|libero_object_swap|0|examples/libero/object_swap_curobo_seed1.py|2000|800x512|sam3,contact_graspnet,pyroki,curobo
robosuite-two-arm-lift-curobo|robosuite|two_arm_lift|0|examples/robosuite/two_arm_lift_curobo_seed1.py|3000|512x512|sam3,pyroki,curobo
robosuite-two-arm-handover-curobo|robosuite|two_arm_handover|0|examples/robosuite/two_arm_handover_curobo_seed1.py|7000|512x512|sam3,pyroki,curobo
libero-bowl-in-drawer|libero-pro|libero_10_swap|3|examples/libero/put_bowl_in_bottom_drawer.py|4000|800x512|sam3,contact_graspnet,pyroki
libero-bottle-on-cabinet|libero-pro|libero_goal_swap|2|examples/libero/put_wine_bottle_on_drawer.py|3000|800x512|sam3,contact_graspnet,pyroki
libero-bottle-on-rack|libero-pro|libero_goal_swap|9|examples/libero/put_wine_bottle_on_rack.py|3000|800x512|sam3,contact_graspnet,pyroki"

task_list() {  # task_list PRESET_OR_FILE
  case "$1" in
    quick) printf '%s\n' "$PRESET_QUICK" ;;
    standard) printf '%s\n' "$PRESET_STANDARD" ;;
    full) printf '%s\n' "$PRESET_FULL" ;;
    *)
      [[ -f "$1" ]] || { echo "error: --tasks must be quick, standard, full or a file" >&2; exit 2; }
      cat "$1" ;;
  esac | sed -e 's/#.*//' -e '/^[[:space:]]*$/d'
}

cmd_tasks() {
  local preset
  for preset in quick standard full; do
    echo "$preset:"
    task_list "$preset" | awk -F'|' '{printf "  %-34s %s %s:%s  (needs %s)\n", $1, $2, $3, $4, ($8 == "" ? "no services" : $8)}'
  done
  cat <<'EOF'

A task file uses the same columns, one task per line ('#' starts a comment):
  label|benchmark|suite|task_id|program|max_steps|WIDTHxHEIGHT|services,it,needs
EOF
}

# ------------------------------------------------------------------------------- recording --
STAGE_OPEN=0
stage_begin() { STAGE_NAME="$1"; STAGE_T="$(now)"; STAGE_OPEN=1; }
stage_end() {  # stage_end STATUS DETAIL
  STAGE_OPEN=0
  local detail="${2//$'\t'/ }"
  detail="${detail//$'\r'/ }"
  detail="${detail//$'\n'/ }"  # one stage = one line in stages.tsv, whatever the detail holds
  printf '%s\t%s\t%s\t%s\t%s\n' "$STAGE_NAME" "$1" "$detail" "$(since_t0 "$STAGE_T")" \
    "$(since_t0 "$(now)")" >> "$SESSION/stages.tsv"
  printf '  %-7s %-36s %s\n' "$1" "$STAGE_NAME" "$detail"
}

write_meta() {  # write_meta KEY=VALUE... ; merges into session.json (values are strings or JSON)
  "$PY" - "$SESSION/session.json" "$@" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
meta = json.loads(path.read_text()) if path.exists() else {}
for pair in sys.argv[2:]:
    key, _, value = pair.partition("=")
    if (value and value[0] in "[{") or value in ("true", "false"):
        meta[key] = json.loads(value)
    else:
        meta[key] = value
path.write_text(json.dumps(meta, indent=2) + "\n")
PY
}

snapshot_env() {
  local env="$SESSION/env"
  mkdir -p "$env"
  {
    echo "commit: $(git rev-parse HEAD 2>/dev/null)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo; echo "status:"; git status --short 2>/dev/null
    echo; echo "recent commits:"; git log -5 --oneline 2>/dev/null
    echo; echo "submodules:"; git submodule status 2>/dev/null
  } > "$env/git.txt"
  git diff HEAD > "$env/git.diff" 2>/dev/null || true
  {
    nvidia-smi 2>&1
    echo
    nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.used,memory.total,power.limit \
      --format=csv 2>&1
  } > "$env/nvidia-smi.txt"
  {
    uname -a; echo "cpus: $(nproc)"; free -g 2>/dev/null | head -n 2
    "$PY" --version 2>&1; uv --version 2>&1; node --version 2>&1
    echo "claude: $(claude --version 2>&1 | head -n 1)"
    ffmpeg -version 2>&1 | head -n 1
  } > "$env/versions.txt"
  # Only names that matter to ASPIRE, and never a credential's value.
  "$PY" - > "$env/variables.txt" <<'PY'
import os, re
secret = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|SESSION", re.I)
prefixes = ("ASPIRE_", "CAP_HARNESS_", "MUJOCO", "PYOPENGL", "EGL_", "CUDA", "NVIDIA", "LIBERO",
            "PYTORCH", "TORCH", "ANTHROPIC_", "CLAUDE_", "AWS_", "HF_", "HUGGING", "WANDB_",
            "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "XLA_", "JAX_", "OMNIGIBSON")
for name in sorted(os.environ):
    if name.startswith(prefixes):
        value = os.environ[name]
        print(f"{name}={f'<set, {len(value)} chars>' if secret.search(name) else value}")
PY
  local venv name
  for venv in .venv-robosuite .venv-libero .venv-sam3 .venv-graspnet .venv-pyroki .venv-curobo; do
    [[ -x "$venv/bin/python" ]] || continue
    name="${venv#.venv-}"
    uv pip freeze --python "$venv/bin/python" > "$env/packages-$name.txt" 2>&1 || true
  done
  local profile="configs/profiles/${CAP_HARNESS_PROFILE:-rtxpro6000}.json"
  [[ -f "$profile" ]] && cp "$profile" "$env/provider-profile.json"
  return 0
}

declare -A LOG_OFFSETS=()
remember_service_logs() {  # note where each service log ends, so the session copies only its part
  local log
  for log in "$SERVICE_LOG_DIR"/*.log; do
    [[ -f "$log" ]] && LOG_OFFSETS["$log"]="$(stat -c %s "$log")"
  done
  return 0
}
copy_service_logs() {
  local log offset dest="$SESSION/service-logs"
  mkdir -p "$dest"
  for log in "$SERVICE_LOG_DIR"/*.log; do
    [[ -f "$log" ]] || continue
    offset="${LOG_OFFSETS[$log]:-0}"
    ((offset > $(stat -c %s "$log"))) && offset=0  # rotated or restarted
    tail -c "+$((offset + 1))" "$log" > "$dest/$(basename "$log")"
  done
  return 0
}

start_monitor() {
  mkdir -p "$SESSION/gpu"
  rm -f "$SESSION/gpu/.stop"
  "$PY" scripts/gpu/gpu_monitor.py --out "$SESSION/gpu" --interval 2 --t0 "$T0" \
    --stop-file "$SESSION/gpu/.stop" > "$SESSION/gpu/monitor.log" 2>&1 &
  MONITOR_PID=$!
}
stop_monitor() {
  [[ -n "${MONITOR_PID:-}" ]] || return 0
  touch "$SESSION/gpu/.stop"
  wait "$MONITOR_PID" 2>/dev/null
  MONITOR_PID=""
}

# ---------------------------------------------------------------------------------- report --
tools_python() {  # prints a Python with wandb installed (.venv-tools), creating it on first use
  if [[ -x "$TOOLS_VENV/bin/python" ]] && "$TOOLS_VENV/bin/python" -c "import wandb, PIL" 2>/dev/null; then
    printf '%s\n' "$TOOLS_VENV/bin/python"; return 0
  fi
  echo "installing wandb into $TOOLS_VENV (first use only)" >&2
  UV_CACHE_DIR="$REPO_ROOT/.uv-cache" uv venv --allow-existing --python "$PY" "$TOOLS_VENV" >&2 \
    && UV_CACHE_DIR="$REPO_ROOT/.uv-cache" uv pip install -q --python "$TOOLS_VENV/bin/python" \
      "wandb>=0.19" pillow numpy >&2 \
    && printf '%s\n' "$TOOLS_VENV/bin/python"
}

build_report() {  # build_report SESSION WANDB_MODE
  local session="$1" mode="$2" tools status=0
  "$PY" scripts/gpu/redact_session.py "$session" | tail -n 1
  "$PY" scripts/gpu/session_report.py "$session" || status=1
  [[ "$mode" == off ]] && return "$status"
  if ! tools="$(tools_python)"; then
    echo "warning: could not install wandb; skipping W&B (the HTML report has everything)" >&2
    return "$status"
  fi
  if ! OTEL_SDK_DISABLED=true "$tools" scripts/gpu/wandb_log.py "$session" --mode "$mode"; then
    if [[ "$mode" == online ]]; then
      echo "warning: W&B online logging failed (not logged in? run: $tools -m wandb login);" >&2
      echo "         writing offline runs instead; upload them later with wandb sync" >&2
      OTEL_SDK_DISABLED=true "$tools" scripts/gpu/wandb_log.py "$session" --mode offline || status=1
    else
      status=1
    fi
  fi
  return "$status"
}

cmd_report() {
  local session="" mode=offline
  while (($#)); do
    case "$1" in
      --wandb) mode="$2"; shift 2 ;;
      -*) echo "error: unknown option: $1" >&2; exit 2 ;;
      *) session="$1"; shift ;;
    esac
  done
  [[ -d "$session" ]] || { echo "error: session directory required" >&2; exit 2; }
  build_report "$(cd "$session" && pwd)" "$mode"
}

cmd_claude_logs() {  # copies transcripts written since the session started (all with --all)
  local session="" all=0
  while (($#)); do
    case "$1" in
      --all) all=1; shift ;;
      -*) echo "error: unknown option: $1" >&2; exit 2 ;;
      *) session="$1"; shift ;;
    esac
  done
  [[ -d "$session" ]] || { echo "error: session directory required" >&2; exit 2; }
  "$PY" - "$session" "$REPO_ROOT" "$all" "${CLAUDE_CONFIG_DIR:-$HOME/.claude}" <<'PY'
import json, re, shutil, sys
from datetime import datetime
from pathlib import Path

session, repo, take_all, config = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", Path(sys.argv[4])
meta = json.loads((session / "session.json").read_text()) if (session / "session.json").exists() else {}
since = 0.0
if not take_all and meta.get("started"):
    since = datetime.fromisoformat(meta["started"]).timestamp()
# Headless runs already left a stream-json transcript.jsonl; copy only their subagents.
captured = set()
for transcript in session.rglob("transcript.jsonl"):
    for line in transcript.read_text(errors="replace").splitlines()[:5]:
        try:
            captured.add(json.loads(line).get("session_id"))
        except ValueError:
            pass
projects = config / "projects"
slug = re.sub(r"[^A-Za-z0-9]", "-", repo)
roots = [p for p in projects.glob("*") if p.name == slug or p.name.startswith(slug + "-")]
copied = 0
for root in roots:
    for path in sorted(root.glob("*.jsonl")):
        if path.stat().st_mtime < since:
            continue
        dest = session / "claude-sessions" / root.name
        sub = path.with_suffix("")
        text = path.read_text(errors="replace")
        if path.stem not in captured and '"tool_use"' not in text:
            continue  # a one-line exchange such as check_stack's Claude Code check
        if path.stem not in captured:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
            copied += 1
        if (sub / "subagents").is_dir():
            target = dest / path.stem / "subagents"
            shutil.copytree(sub / "subagents", target, dirs_exist_ok=True)
            copied += len(list(target.glob("*.jsonl")))
print(f"claude-logs: copied {copied} transcript(s) from {projects} into {session / 'claude-sessions'}")
PY
}

# ------------------------------------------------------------------------------------- run --
run_task() {  # run_task LABEL BENCHMARK SUITE TASK PROGRAM MAX_STEPS WxH NEEDS SEED
  local label="$1" benchmark="$2" suite="$3" task="$4" program="$5" steps="$6" size="$7"
  local needs="$8" seed="$9" venv missing=() service
  stage_begin "episode $label seed $seed"
  for service in ${needs//,/ }; do
    port_up "${PORTS[$service]}" || missing+=("$service")
  done
  if ((${#missing[@]})); then stage_end SKIP "service down: ${missing[*]}"; return 0; fi
  case "$benchmark" in
    robosuite) venv=.venv-robosuite ;;
    libero-pro) venv=.venv-libero ;;
    *) stage_end SKIP "benchmark $benchmark is not run by sessions"; return 0 ;;
  esac
  local log="$SESSION/episodes/logs/$label-s$seed.log" status=0
  mkdir -p "$SESSION/episodes/logs"
  "$venv/bin/cap-harness" run --benchmark "$benchmark" --suite "$suite" --task-id "$task" \
    --seed "$seed" --program "$program" --max-steps "$steps" --camera-width "${size%x*}" \
    --camera-height "${size#*x}" --output-root "$SESSION/episodes" > "$log" 2>&1 || status=$?
  local summary run_dir health=""
  summary="$(grep -E '^program_ok=' "$log" | tail -n 1)"
  run_dir="$(grep -E '^run: ' "$log" | tail -n 1 | cut -c6-)"
  [[ -n "$run_dir" ]] && health="$("$PY" scripts/gpu/provider_health.py "$run_dir")"
  local verdict=FAIL
  if [[ "$summary" == program_ok=True*task_success=True* ]]; then verdict=PASS
  elif [[ "$summary" == program_ok=True* && "$program" == *_smoke.py ]]; then verdict=PASS
  elif [[ "$summary" == program_ok=True* ]]; then verdict=WARN
  fi
  [[ "$health" == *"SERVICE ERRORS"* ]] && verdict=FAIL
  stage_end "$verdict" "${summary:-exit $status, see episodes/logs/$label-s$seed.log}${health:+; $health}"
}

run_agent() {  # run_agent BENCHMARK:SUITE:TASK
  local spec="$1" benchmark suite task
  IFS=: read -r benchmark suite task <<< "$spec"
  local ws="$SESSION/agents/llm-$benchmark-$suite-$task"
  stage_begin "agent $spec"
  local args=(--benchmark "$benchmark" --suite "$suite" --task-id "${task:-0}" --out "$ws"
    --budget-usd "${LLM_BUDGET_NOW:-$LLM_BUDGET}" --max-turns "$LLM_MAX_TURNS")
  [[ -n "$LLM_DEV_SEEDS" ]] && args+=(--dev-seeds "$LLM_DEV_SEEDS")
  [[ -n "$LLM_HELDOUT_SEEDS" ]] && args+=(--heldout-seeds "$LLM_HELDOUT_SEEDS")
  local status=0
  scripts/gpu/demo.sh llm "${args[@]}" || status=$?
  local detail
  detail="$("$PY" - "$ws" <<'PY'
import json, sys
from pathlib import Path
ws = Path(sys.argv[1])
parts = []
result = ws / "result.json"
if result.exists():
    r = json.loads(result.read_text())
    kind = r.get("subtype") or ("error" if r.get("is_error") else "finished")
    parts.append(f"{kind}, {r.get('num_turns')} turns, ${r.get('total_cost_usd') or 0:.2f}")
for batch in sorted(ws.glob("dev_v*/batch.json")):
    b = json.loads(batch.read_text())
    parts.append(f"{batch.parent.name} {b.get('passes')}/{b.get('seeds')}")
held = ws / "heldout/batch.json"
if held.exists():
    b = json.loads(held.read_text())
    parts.append(f"held-out task_success {b.get('passes')}/{b.get('seeds')}")
print("; ".join(parts) or "no result written")
PY
)"
  if ((status)); then stage_end FAIL "exit $status; $detail"
  elif [[ "$detail" == *"held-out task_success 0/"* || "$detail" != *held-out* ]]; then stage_end WARN "$detail"
  else stage_end PASS "$detail"
  fi
}

STARTED_SERVICES=0 SERVICES_STOPPED=0 FINISHED=0 MONITOR_PID="" RESERVE=""
reserve_held() { scripts/gpu/reserve.sh status 2>/dev/null | sed -n 's/^held: *//p'; }
already_reserved() {  # the holder runs with the same target on the same GPU
  local target="${ASPIRE_STATE_DIR:-$HOME/.cache/aspire}/reservation.target" gpu gib
  [[ -f "$target" ]] || return 1
  read -r gpu gib < "$target"
  [[ "$gpu" == "$GPU" ]] && awk -v a="$gib" -v b="$RESERVE" 'BEGIN { exit !(a + 0 == b + 0) }' \
    && [[ "$(reserve_held)" == [0-9]* ]]
}
reservation_gpu_elsewhere() {  # prints the GPU of a running reservation on another card
  local target="${ASPIRE_STATE_DIR:-$HOME/.cache/aspire}/reservation.target" gpu gib
  [[ -f "$target" ]] || return 0
  read -r gpu gib < "$target"
  [[ "$gpu" != "$GPU" && "$(reserve_held)" == [0-9]* ]] && printf '%s\n' "$gpu"
  return 0
}
finish() {  # stop the monitor and the services this session started, then hold --reserve GiB
  stop_monitor
  copy_service_logs
  if ((STARTED_SERVICES && ! KEEP_SERVICES && ! SERVICES_STOPPED)); then
    step "Stopping the services this session started"
    scripts/gpu/services.sh stop || true
    SERVICES_STOPPED=1
  fi
  if [[ -n "$RESERVE" ]]; then
    step "Reserving ${RESERVE} GiB on GPU $GPU for later"
    stage_begin "reserve"
    local held_gpu
    held_gpu="$(reservation_gpu_elsewhere)"
    if [[ -n "$held_gpu" ]]; then
      # One reservation at a time: holding here would release the one on the other card.
      stage_end PASS "kept your reservation on GPU $held_gpu ($(reserve_held)); not moved to GPU $GPU (to move it: scripts/gpu/reserve.sh hold $RESERVE --gpu $GPU)"
    elif already_reserved; then
      # services.sh stop has just restored it; re-holding would free it for a moment.
      stage_end PASS "already holding the ${RESERVE} GiB target ($(reserve_held))"
    elif scripts/gpu/reserve.sh hold "$RESERVE" --gpu "$GPU"; then
      stage_end PASS "holding up to $RESERVE GiB"
    else
      stage_end WARN "could not hold the reservation (scripts/gpu/reserve.sh status)"
    fi
    RESERVE=""
  fi
}
on_exit() {
  ((FINISHED)) && return
  echo; echo "session interrupted; cleaning up (the partial session is kept)" >&2
  ((STAGE_OPEN)) && stage_end INTERRUPTED "stopped before it finished"
  finish
  write_meta status=interrupted "finished=$(date -Iseconds)"
  "$PY" scripts/gpu/redact_session.py "$SESSION" >/dev/null 2>&1
  "$PY" scripts/gpu/session_report.py "$SESSION" >/dev/null 2>&1
  echo "partial report: $SESSION/report/index.html" >&2
}

cmd_run() {
  local name="tasks" tasks="standard" seeds="1" preflight=1 curobo=1 wandb_mode=offline
  local graspnet_cpu=0
  local -a llm=()
  KEEP_SERVICES=0 LLM_BUDGET=10 LLM_MAX_TURNS=120 LLM_DEV_SEEDS="" LLM_HELDOUT_SEEDS=""
  local llm_total="" llm_spent=0
  while (($#)); do
    case "$1" in
      --name) name="$2"; shift 2 ;;
      --tasks) tasks="$2"; shift 2 ;;
      --seeds) seeds="$2"; shift 2 ;;
      --llm) llm+=("$2"); shift 2 ;;
      --llm-tasks)
        [[ -f "$2" ]] || { echo "error: --llm-tasks: no such file: $2" >&2; exit 2; }
        while read -r line; do llm+=("$line"); done < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$2")
        shift 2 ;;
      --llm-total-budget) llm_total="$2"; shift 2 ;;
      --llm-budget) LLM_BUDGET="$2"; shift 2 ;;
      --llm-max-turns) LLM_MAX_TURNS="$2"; shift 2 ;;
      --llm-dev-seeds) LLM_DEV_SEEDS="$2"; shift 2 ;;
      --llm-heldout-seeds) LLM_HELDOUT_SEEDS="$2"; shift 2 ;;
      --no-preflight) preflight=0; shift ;;
      --no-curobo) curobo=0; shift ;;
      --keep-services) KEEP_SERVICES=1; shift ;;
      --graspnet-cpu) graspnet_cpu=1; shift ;;
      --reserve) RESERVE="$2"; shift 2 ;;
      --wandb) wandb_mode="$2"; shift 2 ;;
      --gpu) GPU="$2"; shift 2 ;;
      *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
  done
  case "$wandb_mode" in offline|online|off) ;; *) echo "error: --wandb offline|online|off" >&2; exit 2 ;; esac
  if [[ -z "${MUJOCO_GL:-}" || -z "${LIBERO_CONFIG_PATH:-}" ]]; then
    echo "error: rendering variables are not set; run 'conda activate ${ASPIRE_CONDA_ENV:-aspire}'" >&2
    exit 1
  fi
  local task_lines
  task_lines="$(task_list "$tasks")" || exit 2
  ((curobo)) || task_lines="$(grep -v 'curobo' <<< "$task_lines")"
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "error: --name may use letters, digits, . _ -" >&2; exit 2; }

  SESSION="$REPO_ROOT/outputs/sessions/$(date +%Y%m%d-%H%M%S)-$name"
  mkdir -p "$SESSION"
  # tee ignores Ctrl-C, so it keeps logging (and the cleanup can still print) after the interrupt.
  exec > >(trap '' INT TERM; exec tee -a "$SESSION/session.log") 2>&1
  T0="$(now)"
  trap on_exit EXIT
  trap 'exit 130' INT TERM
  : > "$SESSION/stages.tsv"
  local llm_json
  llm_json="$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${llm[@]}")"
  write_meta "name=$name" "gpu=$GPU" "started=$(date -Iseconds)" "host=$(hostname)" \
    "git_commit=$(git rev-parse --short HEAD 2>/dev/null)" \
    "git_dirty=$([[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo true || echo false)" \
    "tasks=$tasks" "seeds=$seeds" "agents=$llm_json" "profile=${CAP_HARNESS_PROFILE:-}" \
    "model=${ANTHROPIC_MODEL:-}" "claude_effort=${ASPIRE_EFFORT:-max}" \
    "harness=${CAP_HARNESS_COMPAT:-fixed}" "reserve_gib=$RESERVE" "status=running"
  echo "session: $SESSION"
  echo "tasks: $tasks ($(wc -l <<< "$task_lines") per seed) | seeds: $seeds | agents: ${#llm[@]} | GPU $GPU"

  step "Environment snapshot"
  stage_begin "snapshot"
  snapshot_env
  stage_end PASS "git $(git rev-parse --short HEAD 2>/dev/null), env/ (no credential values)"

  if ((preflight)); then
    step "Preflight checks (scripts/gpu/check_stack.sh --quick)"
    stage_begin "preflight"
    local pre=0
    ASPIRE_CHECK_OUT="$SESSION/preflight" scripts/gpu/check_stack.sh --quick --gpu "$GPU" \
      $( ((curobo)) || echo --no-curobo ) || pre=$?
    if [[ -f "$SESSION/preflight/summary.txt" ]]; then
      while IFS='|' read -r status check detail; do
        printf 'preflight: %s\t%s\t%s\t\t\n' "$check" "$status" "$detail" >> "$SESSION/stages.tsv"
      done < "$SESSION/preflight/summary.txt"
    fi
    if ((pre)); then stage_end WARN "some checks failed; continuing (see preflight/)"
    else stage_end PASS "all quick checks passed"; fi
  fi

  step "GPU timeline (gpu/gpu.csv, gpu/processes.csv every 2 s)"
  start_monitor
  echo "  recording (pid $MONITOR_PID)"

  step "Model services"
  stage_begin "services"
  remember_service_logs
  local wanted=(sam3 contact_graspnet pyroki) down=() service
  ((curobo)) && wanted+=(curobo)
  for service in "${wanted[@]}"; do port_up "${PORTS[$service]}" || down+=("$service"); done
  if ((${#down[@]} == 0)); then
    stage_end PASS "already running; this session uses them and leaves them up"
  elif [[ "$(scripts/gpu/services.sh status 2>/dev/null)" == *"supervisor: running"* ]]; then
    # Someone's supervisor (maybe yours, started by hand) is up with part of the stack: use it
    # as it is and never stop it; tasks needing a missing service are skipped.
    stage_end WARN "supervisor already running without: ${down[*]} (scripts/gpu/services.sh stop, then rerun, to get them)"
  else
    local start_args=(--gpu "$GPU")
    ((curobo)) || start_args+=(--no-curobo)
    ((graspnet_cpu)) && start_args+=(--graspnet-cpu)
    LOG_OFFSETS=()  # the supervisor rewrites each log from the start
    if scripts/gpu/services.sh start "${start_args[@]}"; then
      STARTED_SERVICES=1
      stage_end PASS "started ${wanted[*]} on GPU $GPU"
    else
      STARTED_SERVICES=1
      stage_end FAIL "services.sh start failed; tasks that need a missing service are skipped"
    fi
  fi
  write_meta "services_started_by_session=$( ((STARTED_SERVICES)) && echo true || echo false)"

  step "Tasks (one episode per seed; logs in episodes/logs/)"
  local seed line
  for seed in $seeds; do
    while IFS='|' read -r label benchmark suite task program steps size needs; do
      [[ -n "$label" ]] || continue
      run_task "$label" "$benchmark" "$suite" "$task" "$program" "$steps" "$size" "$needs" "$seed"
    done <<< "$task_lines"
  done

  local spec cost
  for spec in "${llm[@]}"; do
    LLM_BUDGET_NOW="$LLM_BUDGET"
    if [[ -n "$llm_total" ]]; then
      LLM_BUDGET_NOW="$(awk -v t="$llm_total" -v s="$llm_spent" -v b="$LLM_BUDGET" \
        'BEGIN { r = t - s; printf "%.2f", (r < b ? r : b) }')"
      if awk -v r="$LLM_BUDGET_NOW" 'BEGIN { exit !(r < 0.5) }'; then
        stage_begin "agent $spec"
        stage_end SKIP "total LLM budget reached (\$$llm_spent of \$$llm_total spent)"
        continue
      fi
    fi
    step "Claude agent on $spec (budget \$$LLM_BUDGET_NOW${llm_total:+, \$$llm_spent of \$$llm_total spent so far})"
    run_agent "$spec"
    cost="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("total_cost_usd") or 0)' \
      "$SESSION/agents/llm-${spec//:/-}/result.json" 2>/dev/null || echo 0)"
    llm_spent="$(awk -v a="$llm_spent" -v b="$cost" 'BEGIN { printf "%.2f", a + b }')"
  done
  ((${#llm[@]})) && write_meta "llm_cost_usd=$llm_spent" "llm_total_budget_usd=${llm_total:-none}"
  if ((${#llm[@]})); then cmd_claude_logs "$SESSION"; fi

  finish

  local counts
  counts="$(awk -F'\t' '$1 !~ /^preflight: / {n[$2]++} END {for (s in n) printf "%s=%d ", s, n[s]}' "$SESSION/stages.tsv")"
  write_meta "finished=$(date -Iseconds)" "status=complete" "stage_counts=${counts% }"
  FINISHED=1

  step "Report"
  build_report "$SESSION" "$wandb_mode" || echo "warning: report or W&B logging had errors (above)" >&2

  step "Summary"
  awk -F'\t' '$1 !~ /^preflight: / {printf "  %-7s %-36s %s\n", $2, $1, $3}' "$SESSION/stages.tsv"
  cat <<EOF

Session:  $SESSION
Browse:   python scripts/gpu/dashboard.py   (videos, every call, agent conversation, zip downloads)
Report:   $SESSION/report/index.html
          (on a remote server: cd $SESSION && python -m http.server 8000 --bind 127.0.0.1,
           then ssh -L 8000:localhost:8000 <server> and open http://localhost:8000/report/)
EOF
  if [[ "$wandb_mode" == offline ]]; then
    echo "W&B:      offline runs in $SESSION/wandb; upload: $TOOLS_VENV/bin/wandb sync $SESSION/wandb/wandb/offline-run-*"
  fi
  return 0
}

main() {
  case "${1:-}" in
    run) shift; cmd_run "$@" ;;
    report) shift; cmd_report "$@" ;;
    claude-logs) shift; cmd_claude_logs "$@" ;;
    why) shift; [[ -d "${1:-}" ]] || { echo "error: session directory required" >&2; exit 2; }
      "$PY" scripts/gpu/episode_summary.py --unsolved "$1" ;;
    tasks) cmd_tasks ;;
    -h|--help|"") usage ;;
    *) echo "error: unknown command: $1" >&2; usage >&2; exit 2 ;;
  esac
}
# One line, so bash never reads further: editing or pulling this file mid-session is safe.
main "$@"; exit $?
