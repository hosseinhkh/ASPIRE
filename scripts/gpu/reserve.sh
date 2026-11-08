#!/usr/bin/env bash
# Keep a block of GPU memory for yourself on a shared server, and hand it to ASPIRE when needed.
#
#   scripts/gpu/reserve.sh hold [GIB] [--gpu N]   hold GIB GiB (default 30) on GPU N (default $ASPIRE_GPU)
#   scripts/gpu/reserve.sh status                 what is held, and the GPU's free memory
#   scripts/gpu/reserve.sh release                give it all back and forget the target
#
# scripts/gpu/services.sh hands the memory over automatically. `start` pauses the hold, starts
# the services, then re-holds the target minus what the services now use, leaving ~2 GiB free
# for the simulator. So your total stays near the target. `stop` re-holds the full target.
# Between pause and resume, memory is briefly unheld; on a busy card someone may take some of it.
#
# The holder is a detached process (setsid), so it survives closing the terminal.
# Holding idle memory blocks colleagues: agree on it with whoever shares the machine.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="${ASPIRE_STATE_DIR:-$HOME/.cache/aspire}"
TARGET_FILE="$STATE_DIR/reservation.target"   # "<gpu> <gib>"
PID_FILE="$STATE_DIR/reservation.pid"
STATUS_FILE="$STATE_DIR/reservation.status.json"
LOG_FILE="$STATE_DIR/reservation.log"
DEVICE="${ASPIRE_RESERVE_DEVICE:-cuda}"         # "cpu" only for testing the scripts
SIM_HEADROOM_GIB="${ASPIRE_RESERVE_SIM_HEADROOM_GIB:-2}"
PY="${CAP_HARNESS_PYTHON:-python3}"
mkdir -p "$STATE_DIR"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

torch_python() {
  local candidate
  for candidate in .venv-sam3 .venv-libero .venv-graspnet; do
    if [[ -x "$REPO_ROOT/$candidate/bin/python" ]]; then
      printf '%s\n' "$REPO_ROOT/$candidate/bin/python"; return 0
    fi
  done
  echo "error: no venv with torch found (run scripts/gpu/setup_conda_env.sh first)" >&2
  return 1
}

holder_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

stop_holder() {
  local pid
  if pid="$(holder_pid)"; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE" "$STATUS_FILE"
}

start_holder() {  # start_holder GPU GIB
  local gpu="$1" gib="$2" py
  py="$(torch_python)"
  stop_holder
  if ! awk -v g="$gib" 'BEGIN { exit !(g > 0) }'; then
    echo "nothing to hold (${gib} GiB)"; return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$py" "$REPO_ROOT/scripts/gpu/gpu_reserve.py" \
    --gib "$gib" --status-file "$STATUS_FILE" --device "$DEVICE" >"$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 120); do
    [[ -s "$STATUS_FILE" ]] && break
    if ! holder_pid >/dev/null; then
      echo "error: the holder exited; see $LOG_FILE" >&2; tail -n 5 "$LOG_FILE" >&2; return 1
    fi
    sleep 1
  done
  [[ -s "$STATUS_FILE" ]] || { echo "error: the holder did not report within 120 s" >&2; return 1; }
  "$PY" - "$STATUS_FILE" "$gpu" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))
held, asked = status["held_gib"], status["requested_gib"]
# The holder takes quarter-GiB blocks, rounded down; only a shortfall beyond that is news.
note = "" if held >= int(asked * 4) / 4 else f"  (only {held:g} GiB was free; the rest was not available)"
print(f"holding {held:g} GiB on GPU {sys.argv[2]} (pid {status['pid']}){note}")
PY
}

ours_on_gpu_gib() {  # memory used on GPU by your ASPIRE processes other than the holder
  local gpu="$1" tmp
  command -v nvidia-smi >/dev/null || { echo 0; return; }
  tmp="$(mktemp)"
  "$PY" "$REPO_ROOT/scripts/gpu/check_gpu.py" measure --gpu "$gpu" --json "$tmp" >/dev/null 2>&1 || true
  "$PY" - "$tmp" "$REPO_ROOT/scripts/gpu" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print(0); raise SystemExit
roles = data.get("peak_by_role_mib", {})
# Services grow during inference (SAM3 especially), so count each running one at its planned
# peak, not at the idle size it shows right after start.
sys.path.insert(0, sys.argv[2])
from check_gpu import PLANNED_MIB  # noqa: E402

total = sum(max(mib, PLANNED_MIB.get(role, 0)) for role, mib in roles.items() if role != "reservation")
# A running service that has not allocated yet (cuRobo builds its planner on the first request)
# still needs its share; services moved to CPU (ASPIRE_CPU_PROVIDERS) do not.
import os, urllib.request  # noqa: E401,E402
on_cpu = set(filter(None, os.environ.get("ASPIRE_CPU_PROVIDERS", "").split(",")))
for role, port in (("sam3", 8114), ("contact_graspnet", 8115), ("curobo", 8118)):
    if role in roles or role in on_cpu:
        continue
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=2).close()
    except OSError:
        continue
    total += PLANNED_MIB.get(role, 0)
print(round(total / 1024, 2))
PY
  rm -f "$tmp"
}

cmd_hold() {
  local gib=30 gpu="${ASPIRE_GPU:-0}"
  while (($#)); do
    case "$1" in
      --gpu) gpu="$2"; shift 2 ;;
      [0-9]*) gib="$1"; shift ;;
      *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  printf '%s %s\n' "$gpu" "$gib" > "$TARGET_FILE"
  start_holder "$gpu" "$gib"
}

cmd_status() {
  if [[ -f "$TARGET_FILE" ]]; then
    read -r gpu gib < "$TARGET_FILE"
    echo "target: ${gib} GiB on GPU ${gpu}"
  else
    echo "target: none"
  fi
  if holder_pid >/dev/null && [[ -s "$STATUS_FILE" ]]; then
    "$PY" - "$STATUS_FILE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"held:   {s['held_gib']:g} GiB since {s['since']} (pid {s['pid']})")
PY
  else
    echo "held:   nothing (holder not running)"
  fi
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
  fi
}

cmd_release() {
  stop_holder
  rm -f "$TARGET_FILE"
  echo "released; no reservation target"
}

# Used by services.sh -------------------------------------------------------------------------
cmd_pause() {  # free the memory but keep the target
  [[ -f "$TARGET_FILE" ]] || return 0
  if holder_pid >/dev/null; then
    stop_holder
    read -r gpu gib < "$TARGET_FILE"
    echo "reservation: released ${gib} GiB on GPU ${gpu} for the services (target kept)"
  fi
}

cmd_resume() {  # re-hold the target minus what our ASPIRE processes already use on that GPU
  [[ -f "$TARGET_FILE" ]] || return 0
  read -r gpu gib < "$TARGET_FILE"
  local ours want
  ours="$(ours_on_gpu_gib "$gpu")"
  want="$(awk -v t="$gib" -v o="$ours" -v h="$SIM_HEADROOM_GIB" \
    'BEGIN { w = t - o; if (o > 0) w -= h; if (w < 0) w = 0; printf "%.2f", w }')"
  echo "reservation: target ${gib} GiB on GPU ${gpu}; ASPIRE services need up to ${ours} GiB there; re-holding ${want} GiB"
  start_holder "$gpu" "$want"
}

case "${1:-}" in
  hold) shift; cmd_hold "$@" ;;
  status) cmd_status ;;
  release) cmd_release ;;
  pause) cmd_pause ;;
  resume) cmd_resume ;;
  -h|--help|"") usage ;;
  *) echo "error: unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
