#!/usr/bin/env bash
# Start / stop / inspect ASPIRE's model services on a shared GPU workstation.
#
# Wraps scripts/supervise_services.sh (the upstream supervisor) and detaches it from the terminal
# with setsid, so the stack survives a closed SSH session. One supervisor at a time; its process
# group is what `stop` kills.
#
#   scripts/gpu/services.sh start [--gpu N] [--no-curobo] [--no-graspnet] [--providers CSV]
#   scripts/gpu/services.sh status
#   scripts/gpu/services.sh logs [sam3|contact_graspnet|pyroki|curobo|supervisor]
#   scripts/gpu/services.sh stop
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${CAP_HARNESS_PROFILE:-rtxpro6000}"
GPU="${ASPIRE_GPU:-0}"
LOG_DIR="${CAP_HARNESS_SERVICE_LOG_DIR:-$REPO_ROOT/validation-artifacts/service-logs}"
PID_FILE="$LOG_DIR/supervisor.pid"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"
declare -A PORTS=([sam3]=8114 [contact_graspnet]=8115 [pyroki]=8116 [curobo]=8118)

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

port_up() {
  curl --fail --silent --max-time 2 "http://127.0.0.1:$1/openapi.json" >/dev/null
}

cmd_status() {
  if pid="$(running_pid)"; then
    echo "supervisor: running (pid $pid, log $LOG_DIR/supervisor.log)"
  else
    echo "supervisor: not running"
  fi
  local name
  for name in sam3 contact_graspnet pyroki curobo; do
    if port_up "${PORTS[$name]}"; then
      printf '  %-17s up    :%s\n' "$name" "${PORTS[$name]}"
    else
      printf '  %-17s down  :%s\n' "$name" "${PORTS[$name]}"
    fi
  done
  if command -v nvidia-smi >/dev/null; then
    echo
    "$PYTHON_BIN" "$REPO_ROOT/scripts/gpu/check_gpu.py" measure --gpu "$GPU" || true
  fi
}

cmd_start() {
  local providers="sam3,pyroki,contact_graspnet,curobo"
  while (($#)); do
    case "$1" in
      --no-curobo) providers="${providers/,curobo/}"; shift ;;
      --no-graspnet) providers="${providers/,contact_graspnet/}"; shift ;;
      --providers) providers="$2"; shift 2 ;;
      --profile) PROFILE="$2"; shift 2 ;;
      --gpu) GPU="$2"; shift 2 ;;
      *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  if pid="$(running_pid)"; then
    echo "already running (pid $pid); use 'stop' first, or 'status'"
    exit 0
  fi
  mkdir -p "$LOG_DIR"
  # SAM3's first start downloads ~3.4 GB of gated weights; give readiness enough time.
  echo "starting [$providers] on GPU $GPU, profile $PROFILE (logs: $LOG_DIR)"
  CAP_HARNESS_GPU_OVERRIDE="$GPU" \
  CAP_HARNESS_SERVICE_READY_TIMEOUT_S="${CAP_HARNESS_SERVICE_READY_TIMEOUT_S:-1800}" \
  CAP_HARNESS_SERVICE_LOG_DIR="$LOG_DIR" \
    setsid nohup "$REPO_ROOT/scripts/supervise_services.sh" \
      --profile "$PROFILE" --providers "$providers" \
      >"$LOG_DIR/supervisor.log" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  local deadline=$((SECONDS + ${CAP_HARNESS_SERVICE_READY_TIMEOUT_S:-1800} + 60))
  while ((SECONDS < deadline)); do
    if grep -q "all provider services ready" "$LOG_DIR/supervisor.log" 2>/dev/null; then
      cat "$LOG_DIR/supervisor.log"
      if [[ ",$providers," == *",curobo,"* ]]; then
        echo "note: cuRobo builds its planner on the first plan request (~50 s); the first"
        echo "      cuRobo call in a run is slow, later ones take tens of milliseconds."
      fi
      echo
      cmd_status
      return 0
    fi
    if ! running_pid >/dev/null; then
      echo "error: supervisor exited before all services were ready:" >&2
      tail -n 30 "$LOG_DIR/supervisor.log" >&2
      local name
      for name in ${providers//,/ }; do
        [[ -f "$LOG_DIR/$name.log" ]] && { echo "--- $name.log (tail)" >&2; tail -n 15 "$LOG_DIR/$name.log" >&2; }
      done
      exit 1
    fi
    sleep 5
  done
  echo "error: services not ready in time; see $LOG_DIR" >&2
  exit 1
}

cmd_stop() {
  if pid="$(running_pid)"; then
    # setsid made the supervisor a process-group leader; TERM the whole group.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "stopped"
  else
    echo "not running"
  fi
}

cmd_logs() {
  local name="${1:-supervisor}"
  tail -n 50 -f "$LOG_DIR/$name.log"
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "$@" ;;
  -h|--help|"") usage ;;
  *) echo "error: unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
