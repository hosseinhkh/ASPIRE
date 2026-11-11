#!/usr/bin/env bash
# Trace2VLA: ASPIRE teacher episodes -> training data -> fine-tuned SmolVLA -> comparison.
#
#   scripts/gpu/t2v.sh init NAME --suite libero_object_swap --tasks 0,1
#   scripts/gpu/t2v.sh check NAME                 sanity checks (environment, services, simulator)
#   scripts/gpu/t2v.sh teacher-add NAME --from SESSION_OR_CAMPAIGN_DIR
#   scripts/gpu/t2v.sh teacher-run NAME --scenes official|train|dev [--workers 2]
#   scripts/gpu/t2v.sh teacher-grade NAME --scenes SET
#   scripts/gpu/t2v.sh convert NAME --scenes train [--speed 3]
#   scripts/gpu/t2v.sh dataset NAME --demos train-3x --name d50 [--per-task 50]
#   scripts/gpu/t2v.sh train NAME --dataset d50 --name ft-full [--method full|lora|lora-vlm]
#   scripts/gpu/t2v.sh eval NAME --name after --checkpoint ft-full [--scenes official] [--seeds 0]
#   scripts/gpu/t2v.sh grade-student RUN_DIR      failure labels for an existing student run
#   scripts/gpu/t2v.sh report NAME                teacher vs student before vs after
#   scripts/gpu/t2v.sh status [NAME]              what exists, what is running, last events
#   scripts/gpu/t2v.sh logs NAME                  the newest log of the experiment, followed
#   scripts/gpu/t2v.sh stop NAME                  stop this experiment's background stage
#
# Put --detach right after the stage name to run it in the background (teacher-run, teacher-grade,
# convert, dataset, train, eval): it survives a closed terminal and logs to
# outputs/trace2vla/NAME/logs/. GPU: --gpu N first, or ASPIRE_T2V_GPU (default 0). teacher-run
# keeps the conda env's own GPU, where ASPIRE's model services are.
# The dashboard shows experiments under "Trace2VLA" (python scripts/gpu/dashboard.py).
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv-libero/bin/python"
OUT="$REPO_ROOT/outputs/trace2vla"
GPU="${ASPIRE_T2V_GPU:-0}"

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }
if [[ "${1:-}" == "--gpu" ]]; then GPU="$2"; shift 2; fi
cmd="${1:-}"
[[ -n "$cmd" ]] && shift
DETACH=0
if [[ "${1:-}" == "--detach" ]]; then DETACH=1; shift; fi

[[ -x "$VENV_PY" ]] || { echo "error: .venv-libero missing; run scripts/gpu/setup_conda_env.sh" >&2; exit 1; }
BASE_ENV=(
  MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
)
# Stages that only simulate, train or evaluate run on the chosen GPU. teacher-run uses the
# conda env's GPU settings: its simulators render where ASPIRE's services already are.
GPU_ENV=(CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU")

experiment_dir() {
  local name="${1:?an experiment name is required (scripts/gpu/t2v.sh status lists them)}"
  if [[ -d "$name" ]]; then (cd "$name" && pwd); else printf '%s\n' "$OUT/$name"; fi
}

run_stage() {  # run_stage STAGE ARGS...
  local stage="$1"; shift
  local env=("${BASE_ENV[@]}")
  [[ "$stage" == teacher-run ]] || env+=("${GPU_ENV[@]}")
  if ((DETACH)); then
    local dir log
    dir="$(experiment_dir "${1:-}")"
    mkdir -p "$dir/logs"
    log="$dir/logs/$(date +%Y%m%d-%H%M%S)-$stage.log"
    setsid nohup env "${env[@]}" "$VENV_PY" -m cap_harness.trace2vla "$stage" "$@" > "$log" 2>&1 < /dev/null &
    echo "$! $stage" > "$dir/logs/running.pid"
    echo "started $stage in the background (pid $!)"
    echo "  follow:  tail -f $log"
    echo "  status:  scripts/gpu/t2v.sh status $(basename "$dir")"
    echo "  stop:    scripts/gpu/t2v.sh stop $(basename "$dir")"
  else
    env "${env[@]}" "$VENV_PY" -m cap_harness.trace2vla "$stage" "$@"
  fi
}

case "$cmd" in
  init|check|teacher-add|teacher-import|report|status)
    run_stage "$cmd" "$@" ;;
  teacher-run)
    echo "teacher-run uses the conda env's GPU (ASPIRE_GPU=${ASPIRE_GPU:-unset}) and the model services" >&2
    run_stage "$cmd" "$@" ;;
  teacher-grade|convert|dataset|train|eval|grade-student)
    echo "$cmd on GPU $GPU (this command only; --gpu N or ASPIRE_T2V_GPU to change)" >&2
    run_stage "$cmd" "$@" ;;
  logs)
    dir="$(experiment_dir "${1:-}")"
    log="$(ls -t "$dir"/logs/*.log 2>/dev/null | head -n 1 || true)"
    [[ -n "$log" ]] || { echo "no background log in $dir/logs" >&2; exit 1; }
    echo "== $log"
    tail -n 40 -f "$log"
    ;;
  stop)
    dir="$(experiment_dir "${1:-}")"
    read -r pid stage < "$dir/logs/running.pid" 2>/dev/null || { echo "no background stage recorded for $dir"; exit 1; }
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "cap_harness.trace2vla"; then
      # The stage and everything it started (trainer, simulator workers) share one session.
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
      echo "stopping $stage (pid $pid); finished episodes are kept, rerun the same command to continue"
    else
      echo "$stage (pid $pid) is not running any more"
    fi
    ;;
  *) usage; [[ -z "$cmd" ]] || exit 2 ;;
esac
