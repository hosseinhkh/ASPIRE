#!/usr/bin/env bash
# The VLA student (SmolVLA through LeRobot) on LIBERO / LIBERO-Pro, in ASPIRE's own simulator.
#
#   scripts/gpu/vla.sh setup                      install LeRobot + SmolVLA into .venv-libero
#   scripts/gpu/vla.sh parity [--policy]          LeRobot's LIBERO env vs ASPIRE's adapter
#   scripts/gpu/vla.sh sanity [--quick]           smolvla_libero on standard LIBERO-Object
#   scripts/gpu/vla.sh baseline [--quick]         step 1: SmolVLA on libero_object_swap
#   scripts/gpu/vla.sh benchmark --suite S [...]  any suite; options of cap_harness.vla.benchmark
#   scripts/gpu/vla.sh status [RUN]               every run, or one run in detail
#   scripts/gpu/vla.sh stop RUN                   stop a running benchmark (it records why)
#   scripts/gpu/vla.sh report RUN                 rebuild a run's results table
#   scripts/gpu/vla.sh lerobot-eval ARGS...       LeRobot's own lerobot-eval on this LIBERO
#
# sanity, baseline and benchmark take --detach: run in the background, survive a closed
# terminal, log to RUN/console.log. Rerunning a command resumes its run. RUN is a folder under
# outputs/vla/ (or a path). GPU ${ASPIRE_VLA_GPU:-1} by default: set ASPIRE_VLA_GPU, or pass
# --gpu N first. The dashboard shows the runs under "VLA student" (scripts/gpu/dashboard.py).
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv-libero/bin/python"
VLA_OUT="$REPO_ROOT/outputs/vla"
GPU="${ASPIRE_VLA_GPU:-1}"

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; }
if [[ "${1:-}" == "--gpu" ]]; then GPU="$2"; shift 2; fi
PY_ENV=(
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL="${MUJOCO_GL:-egl}"
  PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
)

need_vla() {
  [[ -x "$VENV_PY" ]] || { echo "error: .venv-libero missing; run scripts/gpu/setup_conda_env.sh" >&2; exit 1; }
  "$VENV_PY" -c "import lerobot" 2>/dev/null || {
    echo "error: LeRobot is not installed in .venv-libero; run: scripts/gpu/vla.sh setup" >&2
    exit 1
  }
}

announce_gpu() {
  echo "VLA on GPU $GPU (CUDA_VISIBLE_DEVICES for this command only; the conda env is unchanged)" >&2
  if [[ -n "${ASPIRE_GPU:-}" && "$ASPIRE_GPU" == "$GPU" ]]; then
    echo "note: ASPIRE also uses GPU $GPU (ASPIRE_GPU). SmolVLA needs about 3 GB; pass --gpu N to move it." >&2
  fi
}

run_py() { env "${PY_ENV[@]}" "$VENV_PY" "$@"; }

run_dir() {  # a run name under outputs/vla, or a path
  local name="${1:?a run folder is required (scripts/gpu/vla.sh status lists them)}"
  if [[ -d "$name" ]]; then (cd "$name" && pwd); else printf '%s\n' "$VLA_OUT/$name"; fi
}

DETACH=0
ARGS=()
take_args() {  # split --detach off the remaining arguments
  ARGS=()
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "--detach" ]]; then DETACH=1; else ARGS+=("$arg"); fi
  done
}

launch() {  # launch OUT_DIR benchmark-args... : foreground, or detached with --detach
  local out="$1"
  shift
  if ((DETACH)); then
    mkdir -p "$out"
    setsid nohup env "${PY_ENV[@]}" "$VENV_PY" -m cap_harness.vla.benchmark "$@" --out "$out" \
      >> "$out/console.log" 2>&1 < /dev/null &
    local name
    name="$(basename "$out")"
    echo "started in the background (pid $!); it keeps running if you close the terminal"
    echo "  follow:  tail -f $out/benchmark.log"
    echo "  status:  scripts/gpu/vla.sh status $name"
    echo "  stop:    scripts/gpu/vla.sh stop $name"
  else
    run_py -m cap_harness.vla.benchmark "$@" --out "$out"
  fi
}

cmd="${1:-}"
[[ -n "$cmd" ]] && shift
case "$cmd" in parity|sanity|baseline|benchmark|lerobot-eval) announce_gpu ;; esac
case "$cmd" in
  setup)
    "$REPO_ROOT/scripts/bootstrap_libero.sh" --vla --python "${CAP_HARNESS_PYTHON:-python3}"
    echo "Downloading the SmolVLA LIBERO checkpoint (pinned revision)..."
    run_py - <<'PY'
from cap_harness.vla.policy import DEFAULT_POLICY, DEFAULT_REVISION, resolve_checkpoint
print(resolve_checkpoint(DEFAULT_POLICY, DEFAULT_REVISION))
PY
    run_py -m pytest -q -p no:cacheprovider "$REPO_ROOT/tests/test_vla.py"
    ;;
  parity)
    need_vla
    mkdir -p "$VLA_OUT"
    extra=()
    # The policy comparison runs on CPU: there both paths are bit-exact, while GPU kernels are
    # not run-to-run deterministic (the comparison checks the pipeline, not GPU numerics).
    [[ "${1:-}" == "--policy" ]] && extra=(--policy --device cpu)
    status=0
    for target in "libero_object 0 1" "libero_object_swap 3 7"; do
      read -r suite task scene <<< "$target"
      echo "== $suite task $task scene $scene"
      log="$VLA_OUT/parity-$suite.log"
      # The module prints its own summary as its last lines; LIBERO's own chatter and any
      # traceback go to the log.
      if run_py -m cap_harness.vla.parity --suite "$suite" --task "$task" --scene "$scene" \
        "${extra[@]+"${extra[@]}"}" --out "$VLA_OUT/parity-$suite.json" > "$log" 2>&1; then
        grep -E "^(environment|policy|report):" "$log"
      else
        status=1
        if grep -qE "^(environment|policy):" "$log"; then grep -E "^(environment|policy|report):" "$log"
        else echo "parity check crashed; last lines of $log:"; tail -n 15 "$log"; fi
      fi
    done
    exit "$status"
    ;;
  sanity)
    need_vla
    take_args "$@"
    set -- "${ARGS[@]+"${ARGS[@]}"}"
    scenes="1-50"
    [[ "${1:-}" == "--quick" ]] && { scenes="1-10"; shift; }
    # -n10: 10 actions per predicted chunk, the protocol default (cap_harness.vla.protocol).
    launch "$VLA_OUT/sanity-libero_object-n10" --suite libero_object --scenes "$scenes" --seeds 0 "$@"
    ;;
  baseline)
    need_vla
    take_args "$@"
    set -- "${ARGS[@]+"${ARGS[@]}"}"
    scenes="1-50" seeds="0,1,2"
    [[ "${1:-}" == "--quick" ]] && { scenes="1-10"; seeds="0"; shift; }
    launch "$VLA_OUT/baseline-smolvla-libero_object_swap-n10" --suite libero_object_swap \
      --scenes "$scenes" --seeds "$seeds" "$@"
    ;;
  benchmark)
    need_vla
    take_args "$@"
    set -- "${ARGS[@]+"${ARGS[@]}"}"
    out=""
    rest=()
    while (($#)); do
      if [[ "$1" == "--out" ]]; then out="$2"; shift 2; else rest+=("$1"); shift; fi
    done
    if [[ -z "$out" ]]; then
      ((DETACH)) && { echo "error: --detach needs --out RUN" >&2; exit 2; }
      run_py -m cap_harness.vla.benchmark "${rest[@]+"${rest[@]}"}"
    else
      case "$out" in
        /*) ;;
        outputs/*) out="$REPO_ROOT/$out" ;;
        *) out="$VLA_OUT/$out" ;;
      esac
      launch "$out" "${rest[@]+"${rest[@]}"}"
    fi
    ;;
  status)
    if [[ -n "${1:-}" ]]; then "$VENV_PY" -m cap_harness.vla.status "$(run_dir "$1")"
    else "$VENV_PY" -m cap_harness.vla.status; fi
    ;;
  stop)
    dir="$(run_dir "${1:-}")"
    pid="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid") or "")' "$dir/progress.json" 2>/dev/null || true)"
    if [[ -z "$pid" ]] || ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "cap_harness.vla.benchmark"; then
      echo "no running benchmark for $dir"
      exit 1
    fi
    kill -TERM "$pid"
    echo "stopping pid $pid: the episode in progress is dropped; rerun the same command to resume"
    ;;
  report)
    need_vla
    dir="$(run_dir "${1:-}")"
    suite="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["suite"])' "$dir/run.json")"
    run_py -m cap_harness.vla.benchmark --suite "$suite" --out "$dir" --report
    ;;
  lerobot-eval)
    need_vla
    run_py -m cap_harness.vla.lerobot_eval "$@"
    ;;
  *) usage; [[ -z "$cmd" ]] || exit 2 ;;
esac
