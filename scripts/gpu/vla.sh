#!/usr/bin/env bash
# The VLA student (SmolVLA through LeRobot) on LIBERO / LIBERO-Pro, in ASPIRE's own simulator.
#
#   scripts/gpu/vla.sh setup                      install LeRobot + SmolVLA into .venv-libero
#   scripts/gpu/vla.sh parity [--policy]          LeRobot's LIBERO env vs ASPIRE's adapter
#   scripts/gpu/vla.sh sanity [--quick]           smolvla_libero on standard LIBERO-Object
#   scripts/gpu/vla.sh benchmark --suite S [...]  any suite; options of cap_harness.vla.benchmark
#   scripts/gpu/vla.sh baseline [--quick]         step 1: SmolVLA on libero_object_swap
#   scripts/gpu/vla.sh report DIR                 rebuild a run's results table
#   scripts/gpu/vla.sh lerobot-eval ARGS...       LeRobot's own lerobot-eval on this LIBERO
#
# Runs on GPU ${ASPIRE_VLA_GPU:-1} (set ASPIRE_VLA_GPU or pass --gpu N first), so ASPIRE keeps
# its own GPU.  Results go to outputs/vla/.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv-libero/bin/python"
GPU="${ASPIRE_VLA_GPU:-1}"

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
if [[ "${1:-}" == "--gpu" ]]; then GPU="$2"; shift 2; fi

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

run_py() {  # run a module on the VLA GPU, rendering there too
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}" \
    "$VENV_PY" "$@"
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
    mkdir -p "$REPO_ROOT/outputs/vla"
    extra=()
    # The policy comparison runs on CPU: there both paths are bit-exact, while GPU kernels are
    # not run-to-run deterministic (the comparison checks the pipeline, not GPU numerics).
    [[ "${1:-}" == "--policy" ]] && extra=(--policy --device cpu)
    status=0
    for target in "libero_object 0 1" "libero_object_swap 3 7"; do
      read -r suite task scene <<< "$target"
      echo "== $suite task $task scene $scene"
      run_py -m cap_harness.vla.parity --suite "$suite" --task "$task" --scene "$scene" "${extra[@]}" \
        --out "$REPO_ROOT/outputs/vla/parity-$suite.json" 2>/dev/null \
        | "$VENV_PY" -c 'import json,sys; r=json.load(sys.stdin); e=r["environment"]; p=r.get("policy");
print("environment:", "PASS" if e["passed"] else "FAIL", {k: v for k, v in e["worst"].items() if v})
if p: print("policy:", "PASS" if p["passed"] else "FAIL", p["lerobot"], p["aspire"], "max action difference", p["max_action_difference"])' \
        || status=1
    done
    exit "$status"
    ;;
  sanity)
    need_vla
    scenes="1-50"
    [[ "${1:-}" == "--quick" ]] && { scenes="1-10"; shift; }
    run_py -m cap_harness.vla.benchmark --suite libero_object --scenes "$scenes" --seeds 0 \
      --out "$REPO_ROOT/outputs/vla/sanity-libero_object" "$@"
    ;;
  baseline)
    need_vla
    scenes="1-50" seeds="0,1,2"
    [[ "${1:-}" == "--quick" ]] && { scenes="1-10"; seeds="0"; shift; }
    run_py -m cap_harness.vla.benchmark --suite libero_object_swap --scenes "$scenes" --seeds "$seeds" \
      --out "$REPO_ROOT/outputs/vla/baseline-smolvla-libero_object_swap" "$@"
    ;;
  benchmark)
    need_vla
    run_py -m cap_harness.vla.benchmark "$@"
    ;;
  report)
    need_vla
    dir="${1:?usage: vla.sh report DIR}"
    suite="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["suite"])' "$dir/run.json")"
    run_py -m cap_harness.vla.benchmark --suite "$suite" --out "$dir" --report
    ;;
  lerobot-eval)
    need_vla
    run_py -m cap_harness.vla.lerobot_eval "$@"
    ;;
  *) usage; [[ -z "$cmd" ]] || exit 2 ;;
esac
