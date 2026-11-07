#!/usr/bin/env bash
# CPU-only development bootstrap: everything that runs without an NVIDIA GPU.
#
# Builds, from the same frozen uv.lock as the GPU bootstraps:
#   .venv-robosuite  Robosuite simulator client      (scripts/bootstrap_robosuite.sh)
#   .venv-libero     LIBERO-Pro simulator client     (scripts/bootstrap_libero.sh)
#   .venv-pyroki     PyRoki IK provider, JAX on CPU  (scripts/bootstrap_pyroki.sh)
#   .venv-dev        dev + agent + yam-real + robosuite extras, for the full test suite
#
# Not built here, because they need CUDA: SAM3 (also a gated Hugging Face model),
# Contact-GraspNet, cuRobo, and BEHAVIOR-1K (Isaac Sim, RTX GPU). See docs/cpu-dev-setup.md.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-}"
INSTALL_SYSTEM_DEPS=0
SKIP=()

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_cpu_dev.sh [options]

Options:
  --python PATH            Python 3.10-3.12 interpreter (default: first of python3.11,
                           python3.12, python3.10 on PATH; uv downloads 3.11 if none).
  --install-system-deps    apt-get install the Mesa OSMesa/EGL libraries MuJoCo needs for
                           headless rendering (needs root or sudo).
  --skip NAME              Skip one of: robosuite, libero, pyroki, dev (repeatable).
  -h, --help               Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --install-system-deps) INSTALL_SYSTEM_DEPS=1; shift ;;
    --skip) SKIP+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

skipped() {
  local name="$1" item
  for item in "${SKIP[@]:-}"; do [[ "$item" == "$name" ]] && return 0; done
  return 1
}

command -v uv >/dev/null || {
  printf 'error: uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh\n' >&2
  exit 1
}

# pyproject requires >=3.10,<3.13. Ubuntu 24.04 and newer distros may ship a newer
# default `python3`, which the GPU bootstraps would otherwise pick up and fail on.
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.11 python3.12 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$candidate"; break; fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  uv python install 3.11
  PYTHON_BIN="$(uv python find 3.11)"
fi
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"error: Python {sys.version.split()[0]} is outside >=3.10,<3.13")
PY
export CAP_HARNESS_PYTHON="$PYTHON_BIN"
printf 'Using %s (%s)\n' "$PYTHON_BIN" "$("$PYTHON_BIN" --version)"

if [[ "$INSTALL_SYSTEM_DEPS" -eq 1 ]]; then
  SUDO=""
  [[ "$(id -u)" -eq 0 ]] || SUDO="sudo"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
    libosmesa6 libegl1 libegl-mesa0 libgl1 libglu1-mesa ffmpeg git-lfs
fi
if ! ldconfig -p 2>/dev/null | grep -q libOSMesa; then
  printf 'warning: libOSMesa not found; MuJoCo cameras need it without a GPU.\n' >&2
  printf '         re-run with --install-system-deps, or: apt-get install libosmesa6\n' >&2
fi

# Shallow clones keep the checkout small; GitHub serves the pinned commits directly.
# BEHAVIOR-1K is deliberately skipped: it needs Isaac Sim on an RTX GPU.
git -C "$REPO_ROOT" submodule update --init --depth 1 \
  third_party/LIBERO-PRO third_party/robosuite third_party/sam3 \
  third_party/contact_graspnet_pytorch

export CAP_HARNESS_ROBOSUITE_CACHE="${CAP_HARNESS_ROBOSUITE_CACHE:-$REPO_ROOT/.uv-cache}"
skipped robosuite || "$REPO_ROOT/scripts/bootstrap_robosuite.sh"
skipped libero || "$REPO_ROOT/scripts/bootstrap_libero.sh" --python "$PYTHON_BIN"
skipped pyroki || "$REPO_ROOT/scripts/bootstrap_pyroki.sh" --profile rtx4090

if ! skipped dev; then
  UV_CACHE_DIR="$REPO_ROOT/.uv-cache" uv venv --allow-existing --python "$PYTHON_BIN" \
    "$REPO_ROOT/.venv-dev"
  VIRTUAL_ENV="$REPO_ROOT/.venv-dev" UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-dev" \
    UV_CACHE_DIR="$REPO_ROOT/.uv-cache" \
    uv sync --frozen --active --project "$REPO_ROOT" \
      --extra dev --extra agent --extra yam-real --extra robosuite
fi

cat <<EOF

CPU development bootstrap complete.

Headless MuJoCo rendering without a GPU (EGL needs an NVIDIA device):
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export LIBERO_CONFIG_PATH="$REPO_ROOT/.libero"

Start the PyRoki IK provider on CPU (port 8116):
  JAX_PLATFORMS=cpu $REPO_ROOT/.venv-pyroki/bin/python -m cap_harness.providers.pyroki.service --port 8116

Run the test suite:
  $REPO_ROOT/.venv-dev/bin/python -m pytest -q

Provider-free smoke (Robosuite, bimanual control):
  $REPO_ROOT/.venv-robosuite/bin/cap-harness run --benchmark robosuite --suite two_arm_lift \\
    --task-id 0 --seed 1 --program examples/robosuite/bimanual_control_smoke.py \\
    --output-root outputs/cpu-smoke --camera-width 256 --camera-height 256

PyRoki-only smoke (LIBERO-90 close microwave; needs the PyRoki service above):
  $REPO_ROOT/.venv-libero/bin/cap-harness run --benchmark libero-pro --suite libero_90 \\
    --task-id 33 --seed 1 --program examples/libero/close_microwave.py \\
    --output-root outputs/cpu-smoke --max-steps 2000 --camera-width 256 --camera-height 256
EOF
