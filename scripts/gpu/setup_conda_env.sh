#!/usr/bin/env bash
# One-command ASPIRE setup on an NVIDIA GPU workstation, inside a conda environment.
#
# The conda env ("aspire") holds the toolchain: Python 3.11, uv, git, git-lfs, ffmpeg, Node 22,
# boto3 and the Claude Code CLI. ASPIRE's components keep their own isolated venvs, built from
# the frozen uv.lock exactly as upstream does, because the providers pin conflicting stacks
# (JAX for PyRoki, torch for SAM3 / Contact-GraspNet / cuRobo):
#
#   .venv-libero  .venv-robosuite                    simulators (client-only)
#   .venv-sam3  .venv-graspnet  .venv-pyroki  .venv-curobo   model services (HTTP)
#
# Every GPU wheel is prebuilt (torch cu128, cuRobo compiles its kernels at runtime), so no
# system CUDA toolkit is needed; the NVIDIA driver is. Re-running is safe: finished venvs are
# fingerprinted and skipped.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ASPIRE_CONDA_ENV:-aspire}"
GPU="${ASPIRE_GPU:-0}"
PROFILE="${CAP_HARNESS_PROFILE:-rtxpro6000}"
WITH_CUROBO=1
WITH_CLAUDE=1
RUN_TESTS=1

usage() {
  cat <<EOF
Usage: scripts/gpu/setup_conda_env.sh [options]

Options:
  --env NAME        conda environment name (default: $ENV_NAME)
  --gpu N           physical GPU for simulators/rendering (default: $GPU)
  --profile NAME    provider placement profile in configs/profiles (default: $PROFILE)
  --no-curobo       skip the cuRobo planner (saves ~3.4 GiB of GPU memory at runtime)
  --no-claude       do not install the Claude Code CLI
  --no-tests        skip the post-install unit-test pass
  -h, --help        show this help
EOF
}

while (($#)); do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --no-curobo) WITH_CUROBO=0; shift ;;
    --no-claude) WITH_CLAUDE=0; shift ;;
    --no-tests) RUN_TESTS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------------------------
step "Checking the GPU and driver"
command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi not found (NVIDIA driver missing)" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total --format=csv
cap="$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')"
if [[ "$cap" != "12.0" && "$PROFILE" == "rtxpro6000" ]]; then
  echo "warning: GPU $GPU reports compute capability '$cap', but profile rtxpro6000 is sm_120." >&2
  echo "         Pick a matching profile with --profile (see configs/profiles/)." >&2
fi

# ---------------------------------------------------------------------------------------------
step "Creating conda env '$ENV_NAME' (conda-forge only)"
if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH. Install Miniforge or Anaconda first:" >&2
  echo "  https://github.com/conda-forge/miniforge#install" >&2
  exit 1
fi
# conda's own activation hooks are not written for `set -u` (e.g. qt-main's activate.d script
# reads QT_XCB_GL_INTEGRATION unguarded), so relax it around anything that sources them.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
# --override-channels keeps the Anaconda 'defaults' channel (and its ToS prompt) out of it.
CONDA_PACKAGES=(python=3.11 uv git git-lfs ffmpeg "nodejs>=22" boto3 curl)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env exists; making sure its packages are present"
  conda install -y -n "$ENV_NAME" --override-channels -c conda-forge "${CONDA_PACKAGES[@]}"
else
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "${CONDA_PACKAGES[@]}"
fi
set +u
conda activate "$ENV_NAME"
set -u
PY="$CONDA_PREFIX/bin/python"
"$PY" --version

# ---------------------------------------------------------------------------------------------
if [[ "$WITH_CLAUDE" -eq 1 ]]; then
  step "Installing the Claude Code CLI into the env"
  if [[ -x "$CONDA_PREFIX/bin/claude" ]]; then
    echo "already installed: $("$CONDA_PREFIX/bin/claude" --version 2>/dev/null || echo '?')"
  else
    # conda's npm installs global packages under $CONDA_PREFIX, so `claude` lives in the env.
    npm install -g @anthropic-ai/claude-code
  fi
  claude --version
fi

# ---------------------------------------------------------------------------------------------
step "Fetching submodules (BEHAVIOR-1K is skipped: it needs Isaac Sim)"
# Shallow is enough for these. cuRobo is cloned in full by its own bootstrap: it derives its
# package version from git history (setuptools_scm).
git -C "$REPO_ROOT" submodule update --init --depth 1 \
  third_party/LIBERO-PRO third_party/robosuite third_party/sam3 \
  third_party/contact_graspnet_pytorch

step "robosuite: turning off its shared /tmp/robosuite.log file (multi-user machine)"
# shellcheck source=../lib/robosuite_macros.sh
source "$REPO_ROOT/scripts/lib/robosuite_macros.sh"
write_robosuite_private_macros "$REPO_ROOT"

# ---------------------------------------------------------------------------------------------
export CAP_HARNESS_PYTHON="$PY"
export CAP_HARNESS_PROFILE="$PROFILE"
export CAP_HARNESS_ROBOSUITE_CACHE="$REPO_ROOT/.uv-cache"
export CAP_HARNESS_CUROBO_GPU="$GPU"

step "Simulator: Robosuite (.venv-robosuite)"
"$REPO_ROOT/scripts/bootstrap_robosuite.sh"

step "Simulator: LIBERO-Pro (.venv-libero, ~7 GB with torch)"
"$REPO_ROOT/scripts/bootstrap_libero.sh" --python "$PY"

PROVIDERS="sam3,pyroki,contact_graspnet"
[[ "$WITH_CUROBO" -eq 1 ]] && PROVIDERS="$PROVIDERS,curobo"
step "Model services: $PROVIDERS (profile $PROFILE)"
"$REPO_ROOT/scripts/bootstrap_providers.sh" --providers "$PROVIDERS" --profile "$PROFILE"

step "Developer env for tests (.venv-dev)"
UV_CACHE_DIR="$REPO_ROOT/.uv-cache" uv venv --allow-existing --python "$PY" "$REPO_ROOT/.venv-dev"
VIRTUAL_ENV="$REPO_ROOT/.venv-dev" UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-dev" \
  UV_CACHE_DIR="$REPO_ROOT/.uv-cache" \
  uv sync --frozen --active --project "$REPO_ROOT" \
    --extra dev --extra agent --extra yam-real --extra robosuite

# ---------------------------------------------------------------------------------------------
step "Storing runtime variables in the conda env (active on every 'conda activate $ENV_NAME')"
# EGL_PLATFORM=device + MUJOCO_EGL_DEVICE_ID pin MuJoCo rendering to the chosen card: EGL device
# enumeration ignores CUDA_VISIBLE_DEVICES. expandable_segments limits torch fragmentation on a
# card shared with other users. The cuRobo model cache defaults to a shared /tmp path that
# another user may already own, so give it a per-user home.
conda env config vars set -n "$ENV_NAME" \
  ASPIRE_REPO="$REPO_ROOT" \
  ASPIRE_GPU="$GPU" \
  CAP_HARNESS_PROFILE="$PROFILE" \
  CAP_HARNESS_PYTHON="$PY" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  MUJOCO_EGL_DEVICE_ID="$GPU" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  EGL_PLATFORM=device \
  LIBERO_CONFIG_PATH="$REPO_ROOT/.libero" \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CAP_HARNESS_VENDOR_ROOT="$REPO_ROOT/third_party" \
  CAP_HARNESS_CUROBO_MODEL_CACHE="$HOME/.cache/aspire/curobo-models" >/dev/null
conda env config vars list -n "$ENV_NAME"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  step "Unit tests (CPU-only, about 6 minutes)"
  (cd "$REPO_ROOT" && "$REPO_ROOT/.venv-dev/bin/python" -m pytest -q -p no:cacheprovider) || {
    echo "warning: unit tests reported failures; see above" >&2
  }
fi

cat <<EOF

Setup complete. Next steps (docs/gpu-quickstart.md has the full walkthrough):

  conda activate $ENV_NAME                      # re-activate to load the stored variables
  scripts/gpu/set_credentials.sh                # AWS Bedrock keys + Hugging Face token
  scripts/gpu/check_stack.sh --quick            # env, GPU, reservation, torch, EGL, SAM3 access, Bedrock
  scripts/gpu/check_stack.sh                    # + services, LIBERO/Robosuite episodes, API probe
  python scripts/gpu/check_gpu.py budget        # does the stack fit on GPU $GPU right now?
  scripts/gpu/use_gpu.sh N                      # move everything to GPU N if another has more room
  scripts/gpu/services.sh start                 # SAM3, Contact-GraspNet, PyRoki$([[ "$WITH_CUROBO" -eq 1 ]] && echo ", cuRobo")
  scripts/gpu/demo.sh smoke                     # simulator + perception demo, no LLM
  scripts/gpu/demo.sh llm                       # Claude Opus 4.6 writes and debugs a program
EOF
