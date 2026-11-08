#!/usr/bin/env bash
# Move ASPIRE to another physical GPU (on a shared box the card with free memory changes).
#
#   scripts/gpu/use_gpu.sh 1
#
# Stores ASPIRE_GPU, CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID in the conda env. The
# simulators render there, and scripts/gpu/services.sh places every model service there
# (via CAP_HARNESS_GPU_OVERRIDE). Then shows whether the stack fits on that card right now.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ASPIRE_CONDA_ENV:-aspire}"
GPU="${1:-}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
command -v conda >/dev/null 2>&1 || { echo "error: conda not found" >&2; exit 1; }

conda env config vars set -n "$ENV_NAME" \
  ASPIRE_GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" >/dev/null
echo "conda env '$ENV_NAME' now targets GPU $GPU"
echo
python3 "$REPO_ROOT/scripts/gpu/check_gpu.py" budget --gpu "$GPU" || true
cat <<EOF

Apply it:
  conda deactivate && conda activate $ENV_NAME
  scripts/gpu/services.sh stop && scripts/gpu/services.sh start   # if services were running
EOF
