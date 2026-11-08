#!/usr/bin/env bash
# How fast is Contact-GraspNet on this machine: on the GPU, and on the CPU?
#
#   scripts/gpu/check_graspnet.sh [--gpu N] [--requests 3]
#
# Starts two temporary Contact-GraspNet services next to anything already running (GPU on
# :8125, CPU on :8126), times the same synthetic tabletop request on each (one inference per
# request), shows how busy the GPU is, and stops both. The running stack is not touched.
# If the GPU is much slower than the CPU (another user's job keeping it at 100 %), run
# Contact-GraspNet on the CPU: scripts/gpu/services.sh start --graspnet-cpu
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
GPU="${ASPIRE_GPU:-0}"
REQUESTS=3
while (($#)); do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done
PY="$REPO_ROOT/.venv-graspnet/bin/python"
[[ -x "$PY" ]] || { echo "error: .venv-graspnet is missing (scripts/gpu/setup_conda_env.sh)" >&2; exit 1; }
OUT="$REPO_ROOT/outputs/gpu-check/graspnet-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
export CAP_HARNESS_VENDOR_ROOT="${CAP_HARNESS_VENDOR_ROOT:-$REPO_ROOT/third_party}"
PIDS=()
cleanup() { ((${#PIDS[@]})) && kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
port_up() { curl --fail --silent --max-time 2 "http://127.0.0.1:$1/openapi.json" >/dev/null; }

start() {  # start NAME PORT CUDA_VISIBLE_DEVICES DEVICE
  local name="$1" port="$2" visible="$3" device="$4"
  if port_up "$port"; then echo "error: port $port is already in use" >&2; exit 1; fi
  CUDA_VISIBLE_DEVICES="$visible" "$PY" -m cap_harness.providers.graspnet.service \
    --port "$port" --device "$device" > "$OUT/$name.log" 2>&1 &
  PIDS+=($!)
}

echo "starting Contact-GraspNet on GPU $GPU (:8125) and on CPU (:8126); logs in $OUT"
start gpu 8125 "$GPU" cuda
start cpu 8126 "" cpu
for _ in $(seq 1 180); do
  port_up 8125 && port_up 8126 && break
  sleep 1
done
for item in gpu:8125 cpu:8126; do
  port_up "${item#*:}" || { echo "error: the ${item%%:*} service did not start; see $OUT/${item%%:*}.log" >&2; tail -n 5 "$OUT/${item%%:*}.log" >&2; exit 1; }
done
for name in gpu cpu; do
  printf '  %s instance: %s\n' "$name" "$(grep -o 'loaded on .*' "$OUT/$name.log" | head -n 1)"
done

echo
echo "GPU $GPU right now (all users):"
nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
  | sed 's/^/  utilization, memory used, total: /'
for item in gpu:8125 cpu:8126; do
  echo
  echo "${item%%:*}:"
  "$PY" scripts/gpu/checks/graspnet_bench.py "http://127.0.0.1:${item#*:}" --requests "$REQUESTS" \
    | tee "$OUT/bench-${item%%:*}.txt" | sed 's/^/  /'
done

echo
gpu_s="$(awk '/^median/ {print $2}' "$OUT/bench-gpu.txt")"
cpu_s="$(awk '/^median/ {print $2}' "$OUT/bench-cpu.txt")"
if [[ -n "$gpu_s" && -n "$cpu_s" ]]; then
  awk -v g="$gpu_s" -v c="$cpu_s" 'BEGIN {
    printf "GPU %.1f s vs CPU %.1f s per inference: ", g, c
    if (g > c) print "the GPU is slower here; use scripts/gpu/services.sh start --graspnet-cpu"
    else if (g * 3 > c) print "similar; either works (--graspnet-cpu frees GPU memory)"
    else print "the GPU is faster; keep Contact-GraspNet on the GPU"
  }'
fi
