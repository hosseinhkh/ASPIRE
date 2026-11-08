#!/usr/bin/env bash
# Check every part of the ASPIRE stack on this machine, without needing SAM3, and list blockers.
#
#   scripts/gpu/check_stack.sh [--quick] [--with-llm] [--no-curobo] [--keep-services] [--gpu N]
#
#   1 env        conda env variables (rendering GPU, LIBERO config)
#   2 gpu        driver, GPUs, memory budget on the chosen GPU
#   3 reserve    hold 1 GiB, see it on the GPU, release it (scripts/gpu/reserve.sh)
#   4 torch      each torch venv runs a CUDA kernel on the GPU (Blackwell sm_120)
#   5 egl        MuJoCo renders through NVIDIA EGL (not software)
#   6 sam3       Hugging Face access to facebook/sam3 (BLOCKED while the request is pending)
#   7 bedrock    API-key expiry, boto3 Converse + ConverseStream on Opus 4.6, Claude Code reply
#   8 services   PyRoki, Contact-GraspNet, cuRobo (+ SAM3 once granted) start and answer
#   9 sims       Robosuite bimanual control (no services); LIBERO close-microwave (PyRoki)
#  10 probe      every service called through the robot API in a Robosuite episode
#  11 agent      (--with-llm, ~$1-3) Opus 4.6 writes, runs and checks a small program
#
# --quick runs 1-8 only. Statuses: PASS, WARN, FAIL, BLOCKED (waiting on something outside this
# machine, e.g. SAM3 approval), SKIP. Exit code 1 if anything FAILed. Logs: outputs/gpu-check/.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
GPU="${ASPIRE_GPU:-0}"
PY="${CAP_HARNESS_PYTHON:-python3}"
QUICK=0 WITH_LLM=0 WITH_CUROBO=1 KEEP_SERVICES=0
while (($#)); do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --with-llm) WITH_LLM=1; shift ;;
    --no-curobo) WITH_CUROBO=0; shift ;;
    --keep-services) KEEP_SERVICES=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done
OUT="$REPO_ROOT/outputs/gpu-check/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
declare -a ROWS=()
SAM3_STATUS="UNKNOWN"
STARTED_SERVICES=0

c_status() {
  case "$1" in
    PASS) printf '\033[32m%-7s\033[0m' "$1" ;;
    FAIL) printf '\033[31m%-7s\033[0m' "$1" ;;
    WARN|BLOCKED) printf '\033[33m%-7s\033[0m' "$1" ;;
    *) printf '%-7s' "$1" ;;
  esac
}
record() {  # record STATUS NAME DETAIL
  ROWS+=("$1|$2|$3")
  printf '  %s %-28s %s\n' "$(c_status "$1")" "$2" "$3"
}
section() { printf '\n\033[1m[%s]\033[0m\n' "$*"; }
port_up() { curl --fail --silent --max-time 2 "http://127.0.0.1:$1/openapi.json" >/dev/null; }
last_line() { tail -n 1 "$1" 2>/dev/null; }

# 1 ---------------------------------------------------------------------------------------------
section "1 env"
if [[ "${MUJOCO_GL:-}" == "egl" && -n "${LIBERO_CONFIG_PATH:-}" && -n "${ASPIRE_REPO:-}" ]]; then
  record PASS "conda env variables" "MUJOCO_GL=egl, LIBERO_CONFIG_PATH set, ASPIRE_GPU=${ASPIRE_GPU:-unset}"
else
  record FAIL "conda env variables" "run 'conda activate ${ASPIRE_CONDA_ENV:-aspire}' (after scripts/gpu/setup_conda_env.sh)"
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "$GPU" && "${MUJOCO_EGL_DEVICE_ID:-}" == "$GPU" ]]; then
  record PASS "rendering GPU" "CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID = $GPU"
else
  record WARN "rendering GPU" "env says CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-unset}; checking GPU $GPU (fix: scripts/gpu/use_gpu.sh $GPU)"
fi
for venv in .venv-robosuite .venv-libero .venv-pyroki .venv-sam3 .venv-graspnet; do
  [[ -x "$venv/bin/python" ]] || record FAIL "venv $venv" "missing; run scripts/gpu/setup_conda_env.sh"
done
if [[ "$WITH_CUROBO" -eq 1 && ! -x .venv-curobo/bin/python ]]; then
  record WARN "venv .venv-curobo" "missing; cuRobo checks skipped (setup without --no-curobo builds it)"
  WITH_CUROBO=0
fi
if [[ -f third_party/robosuite/robosuite/macros_private.py ]]; then
  record PASS "robosuite /tmp log fix" "macros_private.py present"
else
  record WARN "robosuite /tmp log fix" "missing: 'import robosuite' fails if another user owns /tmp/robosuite.log (git pull; re-run setup)"
fi

# 2 ---------------------------------------------------------------------------------------------
section "2 gpu"
if ! command -v nvidia-smi >/dev/null; then
  record FAIL "nvidia driver" "nvidia-smi not found"
else
  record PASS "nvidia driver" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
  budget_args=(--gpu "$GPU"); ((WITH_CUROBO)) || budget_args+=(--no-curobo)
  if "$PY" scripts/gpu/check_gpu.py budget "${budget_args[@]}" > "$OUT/budget.txt" 2>&1; then
    record PASS "memory budget (GPU $GPU)" "$(grep -E '^OK' "$OUT/budget.txt")"
  else
    record WARN "memory budget (GPU $GPU)" "$(grep -E 'DOES NOT FIT|GPU [0-9]+ has' "$OUT/budget.txt" | tr '\n' ' ')"
  fi
fi

# 3 ---------------------------------------------------------------------------------------------
section "3 reserve"
reservation_mib() {
  "$PY" scripts/gpu/check_gpu.py measure --gpu "$GPU" --json "$OUT/measure.json" >/dev/null 2>&1
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("peak_by_role_mib", {}).get("reservation", 0))' "$OUT/measure.json" 2>/dev/null || echo 0
}
if command -v nvidia-smi >/dev/null; then
  export ASPIRE_TEST_STATE="$OUT/reserve-state"
  before="$(reservation_mib)"
  if ASPIRE_STATE_DIR="$ASPIRE_TEST_STATE" scripts/gpu/reserve.sh hold 1 --gpu "$GPU" > "$OUT/reserve.txt" 2>&1; then
    held="$(reservation_mib)"
    ASPIRE_STATE_DIR="$ASPIRE_TEST_STATE" scripts/gpu/reserve.sh release >> "$OUT/reserve.txt" 2>&1
    sleep 2
    after="$(reservation_mib)"
    if ((held - before >= 900 && after - before < 200)); then
      record PASS "hold / release 1 GiB" "held +$((held - before)) MiB, then freed"
    else
      record FAIL "hold / release 1 GiB" "before ${before} MiB, held ${held} MiB, after ${after} MiB (see $OUT/reserve.txt)"
    fi
  else
    record FAIL "hold / release 1 GiB" "$(tail -n 2 "$OUT/reserve.txt" | tr '\n' ' ')"
  fi
else
  record SKIP "hold / release 1 GiB" "no nvidia-smi"
fi

# 4 ---------------------------------------------------------------------------------------------
section "4 torch on GPU $GPU"
for venv in .venv-sam3 .venv-graspnet .venv-curobo .venv-libero; do
  [[ -x "$venv/bin/python" ]] || continue
  CUDA_VISIBLE_DEVICES="$GPU" "$venv/bin/python" scripts/gpu/checks/gpu_torch.py > "$OUT/torch-$venv.txt" 2>&1
  line="$(last_line "$OUT/torch-$venv.txt")"
  case "$line" in PASS*) record PASS "$venv" "${line#PASS }" ;; *) record FAIL "$venv" "$line" ;; esac
done

# 5 ---------------------------------------------------------------------------------------------
section "5 egl"
.venv-robosuite/bin/python scripts/gpu/checks/egl_render.py > "$OUT/egl.txt" 2>&1
line="$(last_line "$OUT/egl.txt")"
case "$line" in
  PASS*) record PASS "MuJoCo rendering" "${line#PASS }" ;;
  WARN*) record WARN "MuJoCo rendering" "${line#WARN }" ;;
  *) record FAIL "MuJoCo rendering" "$line (see $OUT/egl.txt)" ;;
esac

# 6 ---------------------------------------------------------------------------------------------
section "6 sam3 access"
.venv-sam3/bin/python scripts/gpu/checks/hf_access.py > "$OUT/hf.txt" 2>&1
line="$(last_line "$OUT/hf.txt")"
SAM3_STATUS="${line%% *}"
case "$SAM3_STATUS" in
  GRANTED) record PASS "facebook/sam3" "${line#GRANTED }" ;;
  PENDING|NO_ACCESS|NO_TOKEN) record BLOCKED "facebook/sam3" "${line#* }" ;;
  *) record FAIL "facebook/sam3" "$line" ;;
esac

# 7 ---------------------------------------------------------------------------------------------
section "7 bedrock / Opus 4.6"
if [[ "${CLAUDE_CODE_USE_BEDROCK:-}" != "1" || -z "${ANTHROPIC_MODEL:-}" ]]; then
  record FAIL "Bedrock settings" "run scripts/gpu/set_credentials.sh, then re-activate the env"
else
  record PASS "Bedrock settings" "model ${ANTHROPIC_MODEL}, region ${AWS_REGION:-?}"
  if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
    line="$("$PY" scripts/gpu/bedrock_key_info.py --min-minutes 120)"; rc=$?
    if ((rc == 0)); then record PASS "API key" "$line"
    elif ((rc == 2)); then record FAIL "API key" "$line (renew: scripts/gpu/set_credentials.sh --api-key)"
    else record WARN "API key" "$line"; fi
  fi
  if "$PY" scripts/gpu/checks/bedrock_sample.py > "$OUT/bedrock.txt" 2>&1; then
    record PASS "boto3 Converse + stream" "$(head -n 1 "$OUT/bedrock.txt")"
  else
    record FAIL "boto3 Converse + stream" "$(last_line "$OUT/bedrock.txt")"
  fi
  if command -v claude >/dev/null; then
    reply="$(timeout 180 claude -p "Reply with exactly the word OK." --max-turns 1 2>&1 | tail -n 1)"
    if [[ "$reply" == *OK* ]]; then record PASS "Claude Code -> Bedrock" "replied: $reply"
    else record FAIL "Claude Code -> Bedrock" "${reply:0:160}"; fi
  else
    record FAIL "Claude Code -> Bedrock" "claude not on PATH (setup installs it into the env)"
  fi
fi

if ((QUICK)); then
  section "8-11 skipped (--quick)"
else
  # 8 -------------------------------------------------------------------------------------------
  section "8 services"
  providers="pyroki,contact_graspnet"
  ((WITH_CUROBO)) && providers="$providers,curobo"
  [[ "$SAM3_STATUS" == GRANTED ]] && providers="sam3,$providers"
  if port_up 8116; then
    echo "  services already running; using them"
  else
    if scripts/gpu/services.sh start --gpu "$GPU" --providers "$providers" > "$OUT/services.txt" 2>&1; then
      STARTED_SERVICES=1
    else
      record FAIL "start services" "$(grep -m1 -E '^error' "$OUT/services.txt") (see $OUT/services.txt)"
    fi
  fi
  for item in pyroki:8116 contact_graspnet:8115 curobo:8118 sam3:8114; do
    name="${item%%:*}" port="${item##*:}"
    if port_up "$port"; then record PASS "service $name" "up on :$port"
    elif [[ "$name" == sam3 && "$SAM3_STATUS" != GRANTED ]]; then record BLOCKED "service $name" "waits for Hugging Face access"
    elif [[ "$name" == curobo && "$WITH_CUROBO" -eq 0 ]]; then record SKIP "service $name" "--no-curobo"
    else record FAIL "service $name" "not answering on :$port (scripts/gpu/services.sh logs $name)"; fi
  done

  # 9 -------------------------------------------------------------------------------------------
  section "9 simulators (one episode each)"
  run_episode() {  # run_episode LABEL VENV ARGS... ; sets RUN_DIR and SUMMARY
    local label="$1" venv="$2"; shift 2
    "$venv/bin/cap-harness" run "$@" --output-root "$OUT/runs" > "$OUT/$label.log" 2>&1
    SUMMARY="$(grep -E '^program_ok=' "$OUT/$label.log" | tail -n 1)"
    RUN_DIR="$(grep -E '^run: ' "$OUT/$label.log" | tail -n 1 | cut -c6-)"
  }
  run_episode robosuite-bimanual .venv-robosuite --benchmark robosuite --suite two_arm_lift \
    --task-id 0 --seed 1 --program examples/robosuite/bimanual_control_smoke.py \
    --max-steps 2000 --camera-width 256 --camera-height 256
  if [[ "$SUMMARY" == program_ok=True* ]]; then record PASS "Robosuite two-arm control" "$SUMMARY"
  else record FAIL "Robosuite two-arm control" "${SUMMARY:-no result} (see $OUT/robosuite-bimanual.log)"; fi
  if port_up 8116; then
    run_episode libero-microwave .venv-libero --benchmark libero-pro --suite libero_90 --task-id 33 \
      --seed 1 --program examples/libero/close_microwave.py --max-steps 2000 \
      --camera-width 256 --camera-height 256
    if [[ "$SUMMARY" == *"task_success=True"* ]]; then record PASS "LIBERO close microwave" "$SUMMARY"
    elif [[ "$SUMMARY" == program_ok=True* ]]; then record WARN "LIBERO close microwave" "$SUMMARY (ran; task not solved on this seed)"
    else record FAIL "LIBERO close microwave" "${SUMMARY:-no result} (see $OUT/libero-microwave.log)"; fi
  else
    record SKIP "LIBERO close microwave" "PyRoki not running"
  fi

  # 10 ------------------------------------------------------------------------------------------
  section "10 probe: every service through the robot API"
  run_episode stack-probe .venv-robosuite --benchmark robosuite --suite cube_lifting --task-id 0 \
    --seed 1 --program scripts/gpu/checks/stack_probe.py --max-steps 1500 \
    --camera-width 256 --camera-height 256
  if [[ -n "$RUN_DIR" && -f "$RUN_DIR/source/program-result.json" ]]; then
    "$PY" - "$RUN_DIR/source/program-result.json" "$SAM3_STATUS" "$WITH_CUROBO" > "$OUT/probe.tsv" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
sam3_status, with_curobo = sys.argv[2], sys.argv[3] == "1"
report = data.get("result") or {}
checks, errors = report.get("checks", {}), report.get("errors", {})
for name, ok in checks.items():
    if ok:
        status = "PASS"
    elif name.startswith("sam3") and sam3_status != "GRANTED":
        status = "BLOCKED"
    elif name.startswith(("curobo", "move_to_pose_curobo")) and not with_curobo:
        status = "SKIP"
    else:
        status = "FAIL"
    print(f"{status}\t{name}\t{errors.get(name, '')}")
for key, value in report.get("details", {}).items():
    print(f"INFO\t{key}\t{value}")
PY
    while IFS=$'\t' read -r status name detail; do
      if [[ "$status" == INFO ]]; then printf '          %-28s %s\n' "$name" "$detail"
      else record "$status" "probe: $name" "${detail:-ok}"; fi
    done < "$OUT/probe.tsv"
  else
    record FAIL "probe" "no program result (see $OUT/stack-probe.log)"
  fi

  # 11 ------------------------------------------------------------------------------------------
  if ((WITH_LLM)); then
    section "11 agent: Opus 4.6 writes, runs and checks a program"
    ws="$OUT/agent"; mkdir -p "$ws"
    "$PY" - scripts/gpu/prompts/agent_check.md "$ws/prompt.md" "$ws" <<'PY'
import sys
from pathlib import Path
template, out, ws = sys.argv[1:]
Path(out).write_text(Path(template).read_text().replace("{{WS}}", ws))
PY
    extra=()
    claude --help 2>/dev/null | grep -q -- "--max-budget-usd" && extra+=(--max-budget-usd 3)
    claude -p "$(cat "$ws/prompt.md")" --model "$ANTHROPIC_MODEL" --permission-mode bypassPermissions \
      --max-turns 30 "${extra[@]}" --output-format stream-json --verbose 2> "$ws/claude.stderr" \
      | "$PY" scripts/gpu/stream_view.py --raw "$ws/transcript.jsonl" --summary "$ws/result.json"
    program_result="$(ls -t "$ws"/run/robosuite/*/*/*/run-*/source/program-result.json 2>/dev/null | head -n 1)"
    if [[ -f "$ws/REPORT.md" && -n "$program_result" ]] \
      && "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") is True else 1)' "$program_result"; then
      record PASS "agent coding loop" "program ran; report: $ws/REPORT.md"
    else
      record FAIL "agent coding loop" "no successful run or no REPORT.md (see $ws/)"
    fi
  else
    section "11 agent skipped (add --with-llm; costs about \$1-3)"
  fi

  if ((STARTED_SERVICES && ! KEEP_SERVICES)); then
    scripts/gpu/services.sh stop > "$OUT/services-stop.txt" 2>&1
    echo; echo "  (stopped the services this check started; --keep-services keeps them)"
  fi
fi

# Summary --------------------------------------------------------------------------------------
section "Summary"
fails=0 blocked=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r status name detail <<< "$row"
  [[ "$status" == FAIL ]] && fails=$((fails + 1))
  [[ "$status" == BLOCKED ]] && blocked=$((blocked + 1))
done
printf '  %d checks: %d FAIL, %d BLOCKED\n' "${#ROWS[@]}" "$fails" "$blocked"
if ((fails)); then
  echo "  Fix these:"
  for row in "${ROWS[@]}"; do [[ "$row" == FAIL\|* ]] && echo "    - ${row#FAIL|}" | sed 's/|/: /'; done
fi
if ((blocked)); then
  echo "  Waiting on (outside this machine):"
  for row in "${ROWS[@]}"; do [[ "$row" == BLOCKED\|* ]] && echo "    - ${row#BLOCKED|}" | sed 's/|/: /'; done
fi
printf '%s\n' "${ROWS[@]}" > "$OUT/summary.txt"
echo "  Logs: $OUT"
((fails == 0))
