#!/usr/bin/env bash
# Light demos that show every part of ASPIRE working on a GPU workstation.
#
#   scripts/gpu/demo.sh smoke     [--with-bimanual]
#       No LLM. Doctor checks, then the shipped example programs end to end:
#       Robosuite cube lifting + LIBERO-Pro object swap (SAM3 + Contact-GraspNet + PyRoki),
#       and the cuRobo variant when cuRobo is running. Records peak GPU memory. ~10-20 min.
#
#   scripts/gpu/demo.sh llm       [--benchmark libero-pro|robosuite] [--suite S] [--task-id N]
#                                 [--dev-seeds "51 52 53"] [--heldout-seeds "1 2 3"]
#                                 [--max-versions 4] [--max-turns 120] [--budget-usd 20]
#       Claude Opus 4.6 (Bedrock, via Claude Code) writes, runs and debugs a program for one task
#       on a few development seeds, then evaluates it once on held-out seeds. ~20-60 min.
#
#   scripts/gpu/demo.sh campaign  [--tasks FILE] [--run]
#       Initializes a tiny ASPIRE iterative-debugging campaign (1 task, 3 dev / 1 unseen /
#       3 held-out seeds) and prints, or with --run starts, the interactive Claude Code
#       coordinator that exercises the whole pipeline (worker, validation, verifier, skills).
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_BASE="$REPO_ROOT/outputs/gpu-demo"
STAMP="$(date +%Y%m%d-%H%M%S)"
GPU="${ASPIRE_GPU:-0}"
PY="${CAP_HARNESS_PYTHON:-python3}"
cd "$REPO_ROOT"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
port_up() { curl --fail --silent --max-time 2 "http://127.0.0.1:$1/openapi.json" >/dev/null; }

require_services() {
  local missing=()
  port_up 8114 || missing+=("sam3:8114")
  port_up 8115 || missing+=("contact_graspnet:8115")
  port_up 8116 || missing+=("pyroki:8116")
  if ((${#missing[@]})); then
    echo "error: services down: ${missing[*]}. Start them with: scripts/gpu/services.sh start" >&2
    exit 1
  fi
  CUROBO_UP=0
  port_up 8118 && CUROBO_UP=1
  return 0
}

require_runtime_env() {
  if [[ -z "${MUJOCO_GL:-}" || -z "${LIBERO_CONFIG_PATH:-}" ]]; then
    echo "error: rendering variables are not set; run 'conda activate ${ASPIRE_CONDA_ENV:-aspire}'" >&2
    echo "       (scripts/gpu/setup_conda_env.sh stores them in the env)" >&2
    exit 1
  fi
}

render() {  # render TEMPLATE OUT KEY=VALUE...
  local template="$1" out="$2"; shift 2
  "$PY" - "$template" "$out" "$@" <<'PY'
import sys
from pathlib import Path

template, out, *pairs = sys.argv[1:]
text = Path(template).read_text(encoding="utf-8")
for pair in pairs:
    key, _, value = pair.partition("=")
    text = text.replace("{{" + key + "}}", value)
if "{{" in text:
    raise SystemExit(f"unfilled placeholder in {template}: {text[text.index('{{'):][:40]}")
Path(out).write_text(text, encoding="utf-8")
PY
}

# ------------------------------------------------------------------------------------- smoke --
run_episode() {  # run_episode LABEL VENV ARGS...
  local label="$1" venv="$2"; shift 2
  local log="$OUT/$label.log"
  printf '  %-28s ' "$label"
  local started=$SECONDS status=0
  "$REPO_ROOT/$venv/bin/cap-harness" run "$@" --output-root "$OUT/runs" >"$log" 2>&1 || status=$?
  local summary run_dir
  summary="$(grep -E '^program_ok=' "$log" | tail -n 1 || true)"
  run_dir="$(grep -E '^run: ' "$log" | tail -n 1 | cut -c6- || true)"
  printf '%4ss  %s\n' "$((SECONDS - started))" "${summary:-FAILED (exit $status, see $log)}"
  local providers=""
  [[ -n "$run_dir" ]] && providers="$("$PY" "$REPO_ROOT/scripts/gpu/provider_health.py" "$run_dir")"
  [[ -n "$providers" ]] && printf '  %-28s       %s\n' "" "$providers"
  printf '%s\t%s\t%s\t%s\n' "$label" "${summary:-exit=$status}" "${providers:-no trace}" \
    "${run_dir:-}" >> "$OUT/summary.tsv"
}

doctor_check() {  # doctor_check VENV RUNTIME PROVIDERS
  local venv="$1" runtime="$2" providers="$3" report="$OUT/doctor-$2.json"
  "$REPO_ROOT/$venv/bin/cap-harness" doctor --environment-root "$REPO_ROOT/$venv" \
    --expected-environment-name "$venv" --runtime "$runtime" --providers "$providers" \
    --quiet --output "$report" >/dev/null 2>&1 || true
  "$PY" - "$report" "$runtime" <<'PY'
import json, sys
path, runtime = sys.argv[1:]
try:
    report = json.load(open(path))
except (OSError, ValueError):
    print(f"  doctor ({runtime}): no report written; run it by hand to see why")
    raise SystemExit(0)
bad = [c for c in report["checks"] if c["status"] == "fail"]
status = "PASS" if report.get("success") else "FAIL"
print(f"  doctor ({runtime}): {status}, {len(report['checks'])} checks"
      + ("" if not bad else "; failing: " + ", ".join(c["name"] for c in bad)))
PY
}

cmd_smoke() {
  local bimanual=0
  while (($#)); do
    case "$1" in
      --with-bimanual) bimanual=1; shift ;;
      *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  require_runtime_env
  require_services
  OUT="$OUT_BASE/smoke-$STAMP"
  mkdir -p "$OUT"
  local providers="sam3,contact_graspnet,pyroki"
  ((CUROBO_UP)) && providers="$providers,curobo"

  step "Doctor checks (environment, GPU, EGL, services)"
  doctor_check .venv-robosuite robosuite "$providers"
  doctor_check .venv-libero libero "$providers"

  step "Example programs (each line is one full episode; logs in $OUT)"
  local stop_file="$OUT/.stop-gpu-watch"
  "$PY" scripts/gpu/check_gpu.py measure --gpu "$GPU" --watch 86400 --interval 2 \
    --stop-file "$stop_file" --json "$OUT/gpu-peak.json" > "$OUT/gpu-peak.txt" 2>&1 &
  local watcher=$!
  : > "$OUT/summary.tsv"
  run_episode robosuite-cube-lift .venv-robosuite --benchmark robosuite --suite cube_lifting \
    --task-id 0 --seed 1 --program examples/robosuite/cube_lifting_seed1.py \
    --max-steps 3000 --camera-width 512 --camera-height 512
  run_episode libero-object-swap .venv-libero --benchmark libero-pro --suite libero_object_swap \
    --task-id 0 --seed 1 --program examples/libero/object_swap_seed1.py \
    --max-steps 2000 --camera-width 800 --camera-height 512
  if ((CUROBO_UP)); then
    run_episode robosuite-cube-lift-curobo .venv-robosuite --benchmark robosuite \
      --suite cube_lifting --task-id 0 --seed 1 \
      --program examples/robosuite/cube_lifting_curobo_seed1.py \
      --max-steps 3000 --camera-width 512 --camera-height 512
  fi
  if ((bimanual)); then
    run_episode robosuite-two-arm-lift .venv-robosuite --benchmark robosuite \
      --suite two_arm_lift --task-id 0 --seed 1 --program examples/robosuite/two_arm_lift_seed1.py \
      --max-steps 3000 --camera-width 512 --camera-height 512
  fi
  touch "$stop_file"
  wait "$watcher" || true

  step "GPU memory used by your ASPIRE processes during the demo"
  cat "$OUT/gpu-peak.txt"

  step "Verdict"
  # Every episode must finish its program and reach the services without infrastructure errors.
  # task_success is reported but not required: the examples succeed on most seeds, not all.
  local verdict=PASS
  awk -F'\t' '$2 !~ /^program_ok=True/ {bad = 1} END {exit bad}' "$OUT/summary.tsv" \
    || { verdict=FAIL; echo "- an episode did not finish its program (program_ok != True)"; }
  if grep -q "SERVICE ERRORS" "$OUT/summary.tsv"; then
    verdict=FAIL; echo "- a model service failed during an episode (see the services lines above)"
  fi
  if ! grep -Eq 'sam3 [0-9]+ calls \([1-9][0-9]* ok\)' "$OUT/summary.tsv"; then
    verdict=FAIL; echo "- SAM3 never returned a segmentation"
  fi
  echo "SMOKE $verdict"
  echo "Results: $OUT/summary.tsv"
  echo "Each run directory holds outcome.json, trace/, media/videos/ and media/keyframes/."
  [[ "$verdict" == PASS ]]
}

# --------------------------------------------------------------------------------------- llm --
set_claude_flags() {  # sets CLAUDE_FLAGS, shared by both LLM demos
  if [[ -z "${ANTHROPIC_MODEL:-}" || "${CLAUDE_CODE_USE_BEDROCK:-}" != "1" ]]; then
    echo "error: Bedrock is not configured; run scripts/gpu/set_credentials.sh, then" >&2
    echo "       'conda deactivate && conda activate ${ASPIRE_CONDA_ENV:-aspire}'" >&2
    exit 1
  fi
  command -v claude >/dev/null || { echo "error: claude CLI not found" >&2; exit 1; }
  # bypassPermissions: the demo runs unattended and launches simulator processes itself.
  CLAUDE_FLAGS=(--model "$ANTHROPIC_MODEL" --permission-mode bypassPermissions)
}

cmd_llm() {
  local benchmark="libero-pro" suite="" task_id=0 dev="" heldout="1 2 3"
  local max_versions=4 max_turns=120 budget=20 max_steps="" init_mode=""
  while (($#)); do
    case "$1" in
      --benchmark) benchmark="$2"; shift 2 ;;
      --suite) suite="$2"; shift 2 ;;
      --task-id) task_id="$2"; shift 2 ;;
      --dev-seeds) dev="$2"; shift 2 ;;
      --heldout-seeds) heldout="$2"; shift 2 ;;
      --max-versions) max_versions="$2"; shift 2 ;;
      --max-turns) max_turns="$2"; shift 2 ;;
      --budget-usd) budget="$2"; shift 2 ;;
      *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  case "$benchmark" in
    libero-pro)
      suite="${suite:-libero_object_swap}"; dev="${dev:-51 52 53}"
      max_steps=2000; init_mode=seeded; venv=.venv-libero ;;
    robosuite)
      suite="${suite:-cube_lifting}"; dev="${dev:-101 102 103}"
      max_steps=3000; init_mode=saved; venv=.venv-robosuite ;;
    *) echo "error: --benchmark must be libero-pro or robosuite" >&2; exit 2 ;;
  esac
  require_runtime_env
  require_services
  set_claude_flags
  local ws="$OUT_BASE/llm-$benchmark-$suite-$task_id-$STAMP"
  mkdir -p "$ws"
  local curobo_note=""
  ((CUROBO_UP)) && curobo_note=", cuRobo :8118 (first plan ~50 s while it warms up)"
  render scripts/gpu/prompts/llm_quick_demo.md "$ws/prompt.md" \
    "BENCHMARK=$benchmark" "SUITE=$suite" "TASK_ID=$task_id" "WS=$ws" \
    "CAP_HARNESS=$venv/bin/cap-harness" "DEV_SEEDS=$dev" "HELDOUT_SEEDS=$heldout" \
    "INIT_MODE=$init_mode" "MAX_STEPS=$max_steps" "MAX_VERSIONS=$max_versions" \
    "CUROBO_NOTE=$curobo_note"
  local extra=(--max-turns "$max_turns")
  if claude --help 2>/dev/null | grep -q -- "--max-budget-usd"; then
    extra+=(--max-budget-usd "$budget")
  fi

  step "Claude Code ($ANTHROPIC_MODEL) on $benchmark/$suite task $task_id"
  echo "workspace: $ws"
  echo "dev seeds: $dev | held-out seeds: $heldout | max versions: $max_versions | budget: \$$budget"
  echo "full transcript: $ws/transcript.jsonl  (Ctrl-C stops the agent)"
  echo
  local status=0
  claude -p "$(cat "$ws/prompt.md")" "${CLAUDE_FLAGS[@]}" "${extra[@]}" \
    --output-format stream-json --verbose 2> "$ws/claude.stderr" \
    | "$PY" scripts/gpu/stream_view.py --raw "$ws/transcript.jsonl" --summary "$ws/result.json" \
    || status=$?

  step "Agent report"
  if [[ -f "$ws/REPORT.md" ]]; then cat "$ws/REPORT.md"; else echo "(no REPORT.md written)"; fi
  echo
  for batch in "$ws"/dev_v*/batch.json "$ws"/heldout/batch.json; do
    [[ -f "$batch" ]] || continue
    "$PY" - "$batch" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
print(f"  {path.split('/')[-2]:<10} task_success {data.get('passes', '?')}/{data.get('seeds', '?')}"
      + (f"  (no run for seeds {data['missing']})" if data.get("missing") else ""))
PY
  done
  [[ -s "$ws/claude.stderr" ]] && { echo "claude stderr (tail):"; tail -n 5 "$ws/claude.stderr"; }
  return "$status"
}

# ---------------------------------------------------------------------------------- campaign --
cmd_campaign() {
  local tasks="" run=0
  while (($#)); do
    case "$1" in
      --tasks) tasks="$2"; shift 2 ;;
      --run) run=1; shift ;;
      *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  require_runtime_env
  require_services
  set_claude_flags
  if [[ -z "$tasks" ]]; then
    tasks="$OUT_BASE/campaign-tasks-$STAMP.json"
    mkdir -p "$OUT_BASE"
    cat > "$tasks" <<'JSON'
[
  {"benchmark": "libero-pro", "suite": "libero_object_swap", "task_id": 0,
   "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket"}
]
JSON
  fi
  step "Initializing a mini campaign from $tasks"
  "$PY" .claude/skills/iterative-debugging/scripts/init_run.py --tasks "$tasks" --name gpu-demo \
    --dev-seeds 51 52 53 --unseen-seeds 66 --heldout-count 3
  local run_root
  run_root="$(cd outputs/runs/LATEST && pwd -P)"
  local curobo_note=""
  ((CUROBO_UP)) && curobo_note=", cuRobo :8118"
  render scripts/gpu/prompts/mini_campaign.md "$run_root/kickoff_prompt.md" \
    "RUN_ROOT=$run_root" "DEV_SEEDS=51-53" "UNSEEN_SEEDS=66" "HELDOUT_COUNT=3" \
    "GPU=$GPU" "CUROBO_NOTE=$curobo_note"
  echo "kickoff prompt: $run_root/kickoff_prompt.md"
  if ((run)); then
    step "Starting the interactive coordinator (Claude Code, ${ANTHROPIC_MODEL})"
    exec claude "${CLAUDE_FLAGS[@]}" "$(cat "$run_root/kickoff_prompt.md")"
  fi
  cat <<EOF

Start the coordinator in an interactive Claude Code session (it runs for about an hour, and you
can watch and steer it):

  cd $REPO_ROOT
  claude ${CLAUDE_FLAGS[*]} "\$(cat $run_root/kickoff_prompt.md)"

or simply: scripts/gpu/demo.sh campaign --run
Progress at any time: $PY .claude/skills/iterative-debugging/scripts/gen_progress.py --print
EOF
}

case "${1:-}" in
  smoke) shift; cmd_smoke "$@" ;;
  llm) shift; cmd_llm "$@" ;;
  campaign) shift; cmd_campaign "$@" ;;
  -h|--help|"") usage ;;
  *) echo "error: unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
