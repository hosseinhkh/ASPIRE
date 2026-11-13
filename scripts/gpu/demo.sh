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
#                                 [--out DIR] [--write-skills] [--layouts all|S1,S2,...]
#       Claude Opus 4.6 (Bedrock, via Claude Code) writes, runs and debugs a program for one task
#       on a few development seeds, then evaluates it once on held-out seeds. ~20-60 min.
#       --start-from PROGRAM: repair an existing program instead of writing one: it is run
#       first as version 1, then fixed where it fails without breaking what it solves.
#       --layouts: one program for several layouts of the task (all: the suite and its sibling
#       layouts); every batch runs each layout (development 51 52 53 and held-out 1 2, each layout).
#
#   scripts/gpu/demo.sh campaign  [--tasks FILE | --suite SUITE] [--run | --headless]
#                                 [--budget-usd 30] [--max-turns 400] [--dev-seeds "51 52 53"]
#                                 [--unseen-seeds "66"] [--heldout-count 3] [--paper]
#                                 [--memory on|off] [--library current|empty|released|NAME]
#                                 [--harness fixed|published] [--name NAME] [--resume RUN_ROOT]
#       Initializes an ASPIRE iterative-debugging campaign (default: 1 task, 3 dev / 1 unseen /
#       3 held-out seeds) and prints, with --run starts, the interactive Claude Code
#       coordinator that runs the whole pipeline (worker, validation, verifier, skills).
#       --headless runs it unattended under a hard dollar cap (Claude Code's --max-budget-usd
#       works only headless) and records the transcript in the campaign folder.
#       --paper uses the paper's protocol: LIBERO dev 51-65 / unseen 66-80 / held-out 1-50
#       (Robosuite 101-125 / 126-130 / 1-100), no cap on program versions, $40 per task,
#       and by default the published harness behaviour (--harness fixed to keep the fixes).
#       --suite runs every task of a suite. --memory off is a no-memory baseline: no skill is
#       promoted and the library is restored afterwards. --library picks the starting skill
#       library (scripts/gpu/library.sh). The library is saved as <campaign>-start and -end.
#       --resume continues an interrupted campaign with its own settings.
#
#   Claude runs with the paper's Opus 4.6 settings: max effort and the 1M-token context.
#   ASPIRE_EFFORT=high (Claude Code's default) or ASPIRE_1M=0 change them.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_BASE="$REPO_ROOT/outputs/gpu-demo"
STAMP="$(date +%Y%m%d-%H%M%S)"
GPU="${ASPIRE_GPU:-0}"
PY="${CAP_HARNESS_PYTHON:-python3}"
cd "$REPO_ROOT"

usage() { sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
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
set_claude_flags() {  # set_claude_flags MINUTES_NEEDED; sets CLAUDE_FLAGS for both LLM demos
  local minutes="${1:-60}"
  if [[ -z "${ANTHROPIC_MODEL:-}" || "${CLAUDE_CODE_USE_BEDROCK:-}" != "1" ]]; then
    echo "error: Bedrock is not configured; run scripts/gpu/set_credentials.sh, then" >&2
    echo "       'conda deactivate && conda activate ${ASPIRE_CONDA_ENV:-aspire}'" >&2
    exit 1
  fi
  if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
    # Short-term Bedrock API keys last at most 12 h; a run that outlives one dies midway.
    local status=0
    "$PY" "$REPO_ROOT/scripts/gpu/bedrock_key_info.py" --min-minutes "$minutes" || status=$?
    if ((status == 2)); then
      echo "error: renew the key first: scripts/gpu/set_credentials.sh --api-key" >&2
      echo "       then: conda deactivate && conda activate ${ASPIRE_CONDA_ENV:-aspire}" >&2
      exit 1
    fi
  fi
  command -v claude >/dev/null || { echo "error: claude CLI not found" >&2; exit 1; }
  # The paper ran Opus 4.6 at max effort with the 1M-token context (arXiv 2607.00272, Tables
  # 19-20). Without these Claude Code uses high effort and a 200k window.
  local model="$ANTHROPIC_MODEL" help
  if [[ "${ASPIRE_1M:-1}" == 1 && "$model" != *"[1m]" ]]; then model="$model[1m]"; fi
  export CLAUDE_CODE_EFFORT_LEVEL="${ASPIRE_EFFORT:-max}"
  # The skill library is the only memory an experiment should have: no Claude Code auto
  # memory, no CLAUDE.md notes carried over from other sessions.
  export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
  # A batch of seeds can run for tens of minutes on a busy machine. Claude Code's Bash tool
  # stops a command at 10 minutes unless these allow more, and a batch that is stopped loses
  # its unfinished seeds.
  export BASH_DEFAULT_TIMEOUT_MS="${BASH_DEFAULT_TIMEOUT_MS:-1800000}"
  export BASH_MAX_TIMEOUT_MS="${BASH_MAX_TIMEOUT_MS:-3600000}"
  # Sub-agents ask for "opus": give them the same model and window.
  export ANTHROPIC_DEFAULT_OPUS_MODEL="$model"
  # bypassPermissions: the demo runs unattended and launches simulator processes itself.
  CLAUDE_FLAGS=(--model "$model" --permission-mode bypassPermissions)
  help="$(claude --help 2>/dev/null || true)"
  [[ "$help" == *--effort* ]] && CLAUDE_FLAGS+=(--effort "$CLAUDE_CODE_EFFORT_LEVEL")
  echo "claude: $model, effort $CLAUDE_CODE_EFFORT_LEVEL"
}

model_usage() {  # model_usage RESULT_JSON: the window, thinking and cost each model really used
  "$PY" - "$1" <<'PY'
import json, sys
try:
    usage = json.load(open(sys.argv[1])).get("modelUsage") or {}
except (OSError, ValueError):
    raise SystemExit(0)
for model, u in usage.items():
    window = u.get("contextWindow")
    head = f"  {model}: context window {window:,} tokens" if isinstance(window, int) else f"  {model}:"
    print(head, f"| thinking {u.get('thinkingTokens') or 0:,} tokens | ${u.get('costUSD') or 0:.2f}")
PY
}

cmd_llm() {
  local benchmark="libero-pro" suite="" task_id=0 dev="" heldout=""
  local max_versions=4 max_turns=120 budget=20 init_mode="" out="" write_skills=0 layouts=""
  local start_from=""
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
      --out) out="$2"; shift 2 ;;
      --write-skills) write_skills=1; shift ;;
      --layouts) layouts="$2"; shift 2 ;;
      --start-from) start_from="$2"; shift 2 ;;
      *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  case "$benchmark" in
    libero-pro)
      suite="${suite:-libero_object_swap}"; dev="${dev:-51 52 53}"
      init_mode=seeded; venv=.venv-libero ;;
    robosuite)
      suite="${suite:-cube_lifting}"; dev="${dev:-101 102 103}"
      init_mode=saved; venv=.venv-robosuite ;;
    *) echo "error: --benchmark must be libero-pro or robosuite" >&2; exit 2 ;;
  esac
  # --layouts: the program must solve the task in several layouts (LIBERO-Pro's suite and its
  # sibling layouts); every batch runs them all (scripts/gpu/layouts_batch.py).
  if [[ -n "$layouts" ]]; then
    [[ "$benchmark" == libero-pro ]] || { echo "error: --layouts is for libero-pro" >&2; exit 2; }
    if [[ "$layouts" == all ]]; then
      layouts="$("$REPO_ROOT/.venv-libero/bin/python" - "$suite" <<'PY'
import sys
from cap_harness.libero import layouts
print(",".join([sys.argv[1], *sorted(n for n, s in layouts.suites().items() if s.get("base") == sys.argv[1])]))
PY
)"
    fi
    [[ "$layouts" == *,* ]] || { echo "error: no sibling layouts of $suite (--layouts LIST)" >&2; exit 2; }
    # Every batch runs each layout: two held-out scenes per layout keep the last batch short
    # (the development seeds stay 51 52 53, so 15 episodes per version with five layouts).
    heldout="${heldout:-1 2}"
  fi
  heldout="${heldout:-1 2 3}"
  local batch layouts_note=""
  if [[ -n "$layouts" ]]; then
    batch="$venv/bin/python scripts/gpu/layouts_batch.py --layouts $layouts --task-id $task_id --init-mode $init_mode --workers 2"
    local count list
    count="$(tr ',' '\n' <<< "$layouts" | wc -l | tr -d ' ')"
    list="$(sed 's/,/`, `/g' <<< "$layouts")"
    layouts_note="$(cat <<'NOTE'

This task is solved in @COUNT@ layouts, and one program must solve all of them: the target object
stands in a different place in each (the same objects, the same instruction). The layouts are the
suites `@LIST@`; the first is the official one. Every development and held-out batch below runs
each layout and adds them up in one `batch.json`: each layout's own runs and `batch.json` are in
`<output-root>/<layout>/`, and the combined file's `layouts` section gives each layout's result.
Find everything from what the program observes; never hard-code a position or an approach
direction that works in one layout only. When a version fails in some layouts, open the
keyframes of a failing and a passing layout side by side before changing anything. To look at
another layout in step 2, run the probe with `--suite` set to that layout (it counts toward the 4
probe runs).
NOTE
)"
    layouts_note="${layouts_note//@COUNT@/$count}"
    layouts_note="${layouts_note//@LIST@/$list}
"
  else
    batch="$venv/bin/cap-harness run-batch --benchmark $benchmark --suite $suite --task-id $task_id --init-mode $init_mode --workers 1"
  fi
  if [[ -n "$start_from" ]]; then
    [[ -f "$start_from" ]] || { echo "error: --start-from: no such program: $start_from" >&2; exit 2; }
    start_from="$(cd "$(dirname "$start_from")" && pwd)/$(basename "$start_from")"
  fi
  require_runtime_env
  require_services
  set_claude_flags 75
  local ws="${out:-$OUT_BASE/llm-$benchmark-$suite-$task_id-$STAMP}"
  mkdir -p "$ws"
  ws="$(cd "$ws" && pwd)"
  # --start-from: the earlier program is version 1; the agent repairs it.
  local start_note=""
  if [[ -n "$start_from" ]]; then
    cp "$start_from" "$ws/program_v01.py"
    cp "$start_from" "$ws/program_start.py"
    start_note="$(cat <<'NOTE'

Repair, do not rewrite. An earlier program for this task is already in `@WS@/program_v01.py`
(a copy is in `program_start.py`; it came from `@FROM@`). It solves some scenes and fails others.
Do not write a new program from scratch: in step 3, skip writing program_v01.py and run the
existing one as version 1 (same command, output `dev_v01`). Read its results, then fix what
fails in the next versions while keeping everything it solves solved: each version must still
pass the scenes the previous one passed. Step 2's probe is optional; use it only on a scene that
fails.
NOTE
)"
    start_note="${start_note//@WS@/$ws}"
    start_note="${start_note//@FROM@/$start_from}
"
  fi
  local curobo_note=""
  ((CUROBO_UP)) && curobo_note=", cuRobo :8118 (first plan ~50 s while it warms up)"
  local instruction="" task_note
  if [[ "$benchmark" == libero-pro ]]; then
    # Resolved offline from the task's BDDL file (about a second, no simulator).
    instruction="$("$REPO_ROOT/.venv-libero/bin/python" scripts/gpu/checks/task_instruction.py \
      "$suite" "$task_id" 2>/dev/null | sed -n 's/^INSTRUCTION=//p')"
  fi
  if [[ -n "$instruction" ]]; then
    task_note="The task instruction is: \"$instruction\"."
    echo "instruction: $instruction"
  else
    task_note="Read the instruction from \`get_task_context().language\` (or from a first run) before writing the program."
  fi
  # --write-skills: the agent writes its lessons into a private copy of the skill library in
  # its workspace (the shared library is never touched), so you can see how it writes them.
  local skills_dir=.claude/skills/iterative-debugging/skills skills_step="" lessons_line
  lessons_line="one to three reusable lessons, written as candidate skill entries in the style of \`$skills_dir/*.md\`. Do not edit those skill files."
  if ((write_skills)); then
    mkdir -p "$ws/skills" "$ws/.skills-before"
    cp "$skills_dir"/*.md "$ws/skills/"
    cp "$skills_dir"/*.md "$ws/.skills-before/"
    skills_dir="$ws/skills"
    lessons_line="the lessons you added to the skill library in step 8."
    skills_step="
8. Write what you learned into the skill library at \`$ws/skills/\`. It is a private copy for
   this run: the shared library in \`.claude/skills/\` is not touched, and you must not edit it.
   Add each lesson to the file it belongs in (grasp.md, localize.md, transport.md or
   manipulation.md), following that file's header: a new subsection with the trigger (when it
   applies), the code, and the evidence (task, seeds, the versions with their development and
   held-out results, file paths). Write only lessons your results support, and say plainly when
   one is not confirmed. Keep each entry general enough that an agent on a different task can
   use it.
"
  fi
  render scripts/gpu/prompts/llm_quick_demo.md "$ws/prompt.md" \
    "BENCHMARK=$benchmark" "SUITE=$suite" "TASK_ID=$task_id" "WS=$ws" "TASK_NOTE=$task_note" \
    "CAP_HARNESS=$venv/bin/cap-harness" "PYTHON=$venv/bin/python" "DEV_SEEDS=$dev" "HELDOUT_SEEDS=$heldout" \
    "INIT_MODE=$init_mode" "MAX_VERSIONS=$max_versions" "FIRST_DEV_SEED=${dev%% *}" \
    "CUROBO_NOTE=$curobo_note" "SKILLS_DIR=$skills_dir" "LESSONS_LINE=$lessons_line" \
    "SKILLS_STEP=$skills_step" "BATCH=$batch" "LAYOUTS_NOTE=$layouts_note" "START_NOTE=$start_note"
  local extra=(--max-turns "$max_turns")
  # Capture first: `| grep -q` under pipefail fails when grep exits before the writer.
  if [[ "$(claude --help 2>/dev/null)" == *--max-budget-usd* ]]; then
    extra+=(--max-budget-usd "$budget")
  fi

  step "Claude Code ($ANTHROPIC_MODEL) on $benchmark/$suite task $task_id"
  echo "workspace: $ws"
  echo "dev seeds: $dev | held-out seeds: $heldout | max versions: $max_versions | budget: \$$budget"
  [[ -z "$layouts" ]] || echo "layouts (every batch runs each): ${layouts//,/, }"
  [[ -z "$start_from" ]] || echo "repairing: $start_from (its copy is program_v01.py)"
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
  model_usage "$ws/result.json"
  if ((write_skills)); then
    diff -ru "$ws/.skills-before" "$ws/skills" > "$ws/skills.diff" || true
    step "Skills the agent wrote (private copy: $ws/skills; diff: $ws/skills.diff)"
    if [[ -s "$ws/skills.diff" ]]; then
      echo "$(grep -c '^+[^+]' "$ws/skills.diff" || true) non-empty lines added in: $(grep '^+++ ' "$ws/skills.diff" | awk '{print $2}' | xargs -n1 basename | tr '\n' ' ')"
      sed -n '1,200p' "$ws/skills.diff"
    else
      echo "(no skill written)"
    fi
  fi
  for batch in "$ws"/dev_v*/batch.json "$ws"/heldout/batch.json; do
    [[ -f "$batch" ]] || continue
    "$PY" - "$batch" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
print(f"  {path.split('/')[-2]:<10} task_success {data.get('passes', '?')}/{data.get('seeds', '?')}"
      + (f"  (no run for seeds {data['missing']})" if data.get("missing") else ""))
for warning in data.get("warnings") or []:
    print(f"             warning (seeds {' '.join(str(s) for s in warning['seeds'])}): {warning['text']}")
PY
  done
  [[ -s "$ws/claude.stderr" ]] && { echo "claude stderr (tail):"; tail -n 5 "$ws/claude.stderr"; }
  return "$status"
}

# ---------------------------------------------------------------------------------- campaign --
campaign_meta() {  # campaign_meta RUN_ROOT KEY=VALUE...: merge fields into the campaign's session.json
  "$PY" - "$@" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "session.json"
try:
    data = json.loads(path.read_text())
except (OSError, ValueError):
    data = {}
for pair in sys.argv[2:]:
    key, _, value = pair.partition("=")
    if key.startswith("?"):  # only when not already set (a resumed campaign keeps its start)
        data.setdefault(key[1:], value)
    else:
        data[key] = value
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

campaign_headless() {  # campaign_headless RUN_ROOT BUDGET_USD MAX_TURNS MEMORY
  local run_root="$1" budget="$2" max_turns="$3" memory="$4" status=0 name
  name="$(basename "$run_root")"
  if [[ "$(claude --help 2>/dev/null)" != *--max-budget-usd* ]]; then
    echo "error: this Claude Code has no --max-budget-usd; update it: npm install -g @anthropic-ai/claude-code" >&2
    exit 1
  fi
  # session.json lets the dashboard name the run, and claude-logs pick out its sub-agents.
  campaign_meta "$run_root" "?name=campaign $name" "?started=$(date -Iseconds)" "budget_usd=$budget" \
    "model=$ANTHROPIC_MODEL" "claude_effort=$CLAUDE_CODE_EFFORT_LEVEL" \
    "harness=${CAP_HARNESS_COMPAT:-fixed}" "memory=$memory" \
    "git_commit=$(git rev-parse --short HEAD 2>/dev/null)"
  step "Headless ASPIRE campaign (Claude Code, $ANTHROPIC_MODEL), memory $memory, cap \$$budget"
  echo "campaign: $run_root"
  echo "transcript: $run_root/transcript.jsonl   (Ctrl-C stops it; progress: gen_progress.py --print)"
  claude -p "$(cat "$run_root/kickoff_prompt.md")" "${CLAUDE_FLAGS[@]}" \
    --max-turns "$max_turns" --max-budget-usd "$budget" --output-format stream-json --verbose \
    2>> "$run_root/claude.stderr" \
    | "$PY" scripts/gpu/stream_view.py --raw "$run_root/transcript.jsonl" --summary "$run_root/result.json" \
    || status=$?
  scripts/gpu/session.sh claude-logs "$run_root" || true  # the sub-agents' own transcripts
  campaign_library_end "$run_root" "$memory"
  step "Campaign summary"
  if [[ -f "$run_root/DEMO_SUMMARY.md" ]]; then cat "$run_root/DEMO_SUMMARY.md"
  else echo "(no DEMO_SUMMARY.md: the run stopped early; see result.json and the transcript)"; fi
  "$PY" - "$run_root/result.json" "$budget" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    raise SystemExit("no result.json written")
print(f"\nended: {r.get('subtype')} | turns {r.get('num_turns')} | "
      f"{(r.get('duration_ms') or 0) / 60000:.1f} min | cost ${r.get('total_cost_usd') or 0:.2f} "
      f"of ${float(sys.argv[2]):.2f} cap")
PY
  model_usage "$run_root/result.json"
  "$PY" scripts/gpu/campaign_results.py "$run_root" || true
  echo "Unfinished? Continue it: scripts/gpu/demo.sh campaign --resume $run_root --headless --budget-usd N"
  echo "Browse it: python scripts/gpu/dashboard.py  (Runs -> campaign $name)"
  return "$status"
}

campaign_library_end() {  # campaign_library_end RUN_ROOT MEMORY: keep (memory on) or undo (off) changes
  local run_root="$1" memory="$2" name start
  name="$(basename "$run_root")"
  start="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("library_start_snapshot", ""))' \
    "$run_root/session.json" 2>/dev/null || true)"
  scripts/gpu/library.sh save "$name-end" --force
  # Also under the plain campaign name, so the next run can start from it: --library NAME-end
  local plain
  plain="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("campaign_name", ""))' \
    "$run_root/session.json" 2>/dev/null || true)"
  [[ -n "$plain" ]] && scripts/gpu/library.sh save "$plain-end" --force
  if [[ "$memory" == off && -n "$start" ]]; then
    if ! diff -rq "outputs/libraries/$start" "outputs/libraries/$name-end" >/dev/null; then
      echo "warning: the library changed although memory was off; restoring $start" >&2
      scripts/gpu/library.sh load "$start"
    fi
  fi
  campaign_meta "$run_root" "library_end_words=$(cat "outputs/libraries/$name-end"/*.md | wc -w | tr -d ' ')" \
    "library_end_snapshot=$name-end" "finished=$(date -Iseconds)"
}

cmd_campaign() {
  local tasks="" suite="" run=0 headless=0 budget="" max_turns="" dev="51 52 53" unseen="66" heldout=3
  local paper=0 memory=on library=current harness="" name=gpu-demo resume=""
  while (($#)); do
    case "$1" in
      --tasks) tasks="$2"; shift 2 ;;
      --suite) suite="$2"; shift 2 ;;
      --run) run=1; shift ;;
      --headless) headless=1; shift ;;
      --paper) paper=1; shift ;;
      --memory) memory="$2"; shift 2 ;;
      --library) library="$2"; shift 2 ;;
      --harness) harness="$2"; shift 2 ;;
      --name) name="$2"; shift 2 ;;
      --resume) resume="$2"; shift 2 ;;
      --budget-usd) budget="$2"; shift 2 ;;
      --max-turns) max_turns="$2"; shift 2 ;;
      --dev-seeds) dev="$2"; shift 2 ;;
      --unseen-seeds) unseen="$2"; shift 2 ;;
      --heldout-count) heldout="$2"; shift 2 ;;
      *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
  done
  [[ "$memory" == on || "$memory" == off ]] || { echo "error: --memory on|off" >&2; exit 2; }
  require_runtime_env
  require_services
  set_claude_flags 120
  local scope version_rule memory_rule resume_note="" kind="" count=1 label=mini run_root
  if [[ -n "$resume" ]]; then
    run_root="$(cd "$resume" && pwd -P)"
    # shellcheck disable=SC1091  # written by this script when the campaign was initialized
    source "$run_root/campaign_settings.sh"
    # Keep the earlier part's transcript and cost; campaign_results.py adds the parts up.
    local part=1
    while [[ -e "$run_root/result.part$part.json" ]]; do part=$((part + 1)); done
    [[ -f "$run_root/result.json" ]] && mv "$run_root/result.json" "$run_root/result.part$part.json"
    [[ -f "$run_root/transcript.jsonl" ]] && mv "$run_root/transcript.jsonl" "$run_root/transcript.part$part.jsonl"
    resume_note="
This campaign was interrupted and is being resumed. Regenerate progress with gen_progress.py,
rebuild your ledger from disk as main-agent-prompt.md describes, and continue every unfinished
task. Do not redo or overwrite finished stages, and keep the existing campaign settings.
"
    step "Resuming the $label campaign $run_root (memory $memory)"
  else
    if [[ -n "$suite" ]]; then
      local bench=robosuite venv=.venv-robosuite
      [[ "$suite" == libero* ]] && { bench=libero-pro; venv=.venv-libero; }
      tasks="$OUT_BASE/campaign-tasks-$suite-$STAMP.json"
      mkdir -p "$OUT_BASE"
      "$REPO_ROOT/$venv/bin/python" scripts/gpu/checks/suite_tasks.py "$bench" "$suite" "$tasks"
      [[ "$name" == gpu-demo ]] && name="$suite"
    fi
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
    scope="Run a deliberately small ASPIRE iterative-debugging campaign end to end, as a smoke test of the whole pipeline: coordinator, worker sub-agent, unseen-check, Stage-2 held-out validation, verifier, and skill promotion. Keep it small; do not expand the scope."
    version_rule="Budget: at most 4 program versions per worker."
    if ((paper)); then
      # The paper's seed partitions (iterative-debugging/SKILL.md; arXiv 2607.00272 Sec. 3).
      read -r kind count < <("$PY" scripts/gpu/checks/task_list_kind.py "$tasks")
      case "$kind" in
        libero-pro) dev="$(seq -s ' ' 51 65)"; unseen="$(seq -s ' ' 66 80)"; heldout=50 ;;
        robosuite) dev="$(seq -s ' ' 101 125)"; unseen="$(seq -s ' ' 126 130)"; heldout=100 ;;
        *) echo "error: --paper needs tasks from a single benchmark (got: $kind)" >&2; exit 2 ;;
      esac
      budget="${budget:-$((40 * count))}"; max_turns="${max_turns:-$((600 * count))}"
      label=paper-protocol
      # Paper comparisons default to the published harness (docs/harness-fixes.md).
      harness="${harness:-published}"
      scope="Run an ASPIRE iterative-debugging campaign with the paper's protocol (arXiv 2607.00272, Section 3 and Appendix E): the full seed partitions below, the skill's own stopping rules and Stage-2 held-out validation. Do not shorten the seed lists or skip a stage."
      version_rule="Budget: no fixed cap on program versions; workers debug until the development threshold or a plateau, as subagent-prompt.md says."
    fi
    if [[ "$memory" == on ]]; then
      memory_rule="Memory is ON (the full ASPIRE loop): when a worker proposes skills, dispatch the verifier, and promote what it reproduces into .claude/skills/iterative-debugging/skills/ as the skill describes, so later tasks read it. Choose the task order as the curriculum section says. When every task has its held-out results, run the phase-3 rerun sweep once."
    else
      memory_rule="Memory is OFF (a no-memory baseline): every task starts from the same skill library. Do not dispatch verifiers, do not promote skills, and do not edit anything under .claude/skills/iterative-debugging/skills/. Workers still write skill_report.json with their proposals; they are kept for analysis only. Skip the phase-3 rerun sweep: it exists to apply a changed library."
    fi
    harness="${harness:-fixed}"
    case "$library" in
      current) ;;
      empty) scripts/gpu/library.sh empty ;;
      released) scripts/gpu/library.sh released ;;
      *) scripts/gpu/library.sh load "$library" ;;
    esac
    step "Initializing a $label campaign from $tasks (memory $memory, library $library, harness $harness)"
    # shellcheck disable=SC2086  # seed lists are space-separated on purpose
    "$PY" .claude/skills/iterative-debugging/scripts/init_run.py --tasks "$tasks" --name "$name" \
      --dev-seeds $dev --unseen-seeds $unseen --heldout-count "$heldout"
    run_root="$(cd outputs/runs/LATEST && pwd -P)"
    scripts/gpu/library.sh save "$(basename "$run_root")-start" --force
    # Everything a --resume needs to continue with the same settings.
    {
      printf 'dev=%q unseen=%q heldout=%q memory=%q harness=%q label=%q\n' \
        "$dev" "$unseen" "$heldout" "$memory" "$harness" "$label"
      printf 'scope=%q\nversion_rule=%q\nmemory_rule=%q\n' "$scope" "$version_rule" "$memory_rule"
    } > "$run_root/campaign_settings.sh"
    campaign_meta "$run_root" "campaign_name=$name" "library_start=$library" \
      "library_start_snapshot=$(basename "$run_root")-start" \
      "library_start_words=$(cat .claude/skills/iterative-debugging/skills/*.md | wc -w | tr -d ' ')"
  fi
  if [[ "$harness" == published ]]; then export CAP_HARNESS_COMPAT=published
  else unset CAP_HARNESS_COMPAT; fi
  budget="${budget:-30}"; max_turns="${max_turns:-400}"
  local curobo_note=""
  ((CUROBO_UP)) && curobo_note=", cuRobo :8118"
  render scripts/gpu/prompts/mini_campaign.md "$run_root/kickoff_prompt.md" \
    "RUN_ROOT=$run_root" "DEV_SEEDS=${dev// /, }" "UNSEEN_SEEDS=${unseen// /, }" \
    "HELDOUT_COUNT=$heldout" "SCOPE=$scope" "VERSION_RULE=$version_rule" \
    "MEMORY_RULE=$memory_rule" "RESUME_NOTE=$resume_note" \
    "GPU=$GPU" "CUROBO_NOTE=$curobo_note"
  echo "kickoff prompt: $run_root/kickoff_prompt.md"
  if ((headless)); then
    campaign_headless "$run_root" "$budget" "$max_turns" "$memory"
    return
  fi
  if ((run)); then
    step "Starting the interactive coordinator (Claude Code, ${ANTHROPIC_MODEL})"
    echo "When it is done: scripts/gpu/library.sh save $(basename "$run_root")-end"
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
