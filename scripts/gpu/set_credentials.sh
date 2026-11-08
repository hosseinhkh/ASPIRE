#!/usr/bin/env bash
# Store the credentials ASPIRE's demo needs in the conda env, so every `conda activate aspire`
# shell has them:
#
#   * Amazon Bedrock for Claude Code, pinned to Claude Opus 4.6 (CLAUDE_CODE_USE_BEDROCK=1,
#     AWS_REGION, credentials, ANTHROPIC_MODEL / ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL).
#     The ASPIRE coordinator spawns sub-agents with model="opus"; pinning the opus alias
#     (and the others) keeps every agent on Opus 4.6.
#   * HF_TOKEN for the gated SAM3 weights (facebook/sam3).
#
# Secrets are read with hidden input and never echoed. conda stores env vars in plain text in
# $CONDA_PREFIX/conda-meta/state; this script makes that file readable by you only. Re-run any
# time to change keys; --unset removes everything it set.
#
# Usage: scripts/gpu/set_credentials.sh [--env aspire] [--region us-east-1] [--model-id ID]
#                                       [--1m] [--skip-hf] [--skip-verify] [--unset]
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ASPIRE_CONDA_ENV:-aspire}"
REGION=""
MODEL_ID=""
ONE_M=0
SKIP_HF=0
SKIP_VERIFY=0
UNSET=0

while (($#)); do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --1m) ONE_M=1; shift ;;
    --skip-hf) SKIP_HF=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --unset) UNSET=1; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v conda >/dev/null 2>&1 || { echo "error: conda not found" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

ALL_VARS=(CLAUDE_CODE_USE_BEDROCK AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
  AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK AWS_PROFILE ANTHROPIC_MODEL
  ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL HF_TOKEN)

if [[ "$UNSET" -eq 1 ]]; then
  conda env config vars unset -n "$ENV_NAME" "${ALL_VARS[@]}" >/dev/null
  echo "removed: ${ALL_VARS[*]}"
  echo "run 'conda deactivate && conda activate $ENV_NAME' to drop them from this shell"
  exit 0
fi

# ---------------------------------------------------------------------------------------------
echo "== Amazon Bedrock (Claude Code, Opus 4.6)"
if [[ -z "$REGION" ]]; then
  read -r -p "AWS region with Bedrock access [us-east-1]: " REGION
  REGION="${REGION:-us-east-1}"
fi
echo "Authentication:"
echo "  1) access key ID + secret access key (+ optional session token)"
echo "  2) Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)"
echo "  3) a named AWS profile from ~/.aws (e.g. SSO; run 'aws sso login' yourself)"
read -r -p "Choose 1/2/3 [1]: " METHOD
METHOD="${METHOD:-1}"

declare -a SET_VARS=(CLAUDE_CODE_USE_BEDROCK=1 "AWS_REGION=$REGION")
declare -a CLEAR_VARS=()
export AWS_REGION="$REGION"
case "$METHOD" in
  1)
    read -r -p "AWS_ACCESS_KEY_ID: " KEY_ID
    read -r -s -p "AWS_SECRET_ACCESS_KEY (hidden): " SECRET; echo
    read -r -s -p "AWS_SESSION_TOKEN (hidden, Enter if none): " SESSION; echo
    [[ -n "$KEY_ID" && -n "$SECRET" ]] || { echo "error: key ID and secret are required" >&2; exit 1; }
    SET_VARS+=("AWS_ACCESS_KEY_ID=$KEY_ID" "AWS_SECRET_ACCESS_KEY=$SECRET")
    export AWS_ACCESS_KEY_ID="$KEY_ID" AWS_SECRET_ACCESS_KEY="$SECRET"
    if [[ -n "$SESSION" ]]; then
      SET_VARS+=("AWS_SESSION_TOKEN=$SESSION"); export AWS_SESSION_TOKEN="$SESSION"
    else
      CLEAR_VARS+=(AWS_SESSION_TOKEN); unset AWS_SESSION_TOKEN || true
    fi
    CLEAR_VARS+=(AWS_BEARER_TOKEN_BEDROCK AWS_PROFILE)
    unset AWS_BEARER_TOKEN_BEDROCK AWS_PROFILE || true
    ;;
  2)
    read -r -s -p "AWS_BEARER_TOKEN_BEDROCK (hidden): " BEARER; echo
    [[ -n "$BEARER" ]] || { echo "error: empty API key" >&2; exit 1; }
    SET_VARS+=("AWS_BEARER_TOKEN_BEDROCK=$BEARER"); export AWS_BEARER_TOKEN_BEDROCK="$BEARER"
    CLEAR_VARS+=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE)
    ;;
  3)
    read -r -p "AWS profile name: " PROFILE_NAME
    [[ -n "$PROFILE_NAME" ]] || { echo "error: empty profile name" >&2; exit 1; }
    SET_VARS+=("AWS_PROFILE=$PROFILE_NAME"); export AWS_PROFILE="$PROFILE_NAME"
    CLEAR_VARS+=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK)
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK || true
    ;;
  *) echo "error: choose 1, 2 or 3" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------------------------
# Opus 4.6 on Bedrock needs an inference profile ID ("us.anthropic.claude-opus-4-6-v1");
# the bare model ID is rejected for on-demand use. Ask Bedrock which ones this account has.
if [[ -z "$MODEL_ID" ]]; then
  case "$REGION" in
    us-gov-*) PREFIX="us-gov" ;;
    us-*) PREFIX="us" ;;
    eu-*) PREFIX="eu" ;;
    ap-*) PREFIX="apac" ;;
    *) PREFIX="global" ;;
  esac
  MODEL_ID="$PREFIX.anthropic.claude-opus-4-6-v1"
  if [[ "$METHOD" != "2" ]]; then
    DETECTED="$(python - "$REGION" "$PREFIX" <<'PY' 2>/dev/null || true
import sys
import boto3

region, prefix = sys.argv[1], sys.argv[2]
client = boto3.client("bedrock", region_name=region)
ids, token = [], None
while True:
    kwargs = {"maxResults": 1000, **({"nextToken": token} if token else {})}
    page = client.list_inference_profiles(**kwargs)
    ids += [p["inferenceProfileId"] for p in page.get("inferenceProfileSummaries", [])]
    token = page.get("nextToken")
    if not token:
        break
opus = sorted(i for i in ids if "claude-opus-4-6" in i)
preferred = [i for i in opus if i.startswith(prefix + ".")] or [i for i in opus if i.startswith("global.")] or opus
print(preferred[0] if preferred else "")
PY
)"
    if [[ -n "$DETECTED" ]]; then
      MODEL_ID="$DETECTED"
      echo "found Opus 4.6 inference profile in $REGION: $MODEL_ID"
    else
      echo "warning: could not list inference profiles (missing bedrock:ListInferenceProfiles," >&2
      echo "         or Opus 4.6 not enabled in $REGION); using $MODEL_ID" >&2
    fi
  else
    echo "using $MODEL_ID (an API key cannot list inference profiles; override with --model-id)"
  fi
fi
[[ "$ONE_M" -eq 1 && "$MODEL_ID" != *"[1m]" ]] && MODEL_ID="$MODEL_ID[1m]"
SET_VARS+=(
  "ANTHROPIC_MODEL=$MODEL_ID"
  "ANTHROPIC_DEFAULT_OPUS_MODEL=$MODEL_ID"
  "ANTHROPIC_DEFAULT_SONNET_MODEL=$MODEL_ID"
  "ANTHROPIC_DEFAULT_HAIKU_MODEL=$MODEL_ID"
)

# ---------------------------------------------------------------------------------------------
if [[ "$SKIP_HF" -eq 0 ]]; then
  echo
  echo "== Hugging Face (gated SAM3 weights; request access at https://huggingface.co/facebook/sam3)"
  read -r -s -p "HF_TOKEN (hidden, Enter to keep the current one): " HF; echo
  if [[ -n "$HF" ]]; then
    SET_VARS+=("HF_TOKEN=$HF"); export HF_TOKEN="$HF"
  fi
fi

# ---------------------------------------------------------------------------------------------
if [[ "${#CLEAR_VARS[@]}" -gt 0 ]]; then
  conda env config vars unset -n "$ENV_NAME" "${CLEAR_VARS[@]}" >/dev/null 2>&1 || true
fi
conda env config vars set -n "$ENV_NAME" "${SET_VARS[@]}" >/dev/null
STATE_FILE="$(conda info --base)/envs/$ENV_NAME/conda-meta/state"
[[ -f "$CONDA_PREFIX/conda-meta/state" ]] && STATE_FILE="$CONDA_PREFIX/conda-meta/state"
[[ -f "$STATE_FILE" ]] && chmod 600 "$STATE_FILE"
echo
echo "stored in conda env '$ENV_NAME' (values hidden):"
for entry in "${SET_VARS[@]}"; do
  name="${entry%%=*}"
  case "$name" in
    AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|AWS_BEARER_TOKEN_BEDROCK|HF_TOKEN|AWS_ACCESS_KEY_ID)
      echo "  $name=****" ;;
    *) echo "  $entry" ;;
  esac
done
export CLAUDE_CODE_USE_BEDROCK=1 ANTHROPIC_MODEL="$MODEL_ID" ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_ID" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_ID" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_ID"

# ---------------------------------------------------------------------------------------------
if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  echo
  echo "== Verifying Claude Code -> Bedrock -> $MODEL_ID"
  if command -v claude >/dev/null; then
    if reply="$(timeout 180 claude -p "Reply with exactly the word OK." --model "$MODEL_ID" --max-turns 1 2>&1)"; then
      echo "claude replied: $reply"
    else
      echo "error: Claude Code could not reach Bedrock:" >&2
      echo "$reply" >&2
      echo "check the region, IAM (bedrock:InvokeModel*, bedrock:ListInferenceProfiles) and" >&2
      echo "that Opus 4.6 is enabled in the Bedrock console's model access page" >&2
    fi
  else
    echo "warning: 'claude' not on PATH; run scripts/gpu/setup_conda_env.sh first" >&2
  fi

  if [[ -n "${HF_TOKEN:-}" && -x "$REPO_ROOT/.venv-sam3/bin/python" ]]; then
    echo
    echo "== Checking SAM3 access and pre-downloading its weights (~3.4 GB, once)"
    "$REPO_ROOT/.venv-sam3/bin/python" - <<'PY' || echo "error: SAM3 download failed; is access to facebook/sam3 granted for this token?" >&2
from huggingface_hub import hf_hub_download

for name in ("config.json", "sam3.pt"):
    print("ok:", hf_hub_download(repo_id="facebook/sam3", filename=name))
PY
  fi
fi

echo
echo "Done. Load the variables into this shell with:  conda deactivate && conda activate $ENV_NAME"
