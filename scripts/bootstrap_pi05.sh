#!/usr/bin/env bash
# Build .venv-pi05: the environment for the pi0.5 student (LeRobot v0.4.4's pi05 policy).
#
#   scripts/bootstrap_pi05.sh [--force] [--cpu] [--python PATH]
#
# Why a second environment: LeRobot v0.4.4's pi0.5 needs `transformers` from the branch
# fix/lerobot_openpi (4.53.3, tokenizers < 0.22); SmolVLA's .venv-libero has transformers 4.57
# and LeRobot's SmolVLA extra requires >= 4.57.1. Installing pi0.5 there would break SmolVLA.
#
# How it stays identical to the SmolVLA environment everywhere else:
#   * every package version is pinned to what .venv-libero has (`uv pip freeze`), except
#     transformers and tokenizers (pi0.5's), and scipy (LeRobot's pi extra asks for < 1.15);
#   * the repository's own uv overrides and build settings are reused (numpy 1.26.4,
#     mujoco 3.3.0, ...), with transformers pinned to one commit of that branch;
#   * the same editable checkouts: this repository, third_party/LIBERO-PRO, robosuite, lerobot.
#
# It never touches pyproject.toml, uv.lock or .venv-libero: those are read only.
# --cpu installs the CPU build of torch (for testing on a machine without a GPU).
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
LIBERO_VENV="$REPO_ROOT/.venv-libero"
VENV_ROOT="$(resolve_canonical_venv "$REPO_ROOT" ".venv-pi05")"
UV_CACHE_ROOT="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
TRANSFORMERS_PI05="transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
FORCE=0 CPU=0 PYTHON_BIN=""

# The comment block at the top, whatever its length.
usage() { sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
while (($#)); do
  case "$1" in
    --force) FORCE=1; shift ;;
    --cpu) CPU=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null 2>&1 || { echo "error: uv is required" >&2; exit 1; }
[[ -x "$LIBERO_VENV/bin/python" ]] || {
  echo "error: $LIBERO_VENV is missing: build the SmolVLA environment first (scripts/gpu/vla.sh setup)" >&2
  exit 1
}
"$LIBERO_VENV/bin/python" -c "import lerobot" 2>/dev/null || {
  echo "error: LeRobot is not in $LIBERO_VENV: run scripts/gpu/vla.sh setup first" >&2
  exit 1
}
for path in third_party/lerobot third_party/LIBERO-PRO third_party/robosuite; do
  [[ -f "$REPO_ROOT/$path/pyproject.toml" || -f "$REPO_ROOT/$path/setup.py" ]] || {
    echo "error: $path is not checked out (git submodule update --init $path)" >&2
    exit 1
  }
done
if [[ -x "$VENV_ROOT/bin/python" && -f "$VENV_ROOT/pi05-environment.json" ]] && ((!FORCE)); then
  echo "$VENV_ROOT exists (pass --force to rebuild it)"
  exit 0
fi
# The same Python as the SmolVLA environment.
[[ -n "$PYTHON_BIN" ]] || PYTHON_BIN="$("$LIBERO_VENV/bin/python" -c 'import sys; print(sys._base_executable)')"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
uv pip freeze --python "$LIBERO_VENV/bin/python" > "$WORK/libero-freeze.txt"

# Pins, uv configuration and the package list, from .venv-libero and pyproject.toml.
"$LIBERO_VENV/bin/python" - "$REPO_ROOT" "$WORK" "$TRANSFORMERS_PI05" "$CPU" <<'PY'
import json
from pathlib import Path
import re
import sys
import tomllib

repo, work, transformers_pi05, cpu = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
uv = project["tool"]["uv"]


def name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[@]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


# pi0.5's own versions: never pinned from SmolVLA's environment.
OWN = {"transformers", "tokenizers", "scipy"}
# Installed from the checkouts below, not from the freeze.
EDITABLE = {"cap-harness", "libero", "robosuite", "lerobot"}
constraints = []
for line in (work / "libero-freeze.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith(("#", "-e")) or " @ " in line:
        continue
    if name(line) in OWN | EDITABLE:
        continue
    if cpu and name(line).startswith("nvidia-"):
        continue
    constraints.append(line)
(work / "constraints.txt").write_text("\n".join(constraints) + "\n", encoding="utf-8")

overrides = [o for o in uv["override-dependencies"] if name(o) not in {"transformers", "scipy"}]
# LeRobot's `pi` extra asks for scipy < 1.15; the repository's override would replace that.
overrides += [transformers_pi05, "scipy>=1.10.1,<1.15"]
if cpu:
    lock = json.loads((repo / "configs" / "dependency-lock.json").read_text(encoding="utf-8"))
    runtime = lock["gpu_runtime"]
    overrides += [f"torch=={runtime['torch']}+cpu", f"torchvision=={runtime['torchvision']}+cpu"]


def toml_value(value):
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(value)


config = [f"override-dependencies = {toml_value(overrides)}"]
if uv.get("extra-build-dependencies"):
    config.append(f"extra-build-dependencies = {toml_value(uv['extra-build-dependencies'])}")
(work / "uv.toml").write_text("\n".join(config) + "\n", encoding="utf-8")

extras = project["project"]["optional-dependencies"]
packages = [p for p in extras["libero"] if name(p) not in EDITABLE]
packages += [p for p in extras["vla"] if name(p) != "lerobot"]
packages += ["peft>=0.18.0,<1.0.0", "pytest>=8"]
(work / "packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
print(f"pins from .venv-libero: {len(constraints)}; overrides: {len(overrides)}; packages: {len(packages)}")
PY

echo "creating $VENV_ROOT with $PYTHON_BIN"
UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
INDEX=()
((CPU)) && INDEX=(--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
# From the work folder, so uv reads only the configuration written above.
(
  cd "$WORK"
  UV_CACHE_DIR="$UV_CACHE_ROOT" uv pip install \
    --config-file "$WORK/uv.toml" \
    --python "$VENV_ROOT/bin/python" \
    --constraints "$WORK/constraints.txt" \
    "${INDEX[@]+"${INDEX[@]}"}" \
    -e "$REPO_ROOT" \
    -e "$REPO_ROOT/third_party/LIBERO-PRO" \
    -e "$REPO_ROOT/third_party/robosuite" \
    -e "$REPO_ROOT/third_party/lerobot[pi]" \
    -r "$WORK/packages.txt"
)

# Verify: pi0.5's transformers, the shared simulator stack, the GPU runtime, the imports.
"$VENV_ROOT/bin/python" - "$REPO_ROOT" "$LIBERO_VENV/bin/python" "$CPU" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

repo, libero_python, cpu = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
SHARED = ("numpy", "mujoco", "robosuite", "libero", "lerobot", "torch", "torchvision", "pandas",
          "pyarrow", "huggingface-hub", "safetensors", "peft", "accelerate", "gymnasium")
problems = []

theirs = json.loads(subprocess.check_output([libero_python, "-c", (
    "import importlib.metadata as m, json, sys\n"
    f"print(json.dumps({{p: m.version(p) for p in {SHARED!r}}}))"
)], text=True))
ours = {p: metadata.version(p) for p in SHARED}
for package in SHARED:
    a, b = ours[package].split("+")[0], theirs[package].split("+")[0]
    if a != b:
        problems.append(f"{package} {ours[package]} here, {theirs[package]} in .venv-libero")
from transformers.models.siglip import check  # noqa: E402

if not check.check_whether_transformers_replace_is_installed_correctly():
    problems.append("transformers is not LeRobot's fix/lerobot_openpi branch")
if not cpu:
    import torch

    runtime = json.loads((repo / "configs" / "dependency-lock.json").read_text())["gpu_runtime"]
    if torch.version.cuda != runtime["cuda_runtime"]:
        problems.append(f"CUDA runtime {torch.version.cuda}, expected {runtime['cuda_runtime']}")
import lerobot.policies.pi05.modeling_pi05  # noqa: E402,F401
import cap_harness.trace2vla.cli  # noqa: E402,F401
import cap_harness.vla.policy  # noqa: E402,F401

if problems:
    raise SystemExit("pi0.5 environment check FAILED:\n  " + "\n  ".join(problems))
record = {
    "transformers": metadata.version("transformers"),
    "tokenizers": metadata.version("tokenizers"),
    "scipy": metadata.version("scipy"),
    "shared_with_libero": ours,
    "cpu_only": cpu,
}
(Path(sys.prefix) / "pi05-environment.json").write_text(json.dumps(record, indent=2) + "\n")
print("pi0.5 environment OK: transformers", record["transformers"], "| tokenizers", record["tokenizers"])
print("same as .venv-libero:", ", ".join(f"{k} {v}" for k, v in ours.items()))
PY
echo "done: $VENV_ROOT"
