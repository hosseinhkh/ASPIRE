#!/usr/bin/env bash
# The ASPIRE skill library: the only memory that carries over between tasks and campaigns.
# It is plain markdown in .claude/skills/iterative-debugging/skills/. This script keeps named
# copies of it so experiments can start from a known library and compare what was learned.
#
#   scripts/gpu/library.sh status            words per file, fingerprint, change vs the release
#   scripts/gpu/library.sh list              saved libraries
#   scripts/gpu/library.sh save NAME         copy the current library to outputs/libraries/NAME
#   scripts/gpu/library.sh load NAME         replace the current library with a saved one
#   scripts/gpu/library.sh empty             keep each skill file's header, drop every entry
#   scripts/gpu/library.sh released          restore the library shipped with the repository
#   scripts/gpu/library.sh diff NAME [NAME2] what changed between two libraries (default: current)
#
# load, empty and released first save the current library as outputs/libraries/backup-<time>.
# README.md and task-exploration.md (the exploration procedure, not learned knowledge) are
# never emptied.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILLS="${ASPIRE_SKILLS_DIR:-$REPO_ROOT/.claude/skills/iterative-debugging/skills}"
STORE="${ASPIRE_LIBRARY_STORE:-$REPO_ROOT/outputs/libraries}"
LEARNED=(grasp.md localize.md transport.md manipulation.md)
PY="${CAP_HARNESS_PYTHON:-python3}"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }
fingerprint() { cat "$1"/*.md 2>/dev/null | sha256sum | cut -c1-12; }
words() { cat "$1"/*.md 2>/dev/null | wc -w | tr -d ' '; }

backup() {
  local name n=1
  name="backup-$(date +%Y%m%d-%H%M%S)"
  while [[ -e "$STORE/$name" ]]; do n=$((n + 1)); name="backup-$(date +%Y%m%d-%H%M%S)-$n"; done
  mkdir -p "$STORE/$name"
  cp "$SKILLS"/*.md "$STORE/$name/"
  echo "saved the current library as $name"
}

case "${1:-}" in
  status)
    echo "library: ${SKILLS#"$REPO_ROOT"/}  ($(words "$SKILLS") words, fingerprint $(fingerprint "$SKILLS"))"
    for file in "$SKILLS"/*.md; do
      printf '  %-22s %6s words  %3s sections\n' "$(basename "$file")" "$(wc -w < "$file" | tr -d ' ')" \
        "$(grep -c '^## ' "$file" || true)"
    done
    changed="$(git -C "$REPO_ROOT" status --porcelain -- "$SKILLS" | wc -l | tr -d ' ')"
    if [[ "$changed" == 0 ]]; then echo "same as the released library"
    else echo "$changed file(s) differ from the released library (git diff -- ${SKILLS#"$REPO_ROOT"/})"; fi
    ;;
  list)
    mkdir -p "$STORE"
    for dir in "$STORE"/*/; do
      [[ -d "$dir" ]] || continue
      printf '  %-40s %6s words  fingerprint %s\n' "$(basename "$dir")" "$(words "$dir")" "$(fingerprint "$dir")"
    done
    ;;
  save)
    name="${2:?usage: library.sh save NAME}"
    [[ -e "$STORE/$name" && "${3:-}" != "--force" ]] && { echo "error: $name exists (add --force)" >&2; exit 1; }
    mkdir -p "$STORE/$name"
    cp "$SKILLS"/*.md "$STORE/$name/"
    echo "saved the library as $name ($(words "$STORE/$name") words, fingerprint $(fingerprint "$STORE/$name"))"
    ;;
  load)
    name="${2:?usage: library.sh load NAME}"
    [[ -d "$STORE/$name" ]] || { echo "error: no saved library $name (library.sh list)" >&2; exit 1; }
    backup
    cp "$STORE/$name"/*.md "$SKILLS/"
    echo "loaded $name ($(words "$SKILLS") words, fingerprint $(fingerprint "$SKILLS"))"
    ;;
  empty)
    backup
    for file in "${LEARNED[@]}"; do
      # Keep the front matter, the title and the introduction; drop every "## " section.
      "$PY" - "$SKILLS/$file" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
head = []
for line in lines:
    if line.startswith("## "):
        break
    head.append(line)
text = "\n".join(head).rstrip() + "\n\n_No entries yet: validated skills are added here as the library grows._\n"
path.write_text(text, encoding="utf-8")
PY
    done
    echo "emptied ${LEARNED[*]} ($(words "$SKILLS") words left in the library)"
    ;;
  released)
    backup
    git -C "$REPO_ROOT" checkout -- "$SKILLS"
    echo "restored the released library ($(words "$SKILLS") words)"
    ;;
  diff)
    a="$STORE/${2:?usage: library.sh diff NAME [NAME2]}"
    b="${3:+$STORE/$3}"
    b="${b:-$SKILLS}"
    diff -ru "$a" "$b" || true
    ;;
  *) usage; [[ -z "${1:-}" ]] || exit 2 ;;
esac
