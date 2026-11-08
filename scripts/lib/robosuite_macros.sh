# Shared helper: make `import robosuite` work on a machine shared by several users.
#
# robosuite's DefaultLogger opens the hard-coded file /tmp/robosuite.log. Whoever creates it
# first owns it, and everyone else then fails at import with
#   PermissionError: [Errno 13] Permission denied: '/tmp/robosuite.log'
# robosuite's supported customisation point is an untracked robosuite/macros_private.py (its
# .gitignore lists it), imported first by robosuite/__init__.py. Ours switches the file log off;
# console logging is unchanged.

# write_robosuite_private_macros <repo_root>
write_robosuite_private_macros() {
  local repo_root="$1"
  local target="$repo_root/third_party/robosuite/robosuite/macros_private.py"
  local marker="Written by ASPIRE's setup scripts."
  if [[ ! -d "$(dirname -- "$target")" ]]; then
    return 0  # submodule not checked out yet
  fi
  if [[ -f "$target" ]] && ! grep -qF "$marker" "$target"; then
    printf 'warning: %s exists and was not written by ASPIRE; leaving it alone.\n' "$target" >&2
    printf '         If import robosuite fails on /tmp/robosuite.log, add to it:\n' >&2
    printf '           import robosuite.macros as _macros\n           _macros.FILE_LOGGING_LEVEL = None\n' >&2
    return 0
  fi
  cat > "$target" <<'PY'
"""Machine-local robosuite settings (untracked; robosuite's .gitignore lists this file).

Written by ASPIRE's setup scripts. robosuite's DefaultLogger writes to the hard-coded path
/tmp/robosuite.log. On a shared machine that file belongs to whichever user created it first,
and every other user's `import robosuite` then fails with PermissionError. Turn the file log
off; console logging is unchanged. log_utils reads robosuite.macros, so set it there.
"""

import robosuite.macros as _macros

_macros.FILE_LOGGING_LEVEL = None
PY
}
