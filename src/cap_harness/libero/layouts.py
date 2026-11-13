"""Sibling layouts: LIBERO-Pro's swap with each task's other partners, as extra suites.

LIBERO-Pro's ``libero_object_swap`` moves each task's target object into the place of one other
object, drawn at random from five candidates (``libero_ood/ood_spatial_relation.yaml``); the
official suite is one draw. The other four candidates give four more layouts per task, with the
same instruction and the same objects:

    libero_object_swap_alt1 ... alt4    task i: the target swapped with its k-th other candidate

Training on several of them makes the target's position vary for one instruction, so a student
cannot solve the task by remembering where to reach; the official swap and the standard layout
stay held out for testing.

The files are made with LIBERO-Pro's own rule: exchange the two objects' ``(On object region)``
statements in the original LIBERO-Object task. Given the official partner, the rule reproduces
LIBERO-Pro's official swap files byte for byte (checked by the tests). They are written to
``configs/libero_layouts/`` with a manifest, and registered next to LIBERO's own suites, so every
command that takes a suite takes these too. They ship no saved initial states: use them with
seeded scene sets (``init_mode`` seeded) only.

    python -m cap_harness.libero.layouts           write the files (deterministic)
    python -m cap_harness.libero.layouts --check   confirm the committed files are current

Standard library only (the YAML candidate list is read with a small parser for its two levels).
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LIBERO_PRO = REPOSITORY_ROOT / "third_party" / "LIBERO-PRO"
LAYOUTS_DIR = REPOSITORY_ROOT / "configs" / "libero_layouts"
MANIFEST = LAYOUTS_DIR / "manifest.json"
#: Suites that get sibling layouts: swap suite -> (original suite, its key in the YAML).
FAMILIES = {"libero_object_swap": ("libero_object", "libero_object")}
ALTERNATIVES = 4


# ------------------------------------------------------------------------- the BDDL rule --
def init_regions(text: str) -> dict[str, str]:
    """Object -> region of the ``(On object region)`` statements in ``(:init ...)``, parsed as
    LIBERO-Pro's BDDLParser does.
    """
    block = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", text, re.DOTALL)
    if not block:
        return {}
    return dict(re.findall(r"\(On\s+(\w+)\s+(\w+)\)", block.group(1)))


def swap(text: str, first: str, second: str) -> str:
    """The BDDL with the two objects' starting regions exchanged (LIBERO-Pro's SwapPerturbator,
    without the random choice).
    """
    regions = init_regions(text)
    if first not in regions or second not in regions:
        raise ValueError(f"{first} or {second} has no (On ...) statement in (:init ...)")
    a, b = regions[first], regions[second]
    text, n1 = re.subn(
        rf"\(On\s+{re.escape(first)}\s+{re.escape(a)}\s*\)", f"(On {first} {b})", text, count=1
    )
    text, n2 = re.subn(
        rf"\(On\s+{re.escape(second)}\s+{re.escape(b)}\s*\)", f"(On {second} {a})", text, count=1
    )
    if n1 != 1 or n2 != 1:
        raise ValueError(f"could not exchange {first} and {second}")
    return text


def read_candidates(path: Path) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Suite key -> task -> object -> candidates, from LIBERO-Pro's YAML (its plain layout:
    two-space indents, ``- item`` lists).
    """
    data: dict[str, dict[str, dict[str, list[str]]]] = {}
    suite = task = obj = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            if suite and task and obj:
                data[suite][task][obj].append(line[2:].strip())
            continue
        key = line.rstrip(":").strip()
        if indent == 0:
            suite, task, obj = key, None, None
            data.setdefault(suite, {})
        elif indent == 2 and suite:
            task, obj = key, None
            data[suite].setdefault(task, {})
        elif indent == 4 and suite and task:
            obj = key
            data[suite][task].setdefault(obj, [])
    return data


def task_order(suite: str, libero_pro: Path = LIBERO_PRO) -> list[str]:
    """The task names of a LIBERO-Pro suite in benchmark order (its task map, a Python literal)."""
    source = (libero_pro / "libero/libero/benchmark/libero_suite_task_map.py").read_text(
        encoding="utf-8"
    )
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            mapping = ast.literal_eval(node.value)
            if suite in mapping:
                return list(mapping[suite])
    raise KeyError(f"no task list for {suite!r} in LIBERO-Pro's task map")


# ------------------------------------------------------------------------------ generate --
@dataclass(frozen=True)
class SiblingTask:
    task_id: int
    name: str
    target: str
    partner: str
    official_partner: str
    bddl: str  # relative to LAYOUTS_DIR


def plan(libero_pro: Path = LIBERO_PRO) -> dict[str, list[SiblingTask]]:
    """Alt suite name -> its tasks, for every family in FAMILIES."""
    candidates = read_candidates(libero_pro / "libero_ood" / "ood_spatial_relation.yaml")
    bddl_root = libero_pro / "libero" / "libero" / "bddl_files"
    suites: dict[str, list[SiblingTask]] = {}
    for swap_suite, (original_suite, key) in FAMILIES.items():
        for task_id, name in enumerate(task_order(swap_suite, libero_pro)):
            spec = candidates[key][name]
            if len(spec) != 1:
                raise ValueError(f"{name}: expected one target object, found {sorted(spec)}")
            target, options = next(iter(spec.items()))
            original = init_regions((bddl_root / original_suite / f"{name}.bddl").read_text())
            official = init_regions((bddl_root / swap_suite / f"{name}.bddl").read_text())
            moved = [o for o in original if official.get(o) != original[o] and o != target]
            if len(moved) != 1 or moved[0] not in options:
                raise ValueError(f"{name}: cannot tell the official partner ({moved})")
            others = [c for c in options if c != moved[0]]
            if len(others) < ALTERNATIVES:
                raise ValueError(f"{name}: only {len(others)} other candidates")
            for index, partner in enumerate(others[:ALTERNATIVES], start=1):
                alt = f"{swap_suite}_alt{index}"
                suites.setdefault(alt, []).append(
                    SiblingTask(task_id, name, target, partner, moved[0], f"{alt}/{name}.bddl")
                )
    return suites


def generate(out: Path = LAYOUTS_DIR, libero_pro: Path = LIBERO_PRO) -> dict[str, Any]:
    """Write every sibling BDDL and the manifest; return the manifest."""
    bddl_root = libero_pro / "libero" / "libero" / "bddl_files"
    manifest: dict[str, Any] = {
        "about": "LIBERO-Pro swap with each task's other candidates; see cap_harness.libero.layouts",
        "source": "third_party/LIBERO-PRO libero_ood/ood_spatial_relation.yaml",
        "suites": {},
    }
    for alt, tasks in plan(libero_pro).items():
        swap_suite = alt.rsplit("_alt", 1)[0]
        original_suite = FAMILIES[swap_suite][0]
        entries = []
        for task in tasks:
            text = (bddl_root / original_suite / f"{task.name}.bddl").read_text(encoding="utf-8")
            path = out / task.bddl
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(swap(text, task.target, task.partner), encoding="utf-8")
            entries.append(task.__dict__)
        manifest["suites"][alt] = {"base": swap_suite, "init_mode": "seeded", "tasks": entries}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# ------------------------------------------------------------------------------ register --
class _Task:
    """What LIBERO's Task offers, plus ``bddl_path`` (read first by the registry)."""

    def __init__(self, entry: Mapping[str, Any], suite: str, root: Path) -> None:
        self.name = str(entry["name"])
        self.bddl_path = root / str(entry["bddl"])
        self.bddl_file = str(self.bddl_path)
        self.problem_folder = suite
        self.problem = "Libero"
        self.init_states_file = ""
        # LIBERO's own rule (grab_language_from_filename): the instruction the student is given
        # comes from the task's file name, so every layout of a task gets the same words.
        self.language = " ".join(self.name.split("_"))
        self.target, self.partner = entry["target"], entry["partner"]


class SiblingSuite:
    """A LIBERO-style benchmark over generated sibling layouts. No saved initial states."""

    #: Read by the registry: this suite is used with seeded layouts only.
    seeded_only = True

    def __init__(self, name: str, spec: Mapping[str, Any], root: Path = LAYOUTS_DIR) -> None:
        self.name = name
        self.tasks = [_Task(entry, name, root) for entry in spec["tasks"]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task(self, i: int) -> _Task:
        return self.tasks[i]

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task_bddl_file_path(self, i: int) -> str:
        return self.tasks[i].bddl_file

    def get_task_init_states(self, i: int) -> list[Any]:
        return []


def suites(manifest: Path = MANIFEST) -> dict[str, Any]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(data.get("suites") or {})


def register(mapping: dict[str, Any], manifest: Path = MANIFEST) -> list[str]:
    """Add the sibling suites to LIBERO's benchmark mapping (``get_benchmark_dict()``); keeps
    any suite already there. Returns the names added.
    """
    added = []
    root = manifest.parent
    for name, spec in suites(manifest).items():
        if name in mapping:
            continue

        def factory(*_args: Any, _name: str = name, _spec: Any = spec, **_kwargs: Any) -> Any:
            return SiblingSuite(_name, _spec, root)

        mapping[name] = factory
        added.append(name)
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail if the committed files differ")
    args = parser.parse_args(argv)
    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fresh = generate(Path(tmp))
            stale = [
                t["bddl"]
                for spec in fresh["suites"].values()
                for t in spec["tasks"]
                if not (LAYOUTS_DIR / t["bddl"]).is_file()
                or (LAYOUTS_DIR / t["bddl"]).read_text() != (Path(tmp) / t["bddl"]).read_text()
            ]
            committed = suites()
        if stale or committed != fresh["suites"]:
            print(f"out of date: {stale or 'manifest'}; run python -m cap_harness.libero.layouts")
            return 1
        print(f"sibling layouts up to date: {', '.join(fresh['suites'])}")
        return 0
    manifest = generate()
    for name, spec in manifest["suites"].items():
        print(f"{name}: {len(spec['tasks'])} tasks")
        for task in spec["tasks"]:
            print(f"  {task['task_id']}: {task['target']} in the place of {task['partner']} "
                  f"(official: {task['official_partner']})")  # fmt: skip
    print(f"written to {LAYOUTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
