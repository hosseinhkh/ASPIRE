"""Named scene sets: which scenes are for testing, which for training, which for development.

A scene is (suite, task, init mode, seed). In ``saved`` mode seed k is the task's saved init
state k-1: seeds 1-50 are the official test scenes of LIBERO and LIBERO-Pro. In ``seeded`` mode
the seed drives LIBERO's own placement sampler; those layouts come from the same distribution
as the saved ones (docs/trace2vla-review.md, section 2.3), and a seed reproduces its scene
exactly.

The seed ranges below are disjoint from ASPIRE's own partitions (development 51-65, unseen
66-80, held-out 1-50 in seeded mode), so a teacher program is never trained or tested on a
scene it was debugged on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SceneSet:
    name: str
    init_mode: str
    seeds: tuple[int, ...]
    role: str  # test | train | dev
    #: Another suite than the experiment's (e.g. a sibling layout, libero_object_swap_alt1).
    suite: str | None = None

    def to_dict(self) -> dict[str, Any]:
        spec = {"init_mode": self.init_mode, "seeds": compact(self.seeds), "role": self.role}
        if self.suite:
            spec["suite"] = self.suite
        return spec

    def suite_or(self, default: str) -> str:
        """The suite these scenes are drawn from: their own, else the experiment's."""
        return self.suite or default


def compact(seeds: tuple[int, ...] | list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3,7"."""
    parts: list[str] = []
    values = sorted({int(s) for s in seeds})
    start = previous = None
    for value in [*values, None]:
        if start is None:
            start = previous = value
        elif value is not None and value == previous + 1:
            previous = value
        else:
            parts.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
    return ",".join(parts)


DEFAULT_SCENE_SETS: dict[str, dict[str, str]] = {
    # The official test scenes. Never used for training or for choosing a checkpoint.
    "official": {"init_mode": "saved", "seeds": "1-50", "role": "test"},
    # Layouts for teacher demonstrations (training data).
    "train": {"init_mode": "seeded", "seeds": "1001-1050", "role": "train"},
    # Layouts for choosing between checkpoints (the accept / roll-back gate).
    "dev": {"init_mode": "seeded", "seeds": "2001-2010", "role": "dev"},
}


def seeds_of(text: str) -> list[int]:
    """Seeds of a scene set: "1001-1025,2001" -> [1001, ..., 1025, 2001]. Standard library only:
    the dashboard reads scene sets too.
    """
    values: list[int] = []
    for part in str(text).split(","):
        low, dash, high = part.strip().partition("-")
        if not low:
            continue
        values.extend(range(int(low), int(high) + 1) if dash else [int(low)])
    return values


def scene_sets(settings: dict[str, Any]) -> dict[str, SceneSet]:
    declared = {**DEFAULT_SCENE_SETS, **(settings.get("scene_sets") or {})}
    return {
        name: SceneSet(
            name=name,
            init_mode=str(spec["init_mode"]),
            seeds=tuple(seeds_of(str(spec["seeds"]))),
            role=str(spec.get("role", "train")),
            suite=str(spec["suite"]) if spec.get("suite") else None,
        )
        for name, spec in declared.items()
    }


def scene_set(settings: dict[str, Any], name: str) -> SceneSet:
    sets = scene_sets(settings)
    if name not in sets:
        raise KeyError(f"unknown scene set {name!r}; known: {', '.join(sorted(sets))}")
    return sets[name]


def shared_scenes(
    new: SceneSet, others: dict[str, SceneSet], default_suite: str
) -> list[tuple[SceneSet, list[int]]]:
    """The other scene sets that contain some of the same scenes (same suite, init mode and
    seed), with the seeds they share.
    """
    suite = new.suite_or(default_suite)
    found = []
    for other in others.values():
        if other.name == new.name or other.init_mode != new.init_mode:
            continue
        if other.suite_or(default_suite) != suite:
            continue
        common = sorted(set(new.seeds) & set(other.seeds))
        if common:
            found.append((other, common))
    return found
