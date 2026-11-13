"""Teacher vs student before vs student after, on the same scenes.

Reads what the stages wrote (teacher results, student evaluation runs, demo and dataset
summaries, training runs) and writes ``reports/summary.md`` and ``reports/summary.json``.
An evaluation belongs to a suite; the experiment's own suite gives the main table, any other
suite (for example the standard ``libero_object``) a retention table.
"""

from __future__ import annotations

from typing import Any

from cap_harness.trace2vla import teacher
from cap_harness.trace2vla.convert import demo_summary
from cap_harness.trace2vla.store import (
    Experiment,
    atomic_write,
    is_probe,
    now_iso,
    read_json,
    write_json,
)
from cap_harness.vla import students
from cap_harness.vla.benchmark import family_text


def student_of(*records: dict[str, Any]) -> str:
    """The student the first record that says one names; SmolVLA for records from before."""
    for record in records:
        key, how = students.infer(record)
        if how != students.LEGACY:
            return key
    return students.DEFAULT_STUDENT


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def span(steps: dict[str, Any] | None) -> str:
    """Format a step-count summary as min / mean / max."""
    return "-" if not steps else f"{steps['min']} / {steps['mean']:.0f} / {steps['max']}"


def number(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def evaluations(experiment: Experiment) -> list[dict[str, Any]]:
    """Every student evaluation run in the experiment, oldest first."""
    runs = []
    for folder in sorted(
        (experiment.root / "evals").glob("*/run.json"), key=lambda p: p.stat().st_mtime
    ):
        root = folder.parent
        meta = read_json(root / "eval.json") or {}
        run = read_json(folder) or {}
        settings = run.get("settings") or {}
        results = read_json(root / "results.json") or {}
        progress = read_json(root / "progress.json") or {}
        student = student_of(meta, run)
        runs.append(
            {
                "name": root.name,
                "path": experiment.rel(root),
                "student": student,
                "student_label": students.label(student),
                "suite": settings.get("suite"),
                "checkpoint": meta.get("checkpoint_label", settings.get("policy")),
                "training_run": meta.get("training_run"),
                "checkpoint_step": meta.get("checkpoint_step"),
                "probe": is_probe(root.name, meta),
                "scene_set": meta.get("scene_set"),
                "status": progress.get("status"),
                "done": progress.get("done"),
                "planned": progress.get("planned"),
                "mean": results.get("mean"),
                "successes": results.get("successes"),
                "episodes": results.get("episodes"),
                "ci95": results.get("ci95"),
                "failure_families": results.get("failure_families") or {},
                "tasks": {int(t["task_id"]): t for t in results.get("tasks", [])},
            }
        )
    return runs


def split_evaluations(
    evaluations: list[dict[str, Any]], suite: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(teacher-vs-student columns, other evaluations, checkpoints during training).

    Next to the teacher: full evaluations of the experiment's suite on the official scenes, the
    scenes the teacher was measured on (a chosen step evaluated in full included). The training
    curve: every evaluation of a step, sorted by run, suite, scene set and step. The short
    probes ``eval-steps`` runs appear only on the curve.
    """

    def official(e: dict[str, Any]) -> bool:
        return e["suite"] == suite and (e.get("scene_set") or "official") == "official"

    curve = [e for e in evaluations if e.get("checkpoint_step") is not None]
    main = [e for e in evaluations if official(e) and not e.get("probe")]
    other = [
        e
        for e in evaluations
        if not official(e) and e.get("checkpoint_step") is None and not e.get("probe")
    ]
    curve.sort(
        key=lambda e: (
            str(e.get("training_run")),
            str(e["suite"]),
            str(e.get("scene_set")),
            int(e["checkpoint_step"]),
        )
    )
    return main, other, curve


def trainings(experiment: Experiment) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(
        (experiment.root / "training").glob("*/run.json"), key=lambda p: p.stat().st_mtime
    ):
        record = read_json(path) or {}
        progress = read_json(path.parent / "progress.json") or {}
        student = student_of(record)
        runs.append(
            {
                **record,
                "student": student,
                "student_label": students.label(student),
                "progress": progress,
                "path": experiment.rel(path.parent),
            }
        )
    return runs


def datasets(experiment: Experiment) -> list[dict[str, Any]]:
    found = []
    for path in sorted(
        (experiment.root / "datasets").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime
    ):
        manifest = read_json(path) or {}
        student = student_of(manifest)
        found.append(
            {k: manifest.get(k) for k in ("name", "frames", "per_task", "created")}
            | {
                "student": student,
                "student_label": students.label(student),
                "derived_from": (manifest.get("derived_from") or {}).get("name"),
                "episodes": len(manifest.get("episodes", [])),
                "rejected": len(manifest.get("rejected", [])),
                "path": experiment.rel(path.parent),
            }
        )
    return found


def demo_sets(experiment: Experiment) -> list[dict[str, Any]]:
    return [
        demo_summary(experiment, p.name)
        for p in sorted((experiment.root / "demos").glob("*"))
        if p.is_dir()
    ]


def build(experiment: Experiment) -> dict[str, Any]:
    teacher_sets = {
        path.parent.name: teacher.results(experiment, path.parent.name)
        for path in sorted((experiment.root / "teacher").glob("*/results.json"))
    }
    return {
        "experiment": experiment.root.name,
        "suite": experiment.suite,
        "tasks": experiment.tasks,
        "created": now_iso(),
        "teacher": teacher_sets,
        "evaluations": evaluations(experiment),
        "demos": demo_sets(experiment),
        "datasets": datasets(experiment),
        "training": trainings(experiment),
    }


def to_markdown(report: dict[str, Any]) -> str:
    suite = report["suite"]
    lines = [
        f"# {report['experiment']}: teacher and student on {suite}",
        "",
        f"_Generated {report['created']}._",
        "",
    ]
    official = report["teacher"].get("official")
    main, other, curve = split_evaluations(report["evaluations"], suite)

    lines += ["## Same scenes: teacher vs student", ""]
    header = ["task", "teacher (official scenes)"] + [
        f"{e['student_label']}: {e['name']}" for e in main
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    teacher_tasks = {t["task_id"]: t for t in (official or {}).get("tasks", [])}
    for task_id in report["tasks"]:
        row = [f"{suite}:{task_id}"]
        t = teacher_tasks.get(task_id)
        row.append(
            f"{t['successes']}/{t['episodes']} ({percent(t['success_rate'])})" if t else "not run"
        )
        for e in main:
            s = e["tasks"].get(task_id)
            row.append(
                f"{s['successes']}/{s['episodes']} ({percent(s['success_rate'])})" if s else "-"
            )
        lines.append("| " + " | ".join(row) + " |")
    row = ["**mean**", percent((official or {}).get("mean")) if official else "not run"]
    row += [
        f"**{percent(e['mean'])}**"
        + ("" if e["status"] == "finished" else f" ({e['status']}, {e['done']}/{e['planned']})")
        for e in main
    ]
    lines += ["| " + " | ".join(row) + " |", ""]
    for e in main:
        lines.append(
            f"- `{e['name']}` ({e['student_label']}): checkpoint {e['checkpoint']}; failures: {family_text(e['failure_families']) or 'none graded'}"
        )
    lines.append("")

    if other:
        lines += [
            "## Retention and other scene sets",
            "",
            "| evaluation | student | suite | scene set | checkpoint | success | 95% interval |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in other:
            interval = f"{percent(e['ci95'][0])}-{percent(e['ci95'][1])}" if e.get("ci95") else "-"
            lines.append(
                f"| {e['name']} | {e['student_label']} | {e['suite']} | {e.get('scene_set') or '-'} | {e['checkpoint']} | {e['successes']}/{e['episodes']} ({percent(e['mean'])}) | {interval} |"
            )
        lines.append("")
    if curve:
        lines += [
            "## Checkpoints during training",
            "",
            "Each checkpoint a training run saved (`train --save-freq`), evaluated on its own "
            "(`t2v.sh eval-steps`). Choose a step on the dev scenes, then report the official ones.",
            "",
            "| training run | student | suite | scene set | step | success | failures |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in curve:
            lines.append(
                f"| {e['training_run']} | {e['student_label']} | {e['suite']} | {e.get('scene_set') or '-'} | {e['checkpoint_step']} | "
                f"{e['successes']}/{e['episodes']} ({percent(e['mean'])}) | {family_text(e['failure_families']) or '-'} |"
            )
        lines.append("")

    lines += ["## Teacher", ""]
    for name, summary in report["teacher"].items():
        lines += [
            f"**Scene set `{name}`**: {summary['successes']}/{summary['episodes']} episodes solved.",
            "",
            "| task | success | steps when solved (min / mean / max) | failures ended as | failure families | graded |",
            "|---|---|---|---|---|---|",
        ]
        for t in summary["tasks"]:
            steps = t["steps_when_successful"]
            lines.append(
                f"| {t['task_id']} | {t['successes']}/{t['episodes']} ({percent(t['success_rate'])}) | "
                f"{span(steps)} | "
                f"{family_text(t['terminations']) or '-'} | {family_text(t['failure_families']) or '-'} | {t['graded']}/{t['episodes']} |"
            )
        lines.append("")

    if report["demos"]:
        lines += [
            "## Demonstrations (teacher successes re-executed in the student's control)",
            "",
            "| demo set | task | kept | rejected | steps (min / mean / max) | teacher steps (mean) | idle frames dropped (mean) | kept all idle |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for demo in report["demos"]:
            for t in demo["tasks"]:
                steps = t["steps"]
                lines.append(
                    f"| {demo['name']} | {t['task_id']} | {t['accepted']} | {t['rejected']} {('(' + ', '.join(t['rejected_reasons']) + ')') if t['rejected'] else ''} | "
                    f"{span(steps)} | "
                    f"{number(t['teacher_steps_mean'])} | "
                    f"{number(t['idle_dropped_mean'], 1)} | {t['kept_all_idle']} |"
                )
        lines.append("")
    if report["datasets"]:
        lines += [
            "## Datasets",
            "",
            "| dataset | student | episodes | frames | per task | rejected at build |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {d['name']}{' (from ' + d['derived_from'] + ')' if d.get('derived_from') else ''} | {d['student_label']} | {d['episodes']} | {d['frames']} | {d['per_task']} | {d['rejected']} |"
            for d in report["datasets"]
        ]
        lines.append("")
    if report["training"]:
        lines += [
            "## Training",
            "",
            "| run | student | method | steps | batch | lr | dataset | status | final loss | time |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in report["training"]:
            s = run.get("settings", {})
            wall = run.get("wall_s")
            minutes = "-" if wall is None else f"{wall / 60:.0f} min"
            lines.append(
                f"| {run.get('name')} | {run['student_label']} | {s.get('method')} | {run['progress'].get('step', '-')}/{s.get('steps')} | {s.get('batch_size')} | {s.get('lr')} | "
                f"{str(run.get('dataset', '')).rsplit('/', 1)[-1]} ({run.get('dataset_episodes')} episodes) | {run.get('status', run['progress'].get('status'))} | "
                f"{run.get('final_loss', '-')} | {minutes} |"
            )
        lines.append("")
    return "\n".join(lines)


def write(experiment: Experiment) -> dict[str, Any]:
    report = build(experiment)
    write_json(experiment.root / "reports" / "summary.json", report)
    text = to_markdown(report)
    atomic_write(experiment.root / "reports" / "summary.md", text)
    return {"report": report, "markdown": text}
