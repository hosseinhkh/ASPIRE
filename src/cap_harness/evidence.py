"""Visual evidence for a run: what each perception call saw, as pictures and in 3D.

The trace keeps a typed summary of every call. These files keep what a person needs to judge
whether the call was right:

* ``media/overlays/grasps-NNNNNN-camera-<name>.png`` (+ ``.json``): grasp candidates drawn on
  the camera image, ranked by score.
* ``media/geometry/NNNNNN-<kind>.json``: point clouds (down-sampled, millimetres), object boxes
  and grasps, converted to the robot base frame where the frame is known, for a 3D view.

Each file names the ``trace/calls`` folder it came from (``call``), so viewers attach it to the
right call. A value shared by nested calls (a provider's grasps returned unchanged by the
program-level tool) is written once, for the innermost call. Nothing here is visible to the
generated program, and a failure here never fails the run.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.contracts import (
    GraspCandidate,
    GraspSet,
    LocalizationResult,
    ObjectGeometry,
    Observation,
    PointCloud,
)
from cap_harness.geometry import invert_transform, pose_to_matrix

#: Points kept from one object's cloud, and from a whole-scene projection.
MAX_OBJECT_POINTS = 2000
MAX_SCENE_POINTS = 6000
#: A cloud with more points than this is a projection of the whole image (the scene).
SCENE_POINT_COUNT = 50_000
#: Grasp candidates drawn and kept per request.
MAX_GRASPS = 10
#: Largest image width written (larger renders are scaled down, as SAM3 overlays are).
MAX_IMAGE_WIDTH = 960

# Contact-GraspNet's gripper outline in the harness grasp frame (approach +z, fingers along y,
# origin at the gripper base): stem, palm and the two fingers.
_STEM_M = 0.0659
_FINGER_TIP_M = 0.1034
_COLORS = (
    (255, 64, 64),
    (64, 220, 64),
    (64, 128, 255),
    (255, 192, 64),
    (224, 64, 255),
    (64, 224, 224),
    (255, 128, 192),
    (160, 160, 255),
    (200, 200, 64),
    (255, 255, 255),
)


class Evidence:
    """Writes the visual evidence of one run directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = 0
        # The same object reaches several nested spans; hold references so ids stay unique.
        self._seen: deque[object] = deque(maxlen=64)

    def capture(
        self, value: object, *, call: str, name: str, observation: Observation | None
    ) -> None:
        for kind, item in _items(value):
            if any(item is seen for seen in self._seen):
                continue
            self._seen.append(item)
            if kind == "cloud":
                self._cloud(item, call=call, name=name, observation=observation)
            elif kind == "box":
                self._box(item, call=call, name=name, observation=observation)
            elif kind == "grasps":
                self._grasps(item.grasps, call=call, name=name, observation=observation)
            elif kind == "grasp":
                self._grasps([item], call=call, name=name, observation=observation, chosen=True)

    # ------------------------------------------------------------------------ 3D files --
    def _write(self, kind: str, record: dict[str, Any]) -> str:
        self.index += 1
        directory = self.root / "media/geometry"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.index:06d}-{kind}.json"
        path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
        return path.name

    def _cloud(
        self, cloud: PointCloud, *, call: str, name: str, observation: Observation | None
    ) -> None:
        points = np.asarray(cloud.points, dtype=np.float64)
        count = len(points)
        scene = count > SCENE_POINT_COUNT
        keep = MAX_SCENE_POINTS if scene else MAX_OBJECT_POINTS
        rows = np.linspace(0, count - 1, keep).astype(int) if count > keep else np.arange(count)
        transform, frame = _to_base(cloud.frame, observation)
        chosen = points[rows]
        if transform is not None:
            chosen = chosen @ transform[:3, :3].T + transform[:3, 3]
        record: dict[str, Any] = {
            "kind": "cloud",
            "call": call,
            "name": name,
            "frame": frame,
            "point_count": count,
            "scene": scene,
            "points_mm": np.rint(chosen * 1000.0).astype(int).reshape(-1).tolist(),
        }
        if cloud.colors is not None:
            colors = np.asarray(cloud.colors)[rows]
            if np.issubdtype(colors.dtype, np.floating) and float(np.max(colors, initial=0)) <= 1:
                colors = colors * 255.0
            record["colors"] = np.clip(np.rint(colors), 0, 255).astype(int).reshape(-1).tolist()
        self._write("cloud", record)

    def _box(
        self, geometry: ObjectGeometry, *, call: str, name: str, observation: Observation | None
    ) -> None:
        matrix, frame = _pose_in_base(
            pose_to_matrix(geometry.pose), geometry.pose.frame, observation
        )
        self._write(
            "box",
            {
                "kind": "box",
                "call": call,
                "name": name,
                "frame": frame,
                "center": matrix[:3, 3].round(4).tolist(),
                "rotation": matrix[:3, :3].round(5).tolist(),
                "extents": np.asarray(geometry.extents, dtype=float).round(4).tolist(),
                "point_count": geometry.point_count,
            },
        )

    def _grasps(
        self,
        grasps: Any,
        *,
        call: str,
        name: str,
        observation: Observation | None,
        chosen: bool = False,
    ) -> None:
        rows = []
        for rank, grasp in enumerate(list(grasps)[:MAX_GRASPS]):
            matrix, frame = _pose_in_base(pose_to_matrix(grasp.pose), grasp.pose.frame, observation)
            rows.append(
                {
                    "rank": rank,
                    "score": float(grasp.score),
                    "width_m": float(grasp.width_m),
                    "frame": frame,
                    "position": matrix[:3, 3].round(4).tolist(),
                    "rotation": matrix[:3, :3].round(5).tolist(),
                    "matrix": matrix,
                }
            )
        if not rows:
            return
        kind = "grasp" if chosen else "grasps"
        self._write(
            kind,
            {
                "kind": kind,
                "call": call,
                "name": name,
                "grasps": [{k: v for k, v in row.items() if k != "matrix"} for row in rows],
            },
        )
        if observation is not None and not chosen:
            self._grasp_images(rows, call=call, observation=observation)

    # -------------------------------------------------------------------- 2D overlays --
    def _grasp_images(
        self, rows: list[dict[str, Any]], *, call: str, observation: Observation
    ) -> None:
        from PIL import Image, ImageDraw

        base = observation.robot_state.base_frame
        for camera_name, camera in observation.cameras.items():
            if camera.camera_pose.frame != base:
                continue
            camera_from_base = invert_transform(pose_to_matrix(camera.camera_pose))
            rgb = np.asarray(camera.rgb)
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.rint(rgb * 255.0)
            image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy())
            draw = ImageDraw.Draw(image)
            drawn = []
            for row in reversed(rows):  # best grasp drawn last, on top
                outline = _gripper_outline(row["matrix"], row["width_m"])
                pixels = _project(outline, camera_from_base, camera.intrinsics, image.size)
                if pixels is None:
                    continue
                color = _COLORS[row["rank"] % len(_COLORS)]
                for start, end in ((0, 1), (2, 3), (2, 4), (3, 5)):
                    draw.line((pixels[start], pixels[end]), fill=color, width=3)
                if row["rank"] == 0:  # the table lists the rest; labels would pile up
                    label = f"#1 {row['score']:.2f}"
                    x, y = pixels[5]
                    draw.rectangle((x, y - 13, x + 7 * len(label) + 4, y), fill=(0, 0, 0))
                    draw.text((x + 2, y - 12), label, fill=color)
                drawn.append(row["rank"])
            if not drawn:
                continue
            if image.width > MAX_IMAGE_WIDTH:
                image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_WIDTH * image.height // image.width))
            self.index += 1
            directory = self.root / "media/overlays"
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"grasps-{self.index:06d}-camera-{_component(camera_name)}"
            image.save(directory / f"{stem}.png", optimize=True)
            (directory / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "camera_name": camera_name,
                        "call": call,
                        "candidates": [
                            {k: row[k] for k in ("rank", "score", "width_m", "position")}
                            for row in rows
                            if row["rank"] in drawn
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


def _items(value: object) -> Iterator[tuple[str, Any]]:
    """The parts of a call's output that have something to show."""
    if isinstance(value, PointCloud):
        yield "cloud", value
    elif isinstance(value, ObjectGeometry):
        yield "box", value
    elif isinstance(value, GraspSet):
        if value.ok and value.grasps:
            yield "grasps", value
    elif isinstance(value, GraspCandidate):
        yield "grasp", value
    elif isinstance(value, LocalizationResult):
        if value.point_cloud is not None:
            yield "cloud", value.point_cloud
        if value.geometry is not None:
            yield "box", value.geometry


def _to_base(frame: str, observation: Observation | None) -> tuple[np.ndarray | None, str]:
    """The transform from ``frame`` to the robot base, when the observation knows it."""
    if observation is None:
        return None, frame
    base = observation.robot_state.base_frame
    if frame == base:
        return None, base
    for camera in observation.cameras.values():
        if camera.frame == frame and camera.camera_pose.frame == base:
            return pose_to_matrix(camera.camera_pose), base
    return None, frame


def _pose_in_base(
    matrix: np.ndarray, frame: str, observation: Observation | None
) -> tuple[np.ndarray, str]:
    transform, resolved = _to_base(frame, observation)
    return (matrix if transform is None else transform @ matrix), resolved


def _gripper_outline(grasp_from_gripper: np.ndarray, width_m: float) -> np.ndarray:
    """Base, palm centre, palm ends, finger tips (6 points) in the grasp's frame."""
    half = max(width_m, 0.02) / 2.0
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, _STEM_M],
            [0.0, -half, _STEM_M],
            [0.0, half, _STEM_M],
            [0.0, -half, _FINGER_TIP_M],
            [0.0, half, _FINGER_TIP_M],
        ]
    )
    return local @ grasp_from_gripper[:3, :3].T + grasp_from_gripper[:3, 3]


def _project(
    points: np.ndarray, camera_from_base: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int]
) -> list[tuple[float, float]] | None:
    """Pixel coordinates of base-frame points, or None when any is behind or far off image."""
    in_camera = points @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
    if bool(np.any(in_camera[:, 2] <= 0.02)):
        return None
    k = np.asarray(intrinsics, dtype=float)
    u = k[0, 0] * in_camera[:, 0] / in_camera[:, 2] + k[0, 2]
    v = k[1, 1] * in_camera[:, 1] / in_camera[:, 2] + k[1, 2]
    width, height = size
    if bool(np.all((u < 0) | (u >= width) | (v < 0) | (v >= height))):
        return None
    return [(float(a), float(b)) for a, b in zip(u, v, strict=True)]


def _component(value: object) -> str:
    import re

    return (re.sub(r"[^a-z0-9_.-]+", "-", str(value).lower()).strip("-.") or "unnamed")[:96]


__all__ = ["Evidence"]
