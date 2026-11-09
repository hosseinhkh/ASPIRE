"""Visual evidence: grasp overlays, point clouds and boxes in the base frame (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cap_harness.artifacts import RunRecorder
from cap_harness.contracts import (
    CameraObservation,
    GraspCandidate,
    GraspSet,
    LocalizationResult,
    ObjectGeometry,
    Observation,
    PointCloud,
    Pose,
    RobotState,
)
from cap_harness.evidence import Evidence
from cap_harness.geometry import matrix_to_pose

BASE = "robot_base"


def _top_camera_observation() -> Observation:
    """A 100x80 camera 1 m above the base origin, looking straight down."""
    # Image frame: +x right, +y down, +z forward (down). In the base frame: x, -y, -z.
    camera_pose = matrix_to_pose(
        np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]]),
        frame=BASE,
    )
    camera = CameraObservation(
        rgb=np.full((80, 100, 3), 90, dtype=np.uint8),
        depth_m=np.ones((80, 100)),
        intrinsics=np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]]),
        frame="camera/top",
        camera_pose=camera_pose,
    )
    hand = Pose(np.zeros(3), np.array([1.0, 0, 0, 0]), BASE)
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=hand,
        gripper_positions=1.0,
        base_frame=BASE,
    )
    return Observation(cameras={"top": camera}, robot_state=state)


def _top_down_grasp(x: float, y: float, z: float, score: float) -> GraspCandidate:
    # Approach +z points down in the base frame; fingers along base x.
    matrix = np.array([[0, 1.0, 0, x], [1.0, 0, 0, y], [0, 0, -1.0, z], [0, 0, 0, 1.0]])
    return GraspCandidate(pose=matrix_to_pose(matrix, frame=BASE), score=score, width_m=0.08)


def _read(root: Path, kind: str) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((root / "media/geometry").glob(f"*-{kind}.json"))
    ]


def test_camera_frame_cloud_is_stored_in_the_base_frame_and_downsampled(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path)
    points = np.tile([[0.0, 0.0, 0.5]], (5000, 1))  # 0.5 m in front of the camera
    evidence.capture(
        PointCloud(points=points, frame="camera/top"),
        call="call-000003-mask-to-point-cloud",
        name="mask_to_point_cloud",
        observation=_top_camera_observation(),
    )
    (cloud,) = _read(tmp_path, "cloud")
    assert cloud["frame"] == BASE and cloud["point_count"] == 5000 and not cloud["scene"]
    assert len(cloud["points_mm"]) == 3 * 2000  # MAX_OBJECT_POINTS
    assert cloud["points_mm"][:3] == [0, 0, 500]  # 1 m camera height - 0.5 m
    assert cloud["call"] == "call-000003-mask-to-point-cloud"


def test_grasps_are_written_once_and_drawn_where_the_camera_sees_them(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path)
    observation = _top_camera_observation()
    grasps = GraspSet(
        ok=True,
        grasps=(_top_down_grasp(0.0, 0.0, 0.4, 0.9), _top_down_grasp(0.05, 0.0, 0.4, 0.5)),
    )
    for call, name in (("call-000007-cgn", "contact_graspnet"), ("call-000006-gg", "generate")):
        evidence.capture(grasps, call=call, name=name, observation=observation)
    (record,) = _read(tmp_path, "grasps")  # the program span returned the same object
    assert record["call"] == "call-000007-cgn"
    assert [g["score"] for g in record["grasps"]] == [0.9, 0.5]
    assert record["grasps"][0]["position"] == [0.0, 0.0, 0.4]
    overlay = json.loads(next((tmp_path / "media/overlays").glob("grasps-*.json")).read_text())
    assert overlay["camera_name"] == "top" and [c["rank"] for c in overlay["candidates"]] == [0, 1]
    assert (tmp_path / "media/overlays" / "grasps-000002-camera-top.png").exists()


def test_localization_yields_cloud_and_box_and_chosen_grasp_is_marked(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path)
    observation = _top_camera_observation()
    geometry = ObjectGeometry(
        pose=Pose(np.array([0.1, 0.2, 0.05]), np.array([1.0, 0, 0, 0]), BASE),
        extents=np.array([0.06, 0.06, 0.1]),
        point_count=10,
    )
    cloud = PointCloud(points=np.full((10, 3), 0.1), frame=BASE)
    result = LocalizationResult(ok=True, geometry=geometry, point_cloud=cloud)
    evidence.capture(result, call="call-000002-localize", name="localize", observation=observation)
    evidence.capture(
        _top_down_grasp(0.1, 0.2, 0.2, 0.7),
        call="call-000009-select-grasp",
        name="select_grasp",
        observation=observation,
    )
    (box,) = _read(tmp_path, "box")
    assert box["center"] == [0.1, 0.2, 0.05] and box["extents"] == [0.06, 0.06, 0.1]
    assert len(_read(tmp_path, "cloud")) == 1
    (chosen,) = _read(tmp_path, "grasp")
    assert chosen["grasps"][0]["score"] == 0.7
    # A chosen grasp is not drawn again on the images: the candidates already were.
    assert not list((tmp_path / "media/overlays").glob("grasps-*.png"))


def test_a_failure_while_recording_evidence_never_fails_the_run(tmp_path: Path) -> None:
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="libero-pro",
        suite="suite",
        task_id=0,
        task_name="task",
        seed=1,
        max_steps=1,
    )

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    recorder.evidence.capture = broken  # type: ignore[method-assign]
    with recorder.span("localize_object", category="program", inputs={}) as span:
        span.output(PointCloud(points=np.zeros((3, 3)), frame=BASE))
    log = (recorder.root / "logs/evidence-errors.txt").read_text()
    assert "RuntimeError: disk full" in log and "localize-object" in log
