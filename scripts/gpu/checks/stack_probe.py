# Stack probe (a CaP program, run by scripts/gpu/check_stack.sh on Robosuite cube_lifting).
#
# Calls every model service through the public API, without needing SAM3:
#   PyRoki IK + interpolated motion, cuRobo IK + c-space planning + integrated pose planning,
#   SAM3 text segmentation of the cube, and Contact-GraspNet on SAM3's best cube mask (on an
#   image-centre mask when SAM3 is not available).
# Each check records ok / failure code; check_stack.sh decides what was expected from which
# services are running. The program never raises, so one failure cannot hide the others.
report = {"checks": {}, "errors": {}, "details": {}}


def record(name, outcome):
    ok = bool(outcome.ok)
    report["checks"][name] = ok
    if not ok and outcome.error is not None:
        report["errors"][name] = (
            str(outcome.error.code.value) + ": " + str(outcome.error.message)[:160]
        )
    return ok


state = get_robot_state()
frame = state.base_frame
start = state.end_effector_poses["primary"]
up = Pose(
    (start.position[0], start.position[1], start.position[2] + 0.05),
    start.quaternion_wxyz,
    frame,
)

# PyRoki: IK, then the default strategy (PyRoki IK + joint interpolation) executed in the sim.
record("pyroki_ik", solve_ik(up, backend="pyroki"))
moved = move_to_pose(up, tolerance=0.02, max_steps=200)
record("move_to_pose_pyroki", moved)
if moved.ok:
    report["details"]["pyroki_position_error_m"] = moved.final_errors["primary"]

# cuRobo: IK, a collision-aware c-space plan executed back down, and an integrated pose plan.
record("curobo_ik", solve_ik(start, backend="curobo"))
back = move_to_pose(
    start, tolerance=0.02, max_steps=600, strategy=MotionStrategy(trajectory_planner="curobo")
)
record("move_to_pose_curobo", back)
record(
    "curobo_integrated_plan",
    plan_motion(up, strategy=MotionStrategy(pose_planner="curobo-integrated")),
)

# SAM3 first: its cube mask is what a real program hands to Contact-GraspNet.
segmentation = segment_text("agentview", "red cube")
mask = None
if record("sam3_segment_text", segmentation):
    report["details"]["sam3_masks"] = len(segmentation.segmentations)
    if segmentation.segmentations:
        best = segmentation.segmentations[0]
        for candidate in segmentation.segmentations:
            if float(candidate.score) > float(best.score):
                best = candidate
        mask = best.mask
        report["details"]["grasp_mask"] = "sam3 red cube, score " + str(round(float(best.score), 3))

# Contact-GraspNet. Without SAM3, use the centre third of the image (the cube and table around
# it): a large, mostly flat region, so expect few grasps and several service-side retries.
if mask is None:
    camera = get_observation("agentview").cameras["agentview"]
    mask = camera.depth_m < 0.0
    rows = len(mask)
    cols = len(mask[0])
    mask[rows // 3 : 2 * rows // 3, cols // 3 : 2 * cols // 3] = True
    report["details"]["grasp_mask"] = "image centre third (no SAM3)"
grasps = generate_grasps("agentview", mask, max_candidates=5)
if record("contact_graspnet", grasps):
    report["details"]["grasp_candidates"] = len(grasps.grasps)

result = report
