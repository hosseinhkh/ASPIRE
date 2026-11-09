# Harness fixes: why programs failed for reasons outside the program

A Claude Opus 4.6 agent session on LIBERO-Pro `libero_object_swap` task 0 (pick up the alphabet
soup, place it in the basket) failed every version: `dev_v01` on seeds 51-53 and `v02`-`v04`
on seed 51. The recordings (videos, every call, the SAM3 masks and the point clouds) show four
causes. Two are harness defects that would trip any program; two are what the agent did with
the information the harness gave it.

| Version | What happened | Cause |
|---|---|---|
| v01 (3 seeds) | Stopped at step 146: "pre-grasp failed: final trajectory waypoint did not converge" | **Harness: motion timing** |
| v02 | Picked up the **milk carton** and put it in the basket | SAM3 does not know "alphabet soup" (best score 0.11, the milk); the program used the guess |
| v03 | Closed the gripper on air 17 cm from the can | **Harness: point cloud and box** (box 30 cm long) |
| v04 | Chose the right can, closed on air 5.5 cm from it | **Harness: point cloud and box** (box 40 cm long) |

GPU load was not a cause. Simulated time advances one control step at a time whatever the
wall clock does, no model-service request failed or timed out, and the latencies were normal
(SAM3 about 0.5 s, Contact-GraspNet 2.3 s, cuRobo 1.1 s per request).

## 1. Motion: the plan is six times faster than the arm

LIBERO and Robosuite use robosuite's JOINT_POSITION controller: proportional gain 50,
critically damped, and a goal that may move at most 0.05 rad per 20 Hz control step. The motion
planners time trajectories by that limit, 0.05 rad per step. But with the goal clipped 0.05 rad
ahead, the arm settles at

```text
v = (kp / kd) * output_max / (1 + (kp / kd) * dt / 2) = 0.16 rad/s = 0.008 rad per step
```

(measured in the v01 recording: 0.0079). So on the first long move the arm fell up to 1.07 rad
behind its plan, every joint moving at the same capped speed. The hand left the planned path,
dropping to 9 cm above the table on a move to 21 cm, and after the plan ended the fixed 120
settle steps ran out while the arm was still moving steadily. Nothing blocked it; it was just
slow. The move failed, and the program with it.

**Fix** (`src/cap_harness/trajectory_timing.py`, both adapters): what real robot drivers do,
time-scaling to the actuator. Each adapter derives the speed its joint controllers can follow
from their own gains (85 % of the speed above, 0.0069 rad per step for these gains, so other
robots and gains get their own limit), re-times every trajectory (interpolation or cuRobo) to
it, and keeps settling while the error still falls (up to 600 steps; it stops after 25 steps
without progress, for an arm blocked by contact). Replaying v01's exact move on the same scene:

| | Published | Fixed |
|---|---|---|
| Result | fails after 146 steps, 0.164 rad short | reaches the target in 215 steps |
| Lowest hand height on the way | 0.093 m | 0.216 m (the planned path) |

## 2. Perception: a few depth-edge points decide the box

`mask_to_point_cloud` back-projected every masked pixel. Pixels on a mask's border have depths
between the object and what is behind it (renderers and depth cameras both do this), so they
land in mid-air or on the floor behind the object. `estimate_geometry` then took the box's size
and centre from the extreme points. In v04, 8 such points out of 1,475 stretched the soup can
(7 cm) into a 40 cm box whose centre was 5.5 cm off; the gripper closed on air. A container has
a second problem: from one camera most points lie on its near wall, so the median of the points
(what the agents' own code used) sits about 7 cm toward the camera, near the front rim.

**Fix** (`src/cap_harness/geometry.py`):

* `mask_to_point_cloud` leaves out mask pixels whose 3x3 neighbourhood jumps in depth by more
  than 1 cm or 2 % of the depth (keeping them for a thin object that would otherwise vanish);
* `estimate_geometry`, from 50 points on, drops points far from the bulk (a median-distance
  inlier test) and takes extents between the 1st and 99th percentile along each axis.

Replaying the recorded masks through the harness:

| Object (seed 51) | Published: centre error, box | Fixed: centre error, box |
|---|---|---|
| Alphabet soup (v04's mask) | 5.5 cm, 40 x 11 x 11 cm | 1.7 cm, 9.5 x 5.9 x 5.4 cm |
| Basket | 6.5 cm, 31 x 25 x 21 cm | 1.0 cm, 19 x 17 x 13 cm |
| Milk (v02's mask) | 4.6 cm, 17 x 14 x 7 cm | 1.4 cm, 14 x 6 x 4 cm |

The remaining 1-2 cm is the camera seeing only the front half of a can, well within the grasp.

## 3. Ambiguous detections and missed grasps were silent

SAM3 does not know the brand name "alphabet soup". Its best mask (score 0.111) was the milk
carton, with four other guesses between 0.06 and 0.09; `localize_object` used the best one
without saying how unsure it was, and v02 placed the milk. "soup can" finds both cans with high
scores (0.77 tomato sauce, 0.73 alphabet soup), which v04 then told apart correctly by size.
In v03 and v04 the gripper closed fully (0.049 of its opening, nothing in it) and the programs
went on to "place" nothing.

**Fix** (`src/cap_harness/api.py`): the API now says so.

* `localize_object(...).diagnostics`: `segmentation_score`, `runner_up_score`,
  `low_confidence` (below 0.3), `ambiguous` (runner-up within 0.1) and the top five `candidates`
  with their boxes.
* `close_gripper().diagnostics`: `gripper_position` (the measured opening) and `closed_empty`
  (below 0.1: the grasp missed).

The dashboard flags low-confidence SAM3 calls on each episode, and its 3D view shows the
point clouds and boxes, so all of the above is visible without reading JSON.

## 4. What the fixes do not change

* Which object a name refers to is still the program's job. The prompt registry in the released
  skill library is empty; the paper's campaigns grew it task by task. The full pipeline
  (`scripts/gpu/demo.sh campaign`) lets verified prompts such as "soup can, the smaller one"
  accumulate there.
* The quick agent (`session.sh --llm`) has 3 development seeds and at most 4 versions. Each of
  the causes above cost this session one version; the earlier session that reached 3/3 found a
  way around them by its last version. With the fixes, a version is no longer lost to 1 and 2.

## Comparing with the paper

The fixes are on by default. `CAP_HARNESS_COMPAT=published` restores the published release's
behaviour for all of them (motion timing, point clouds, boxes), and
`scripts/gpu/demo.sh campaign --paper` sets it, so paper comparisons stay like for like. The
diagnostics in section 3 are added in both modes. Each session records which mode it ran in
(`harness` in `session.json`), and the dashboard's Compare page shows it.
