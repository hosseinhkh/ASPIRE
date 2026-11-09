"""One switch for reproducing the published ASPIRE release exactly.

The harness fixes several behaviours of the published release that make programs fail for
reasons outside the program (see docs/harness-fixes.md):

* motion: trajectories are re-timed to the speed the joint controller can follow, and settling
  continues while the arm still converges (cap_harness.trajectory_timing);
* perception: mask_to_point_cloud drops depth-edge pixels (mixed foreground/background depth
  at object borders), and estimate_geometry sizes boxes robustly against stray points
  (cap_harness.geometry).

``CAP_HARNESS_COMPAT=published`` turns all of them off, for runs that must be compared with the
paper's numbers. ``scripts/gpu/demo.sh campaign --paper`` sets it.
"""

from __future__ import annotations

import os

PUBLISHED = "published"


def published_behavior() -> bool:
    """Whether to behave exactly like the published release (CAP_HARNESS_COMPAT=published)."""
    return os.environ.get("CAP_HARNESS_COMPAT", "").strip().lower() == PUBLISHED


__all__ = ["PUBLISHED", "published_behavior"]
