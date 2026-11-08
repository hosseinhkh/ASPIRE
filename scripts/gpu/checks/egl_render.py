#!/usr/bin/env python3
"""Render one MuJoCo frame with the current MUJOCO_GL and report which GPU/driver drew it."""

from __future__ import annotations

import os
import sys
import time


def main() -> int:
    import mujoco

    xml = (
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>'
        '<camera name="c" pos="0 -1 0.3" xyaxes="1 0 0 0 0.3 1"/></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    started = time.perf_counter()
    renderer = mujoco.Renderer(model, 256, 256)
    renderer.update_scene(data, "c")
    image = renderer.render()
    elapsed_ms = (time.perf_counter() - started) * 1000
    gl_renderer = "unknown"
    try:
        context = getattr(renderer, "_gl_context", None)
        if context is not None:
            context.make_current()  # render() may leave no context current
        from OpenGL import GL

        raw = GL.glGetString(GL.GL_RENDERER)
        if raw:
            gl_renderer = raw.decode()
    except Exception as exc:  # report, never fail on the vendor query alone
        gl_renderer = f"unknown ({type(exc).__name__})"
    if gl_renderer.startswith("unknown") and os.environ.get("MUJOCO_GL") == "egl":
        try:  # fall back to the EGL vendor of the display MuJoCo opened
            import mujoco.egl as mujoco_egl
            from OpenGL import EGL

            vendor = EGL.eglQueryString(mujoco_egl.EGL_DISPLAY, EGL.EGL_VENDOR)
            if vendor:
                gl_renderer = f"EGL vendor {vendor.decode()}"
        except Exception:
            pass
    backend = os.environ.get("MUJOCO_GL", "?")
    if float(image.mean()) <= 0.0:
        print(f"FAIL MUJOCO_GL={backend}: rendered an all-black frame ({gl_renderer})")
        return 1
    lowered = gl_renderer.lower()
    software = any(word in lowered for word in ("llvmpipe", "softpipe", "swrast", "vendor mesa"))
    status = "WARN" if software else "PASS"
    note = " (software rendering: slow; NVIDIA EGL not in use)" if software else ""
    print(
        f"{status} MUJOCO_GL={backend}, renderer: {gl_renderer}, first frame {elapsed_ms:.0f} ms{note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
