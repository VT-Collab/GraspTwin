"""Pixels <-> rays for the Orbbec Femto Mega colour camera.

The one place in the pipeline that knows the colour lens is not a pinhole.
Everything here works in the ROTATED frame — the frame `capture_rgbd()` hands
out, after the 180deg rotation that undoes the upside-down sensor mounting —
so K/DIST pair directly with config.CAMERA_FX..CAMERA_CY.

Why this exists
---------------
`AlignFilter(COLOR_STREAM)` registers the depth map into the *distorted* colour
image (same contract as K4A's depth_image_to_color_camera), and the colour
stream is raw MJPG that the device never rectifies. So `depth[v, u]` really is
the depth of what you see at `rgb[v, u]` — but the ray through pixel (u, v) is
NOT `((u - cx)/fx, (v - cy)/fy, 1)`. Using that pinhole ray anyway (which the
pipeline did until this module landed) leaves the cloud correct at the
principal point and increasingly wrong outwards: ~14 px at the left edge of the
valid depth region, ~18 px at the right, ~20 px in the corners — 12 to 17 mm of
lateral error at 1 m.

Use
---
    from semantic_grasp import camera_model as cam

    pts = cam.unproject(us, vs, z_metres)   # (N,) pixel cols/rows + depth -> (N,3)
    uv  = cam.project(pts)                  # (N,3) camera-frame points -> (N,2) px

`unproject` is a lookup into a full-frame ray table built once on first use
(~145 ms, 16 MB) and kept for the life of the process; `project` runs the
forward distortion model, so the two are exact inverses to ~1e-8 px.

Both directions live here on purpose: a cloud back-projected with `unproject`
can only be indexed back into image space with `project`. Mixing one with the
naive pinhole reintroduces the very error this module removes.
"""
import cv2
import numpy as np

from .config import (
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_DIST_CV,
)

# Intrinsics + distortion of the rotated frame, in the layout OpenCV wants.
# Hand these to solvePnP/projectPoints/undistortPoints as a pair — a K without
# its DIST is what put the pipeline in this mess to begin with.
K = np.array([[CAMERA_FX, 0.0, CAMERA_CX],
              [0.0, CAMERA_FY, CAMERA_CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.asarray(CAMERA_DIST_CV, dtype=np.float64)

_rays = None


def rays():
    """(H, W, 2) table of undistorted normalized rays, one per integer pixel.

    `rays()[v, u]` is (x/z, y/z) of the direction light entering pixel (u, v)
    actually came from. Built on first call and cached for the process.
    """
    global _rays
    if _rays is None:
        grid = np.stack(np.meshgrid(
            np.arange(CAMERA_WIDTH, dtype=np.float32),
            np.arange(CAMERA_HEIGHT, dtype=np.float32),
        ), axis=-1).reshape(-1, 1, 2)
        _rays = (cv2.undistortPoints(grid, K, DIST)
                 .reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 2)
                 .astype(np.float32))
    return _rays


def unproject(u, v, z):
    """Integer pixel coords + depth (metres) -> (N, 3) camera-frame points.

    CV convention: x right, y down, z forward. `z` is the depth map's value,
    i.e. distance along the optical axis, not along the ray — which is what the
    Orbbec writes and what makes the ray table a plain multiply.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    z = np.asarray(z, dtype=np.float32)
    n = rays()[v.astype(np.intp), u.astype(np.intp)]
    return np.stack([n[:, 0] * z, n[:, 1] * z, z], axis=1)


def unproject_depth(depth_m, mask=None):
    """Depth map in METRES -> (N, 3) camera-frame points for every valid pixel.

    Invalid (non-positive) depth is dropped, so N < H*W and the result carries
    no pixel grid. `mask` optionally restricts it to one object's pixels.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32).squeeze()
    if depth_m.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        raise ValueError(
            f"depth map is {depth_m.shape[1]}x{depth_m.shape[0]} but the ray "
            f"table is built for {CAMERA_WIDTH}x{CAMERA_HEIGHT}. The intrinsics "
            f"in config.py are per-resolution — a resized or cropped frame needs "
            f"its own calibration, not a rescaled one."
        )
    valid = depth_m > 0
    if mask is not None:
        valid &= np.asarray(mask) > 0
    vs, us = np.nonzero(valid)
    return unproject(us, vs, depth_m[vs, us])


def project(pts):
    """(N, 3) camera-frame points (metres) -> (N, 2) float pixel coords.

    Applies the distortion, so this is the exact inverse of `unproject` and the
    right way to ask "which pixel did this point come from". Points must have
    z > 0; behind-camera points come back meaningless (OpenCV does not check).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, DIST)
    return uv.reshape(-1, 2)


def project_pinhole(pts):
    """(N, 3) -> (N, 2) with distortion IGNORED — for legacy clouds only.

    Point clouds captured before the ray table existed were back-projected with
    the naive pinhole, so this is the projection that round-trips them back to
    the pixels they came from. It is the wrong camera model; it is only right
    for undoing itself. New captures go through `project`.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return np.stack([CAMERA_FX * pts[:, 0] / pts[:, 2] + CAMERA_CX,
                     CAMERA_FY * pts[:, 1] / pts[:, 2] + CAMERA_CY], axis=1)
