"""Perception: camera capture, object detection, segmentation, 3D reconstruction,
and the VLM grasp-proposal stages — semantic first (WHERE to grasp and HOW to
approach, in words), then the numeric stages those phrases condition (metric
xyz, arrow-vote rpy, scale-bar gripper width)."""
import io
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import yaml
import zmq
from PIL import Image

from .. import camera_model
from ..camera import OrbbecClient
from ..config import (
    PERCEPTION, USE_LIVE_CAMERA, USE_LIVE_ER, USE_LIVE_SAM3D, TEST_DIR,
    GRIPPER_MAX_WIDTH_M,
)

# NOTE: .rendering (pyrender/EGL) and google.genai are imported lazily inside the
# functions that need them, so scripts that only want capture_rgbd keep working in
# environments without a GPU/EGL or with an old typing_extensions (Isaac Sim's env).

_PROMPT_DIR = Path(__file__).parents[2] / "config"   # repo-root config/ (prompt yamls)

# langsam segmentation server (segment_objects).
LANGSAM_TIMEOUT_S = 300           # cap the wait so a down server errors, not hangs
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, int(LANGSAM_TIMEOUT_S * 1000))
socket.setsockopt(zmq.LINGER, 0)
socket.connect(PERCEPTION)

# sam3d reconstruction server on the HPC node, reached through the SSH tunnel:
#   ssh -J <user>@<login-node> -L 5561:localhost:5561 <user>@<gpu-node>
# ZMQ REQ/REP in the camera_server style: multipart [b"predict", png, ply,
# masks_zip] in, [b"ok", zip_bytes] out (b"error" on failure; b"health" -> b"ok").
SAM3D_ADDR = "tcp://localhost:5561"
SAM3D_TIMEOUT_S = 1200


def _load_prompt(filename):
    with open(_PROMPT_DIR / filename) as f:
        return yaml.safe_load(f)


def _ask_gemini(image_bgr, prompt, model="gemini-robotics-er-2-preview",
                temperature=0.3, json_output=False):
    """One BGR image + text prompt -> Gemini response text (thinking disabled).

    json_output=True turns on constrained decoding: the API can then only
    emit syntactically valid JSON."""
    from google import genai
    from google.genai import types

    _, buf = cv2.imencode(".png", image_bgr)
    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png"),
                  prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if json_output else None,
        ),
    )
    return resp.text


_ABSTAIN_WORDS = {"none", "unclear", "skip", "abstain", "na"}


def _parse_arrow_vote(text, names):
    """The chosen arrow color from one approach-direction reply, or None.

    The prompt asks the model to reason briefly and end with a 'FINAL: <color>'
    line (or 'FINAL: none' to abstain on a cluttered/edge-on view), so the reply
    names several colors before its answer — a plain "first color in the text"
    scan would grab a color from the reasoning, not the decision. Prefer the FINAL
    line; treat an abstention as no vote; only when there is no usable FINAL line
    fall back to the LAST color named (the conclusion of a bare reply)."""
    low = str(text).lower()
    m = re.search(r"final\s*[:\-]?\s*\**\s*([a-z]+)", low)
    if m:
        choice = m.group(1)
        if choice in names:
            return choice
        if choice in _ABSTAIN_WORDS:
            return None
    hits = [c for c in re.findall(r"[a-z]+", low) if c in names]
    return hits[-1] if hits else None


# ══════════════════════════════════════════════════════════════════════════════
#  Capture -> detect -> segment -> reconstruct
# ══════════════════════════════════════════════════════════════════════════════

def capture_rgbd():
    """One RGB-D frame: (BGR image, depth). Depth is a raw uint16 mm map from the
    live camera, or an Open3D point cloud on the offline test-data path."""
    if not USE_LIVE_CAMERA:
        rgb_path = TEST_DIR / "rgb.png"
        depth_path = TEST_DIR / "depth.png"
        ply_path = TEST_DIR / "pointcloud.ply"
        img = cv2.imread(str(rgb_path))
        if img is None:
            raise FileNotFoundError(
                f"USE_LIVE_CAMERA=False but no readable RGB image at {rgb_path}. "
                f"The offline test dataset is missing. Either drop an rgb.png + "
                f"pointcloud.ply into {TEST_DIR}, or set USE_LIVE_CAMERA=True in "
                f"semantic_grasp/config.py to capture from the camera."
            )
        # Prefer the raw depth map when the scene has one: it is the same uint16
        # mm array the live camera returns, so the offline path back-projects
        # through camera_model exactly like a live run. pointcloud.ply is the
        # fallback for pre-depth.png captures (scenes 1-6), whose points were
        # baked with the old distortion-free pinhole and cannot be un-baked.
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            return img, depth_img

        depth = o3d.io.read_point_cloud(str(ply_path))
        if depth.is_empty():
            raise FileNotFoundError(
                f"USE_LIVE_CAMERA=False but no readable depth at {depth_path} or "
                f"{ply_path} (0 points). Provide one, or set USE_LIVE_CAMERA=True."
            )
        return img, depth

    # Live capture over ZMQ from camera_server (`python servers/camera_server.py`
    # on the camera host). _depth_to_ply_bytes back-projects the depth map later.
    with OrbbecClient() as client:
        # capture() already returns BGR — camera_server converts the sensor
        # frame to BGR before cv2.imencode, and cv2.imdecode client-side gives
        # BGR back. Converting again here swapped R and B (a red object reached
        # the VLM as blue), and left this branch disagreeing with the offline
        # one above, where cv2.imread is BGR already.
        img, depth, _header = client.capture()
    return img, depth


# Boxes from the most recent get_pertinent_objects call: {name: [x0, y0, x1, y1]}
# in pixels of the image it looked at (shape remembered alongside). segment_objects
# picks these up by default, so the many existing
#     objects = get_pertinent_objects(img); masks = segment_objects(img, objects)
# call sites get the box->SAM path without changing. Boxes come from the same
# Gemini Robotics-ER call that names the objects: it sees the whole scene at once
# and gives each label its own region, whereas the old per-name text-to-mask
# (LangSAM/GroundingDINO) query could return the SAME region for two different
# names (benchmark scene_6/7: "spray bottle" came back as its neighbour, so two
# meshes were reconstructed at one pose and the real object was never in the
# scene). Names with no box (legacy replies, cached objects.json without a
# boxes.json, a degenerate box) still fall back to LangSAM per name.
_detected_boxes = {}
_detected_boxes_shape = None   # (H, W) of the image the boxes belong to


def last_detected_boxes():
    """{name: [x0, y0, x1, y1]} pixel boxes from the last detection (a copy)."""
    return {n: list(b) for n, b in _detected_boxes.items()}


def _fix_decimal_slip(v):
    """box_2d coordinates are integers in 0-1000, but Gemini sometimes slips
    the decimal point left ("235" arrives as "2.35" — scene_11 got six of
    eight ymins that way, stretching every box to the top of the frame). A
    fractional coordinate is that glitch: shift the point back until the
    value is integral, and keep the original if that overshoots 1000."""
    if abs(v - round(v)) < 1e-6:
        return v
    fixed = v
    while abs(fixed - round(fixed)) >= 1e-6 and fixed <= 1000:
        fixed *= 10
    fixed = float(round(fixed))
    return fixed if fixed <= 1000 else v


def _parse_detections(text, img_shape):
    """Gemini detection reply -> (names, {name: [x0, y0, x1, y1] px}).

    Accepts the current [{"label", "box_2d"}, ...] format and the legacy bare
    ["name", ...] list (no boxes). box_2d is Gemini's [ymin, xmin, ymax, xmax]
    normalized to 0-1000 over the full image. Duplicate labels are made unique
    ("white mug", "white mug 2") — the names key the masks/meshes downstream, so
    a repeat would silently drop an object. Degenerate boxes are dropped (that
    name falls back to LangSAM) rather than sent to SAM. raw_decode parses the
    first JSON value and ignores trailing garbage — Gemini sometimes emits a
    duplicate closing bracket after the array."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip())
    data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, list):
        raise ValueError(f"detection reply is not a JSON array: {text[:200]!r}")
    h, w = img_shape[:2]
    names, boxes = [], {}
    for entry in data:
        if isinstance(entry, str):
            name, box = entry, None
        elif isinstance(entry, dict):
            name = entry.get("label", entry.get("name"))
            box = entry.get("box_2d")
        else:
            continue
        if not name:
            continue
        name = str(name).strip()
        base, k = name, 2
        while name in names:
            name = f"{base} {k}"
            k += 1
        names.append(name)
        if box is None:
            continue
        try:
            ymin, xmin, ymax, xmax = (_fix_decimal_slip(float(v)) for v in box)
        except (TypeError, ValueError):
            print(f"[get_pertinent_objects] WARNING: bad box_2d for {name!r}: {box!r}")
            continue
        x0 = min(max(xmin / 1000.0 * w, 0.0), w - 1.0)
        x1 = min(max(xmax / 1000.0 * w, 0.0), w - 1.0)
        y0 = min(max(ymin / 1000.0 * h, 0.0), h - 1.0)
        y1 = min(max(ymax / 1000.0 * h, 0.0), h - 1.0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            print(f"[get_pertinent_objects] WARNING: degenerate box for {name!r}: {box!r}")
            continue
        boxes[name] = [x0, y0, x1, y1]
    return names, boxes


def _remember_boxes(boxes, img_shape):
    global _detected_boxes_shape
    _detected_boxes.clear()
    _detected_boxes.update(boxes)
    _detected_boxes_shape = tuple(img_shape[:2])


def get_pertinent_objects(img):
    """Ask the VLM which graspable objects are in the BGR image; returns the
    list of object names. The same call also returns a 2D box per object, kept
    in last_detected_boxes() and used by segment_objects to prompt SAM.

    With USE_LIVE_ER=False these names are replayed from TEST_DIR/objects.json
    (and the boxes from TEST_DIR/boxes.json, if the cache run wrote one)
    instead. That matters when USE_LIVE_SAM3D is also False: the cached meshes
    are keyed by the names of the run that produced them, and the live VLM call
    can rename an object between runs — "spray bottle" one run, "bottle" the
    next — which would miss the cached mesh. Cache both, or neither.
    scripts/cache_meshes.py writes the two together."""
    if not USE_LIVE_ER:
        objects_path = TEST_DIR / "objects.json"
        if not objects_path.exists():
            raise FileNotFoundError(
                f"USE_LIVE_ER=False but no cached object list at {objects_path}. "
                f"Generate it with `python scripts/cache_meshes.py --scene "
                f"{TEST_DIR}`, or set USE_LIVE_ER=True in semantic_grasp/config.py."
            )
        with open(objects_path) as f:
            objects = json.load(f)
        boxes = {}
        boxes_path = TEST_DIR / "boxes.json"
        if boxes_path.exists():
            with open(boxes_path) as f:
                boxes = {n: [float(v) for v in b] for n, b in json.load(f).items()
                         if n in objects}
        _remember_boxes(boxes, img.shape)
        print(f"[get_pertinent_objects] cached: {objects}"
              + (f" ({len(boxes)} boxes)" if boxes else ""))
        return objects

    cfg = _load_prompt("scene_object_detection_prompt.yaml")
    prompt = cfg["system_prompt"].strip() + "\n\n" + cfg["user_prompt_template"]
    for attempt in (1, 2):
        text = _ask_gemini(img, prompt, json_output=True)
        print(f"[get_pertinent_objects] {text}")
        try:
            names, boxes = _parse_detections(text, img.shape)
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                raise
            print(f"[get_pertinent_objects] bad reply ({e}); retrying once")
    _remember_boxes(boxes, img.shape)
    missing = [n for n in names if n not in boxes]
    if missing:
        print(f"[get_pertinent_objects] WARNING: no box for {missing}; "
              f"segment_objects will fall back to LangSAM for these")
    return names


def _segmentation_request(payload):
    """One REQ/REP round-trip to the perception server; returns {name: mask}."""
    socket.send(pickle.dumps(payload))
    try:
        reply = socket.recv()
    except zmq.Again:
        raise TimeoutError(
            f"no reply from the segmentation server at {PERCEPTION} within "
            f"{LANGSAM_TIMEOUT_S}s. Is it running?  ->  python servers/perception_server.py"
        )
    result = pickle.loads(reply)
    if isinstance(result, dict) and "error" in result:   # server-side failure reply
        raise RuntimeError(f"segmentation server error: {result['error']}")
    return result


def _mask_iou(a, b):
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def check_distinct_masks(masks, iou_thresh=0.5, raise_on_dup=False):
    """Flag pairs of masks that cover (nearly) the same pixels — the signature
    of two names being resolved to one physical object. Returns the offending
    (name_a, name_b, iou) triples; prints them, and raises if asked."""
    names = list(masks)
    dups = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            iou = _mask_iou(masks[a], masks[b])
            if iou > iou_thresh:
                dups.append((a, b, iou))
    for a, b, iou in dups:
        print(f"[segment_objects] WARNING: masks for {a!r} and {b!r} overlap "
              f"(IoU {iou:.2f}) — same physical object under two names?")
    if dups and raise_on_dup:
        raise RuntimeError(
            "segmentation resolved two names to one object: "
            + "; ".join(f"{a!r}~{b!r} (IoU {iou:.2f})" for a, b, iou in dups))
    return dups


def segment_objects(bgr, objects, boxes=None):
    """{name: uint8 mask} for each name in `objects` (BGR image).

    Names with a 2D box — `boxes` ({name: [x0, y0, x1, y1]} px) if given, else
    the boxes remembered from the last get_pertinent_objects call on an image
    of the same shape — are segmented by prompting SAM with the box
    ("sam_boxes" request). Any name without a box goes to the LangSAM text
    query ("langsam" request), the old path. Overlapping results are flagged
    (check_distinct_masks) but returned as-is."""
    if boxes is None:
        if _detected_boxes_shape is not None and tuple(bgr.shape[:2]) != _detected_boxes_shape:
            print(f"[segment_objects] WARNING: image shape {bgr.shape[:2]} != "
                  f"detection image shape {_detected_boxes_shape}; ignoring the "
                  f"remembered boxes")
            boxes = {}
        else:
            boxes = {n: _detected_boxes[n] for n in objects if n in _detected_boxes}
    boxed = [n for n in objects if n in boxes]
    unboxed = [n for n in objects if n not in boxes]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    masks = {}
    if boxed:
        masks.update(_segmentation_request({
            "model": "sam_boxes",
            "boxes": {n: [float(v) for v in boxes[n]] for n in boxed},
            "image": rgb,
        }))
    if unboxed:
        if boxed:
            print(f"[segment_objects] no boxes for {unboxed}; using LangSAM for these")
        masks.update(_segmentation_request({
            "model": "langsam", "objects": unboxed, "image": rgb,
        }))
    missing = [n for n in objects if n not in masks]
    if missing:
        raise RuntimeError(f"segmentation server returned no mask for {missing}")
    masks = {n: masks[n] for n in objects}
    check_distinct_masks(masks)
    return masks


def _sam3d_request(png_bytes, ply_bytes, masks_zip_bytes):
    """One predict round-trip. Fresh REQ socket per call — REQ sockets can't be
    reused after a timeout, and there's one reconstruction per run."""
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(SAM3D_TIMEOUT_S * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(SAM3D_ADDR)
    try:
        sock.send_multipart([b"predict", png_bytes, ply_bytes, masks_zip_bytes])
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            raise TimeoutError(
                f"no reply from sam3d at {SAM3D_ADDR} within {SAM3D_TIMEOUT_S}s "
                f"(is the SSH tunnel to the HPC node up?)"
            )
    finally:
        sock.close(0)
    if parts[0] != b"ok" or len(parts) < 2:
        raise RuntimeError(f"sam3d server error reply: {parts[0][:200]!r}")
    return parts[1]


def _is_organized_cloud(depth) -> bool:
    """True for an (H, W, 3) camera-frame cloud (metres, z <= 0 = invalid) —
    the pixel-aligned observation the sim baseline_server serves."""
    return (isinstance(depth, np.ndarray) and depth.ndim == 3
            and depth.shape[-1] == 3)


def _depth_to_ply_bytes(depth) -> bytes:
    """Back-project a registered depth map (mm) into a point cloud and serialize
    it as a PLY, through camera_model's undistorted ray table. An organized
    (H, W, 3) cloud or an o3d PointCloud (legacy offline capture) is exported
    directly.

    Whoever consumes this PLY and needs to get back to pixels must project with
    camera_model.project() — sam3d_server.ply_to_pointmap does."""
    if isinstance(depth, o3d.geometry.PointCloud):
        return trimesh.PointCloud(np.asarray(depth.points)).export(file_type="ply")
    if _is_organized_cloud(depth):
        pts = np.asarray(depth, dtype=np.float32).reshape(-1, 3)
        return trimesh.PointCloud(pts[pts[:, 2] > 0]).export(file_type="ply")
    depth = np.asarray(depth, dtype=np.float32).squeeze() / 1000.0   # mm -> m
    pts = camera_model.unproject_depth(depth)
    return trimesh.PointCloud(pts).export(file_type="ply")


# Camera->robot extrinsics for the client-side box path (reconstruct_3d's
# non-target branch). Same calibration file the sam3d server loads (C2R_PATH
# env var, repo-root C2R.npy by default), so the calibration still lives in
# exactly one file. As stored, the matrix maps the pinhole CV camera frame
# (x right, y down, z forward, meters) to the robot base — the server only
# remixes its rotation block because its poses arrive in a blender axis
# convention; raw back-projected points are already CV-frame.
_C2R_PATH = Path(os.environ.get("C2R_PATH", Path(__file__).parents[2] / "C2R.npy"))
_c2r_matrix = None


def _c2r():
    global _c2r_matrix
    if _c2r_matrix is None:
        _c2r_matrix = np.load(_C2R_PATH)
    return _c2r_matrix


def _masked_camera_points(depth, mask):
    """CV-camera-frame points (N, 3, meters) of one object's mask. `depth` is a
    registered uint16 mm map (live camera, and any offline scene with a
    depth.png), an organized (H, W, 3) cloud already pixel-aligned with the
    mask (sim observations), or an o3d PointCloud (legacy offline capture);
    only the last one has to be projected back into image space to pick out
    the mask's pixels."""
    m = np.asarray(mask) > 0
    h, w = m.shape[:2]
    if _is_organized_cloud(depth):
        pts = np.asarray(depth, dtype=np.float32)
        return pts[m & (pts[:, :, 2] > 0)]
    if isinstance(depth, o3d.geometry.PointCloud):
        # Legacy clouds only — scenes captured before the ray table existed.
        # Their points were built with the naive pinhole, so project_pinhole is
        # what round-trips them to the pixel they came from. Correcting the
        # projection here without correcting the points would shear the mask
        # off its object by up to 20 px at the frame edges.
        pts = np.asarray(depth.points, dtype=np.float32)
        pts = pts[pts[:, 2] > 0]
        uv = np.round(camera_model.project_pinhole(pts)).astype(int)
        u, v = uv[:, 0], uv[:, 1]
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        pts, u, v = pts[inb], u[inb], v[inb]
        return pts[m[v, u]]
    d = np.asarray(depth, dtype=np.float32).squeeze() / 1000.0   # mm -> m
    return camera_model.unproject_depth(d, mask=m)


def _cleaned_obb_from_mask(depth, mask, name):
    """Minimal OBB of one object's cleaned, robot-frame masked depth points.

    The object's masked depth pixels go camera->robot through the C2R
    extrinsics, then open3d cleans them up — 5mm voxel downsample, a z>5mm cut
    (the table sits at z=0 in the robot frame; RANSAC plane removal on a small
    masked patch can pick the object's own top face instead), statistical
    outlier removal, and DBSCAN keeping only the largest cluster (mask bleed
    onto the background lands in other clusters or noise and would fatten the
    box). The survivors' minimal OBB (not the default PCA one: on a partial,
    camera-visible surface PCA tilts the axes and fattens every extent by
    centimetres) is returned as (center (3,), R (3,3) right-handed, extent (3,),
    n_points), or None when too few points survive to box anything.
    """
    pts = _masked_camera_points(depth, mask)
    if len(pts) < 50:
        print(f"[reconstruct_3d] {name!r}: only {len(pts)} depth points in mask; no box")
        return None
    C2R = _c2r()
    pts = pts @ C2R[:3, :3].T + C2R[:3, 3]           # camera -> robot base

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=0.005)    # 5mm: speed + uniform density
    above = np.flatnonzero(np.asarray(pcd.points)[:, 2] > 0.005)
    if above.size >= 50:                             # keep flat objects boxable
        pcd = pcd.select_by_index(above)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.asarray(pcd.cluster_dbscan(eps=0.02, min_points=30))
    if labels.size and labels.max() >= 0:
        largest = np.bincount(labels[labels >= 0]).argmax()
        pcd = pcd.select_by_index(np.flatnonzero(labels == largest))
    if len(pcd.points) < 50:
        print(f"[reconstruct_3d] {name!r}: {len(pcd.points)} points after cleanup; no box")
        return None

    obb = pcd.get_minimal_oriented_bounding_box()
    R = np.asarray(obb.R, dtype=float).copy()
    if np.linalg.det(R) < 0:          # o3d OBB axes can come out left-handed
        R[:, 2] *= -1.0               # the box is axis-symmetric, so this is free
    return (np.asarray(obb.center, dtype=float).copy(), R,
            np.asarray(obb.extent, dtype=float).copy(), len(pcd.points))


def _obbs_collide(obb_a, obb_b, grow_a=0.0):
    """Separating-axis (Gottschalk) OBB-vs-OBB intersection test.

    `obb_a` / `obb_b` are (center, R, extent, ...) tuples from
    _cleaned_obb_from_mask. `grow_a` inflates every SIDE of box A by that
    margin in metres (extents grow by 2*grow_a) before testing — the target's
    clearance. Touching boxes count as colliding.
    """
    ca, Ra = np.asarray(obb_a[0], float), np.asarray(obb_a[1], float)
    cb, Rb = np.asarray(obb_b[0], float), np.asarray(obb_b[1], float)
    ha = (np.asarray(obb_a[2], float) + 2.0 * grow_a) / 2.0    # half-extents
    hb = np.asarray(obb_b[2], float) / 2.0
    R = Ra.T @ Rb                     # B's axes in A's frame
    t = Ra.T @ (cb - ca)              # B's center in A's frame
    aR = np.abs(R) + 1e-9             # epsilon stabilizes near-parallel edge axes
    for i in range(3):                # A's face normals
        if abs(t[i]) > ha[i] + hb @ aR[i]:
            return False
    for j in range(3):                # B's face normals
        if abs(t @ R[:, j]) > ha @ aR[:, j] + hb[j]:
            return False
    for i in range(3):                # edge-edge cross axes
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = ha[i1] * aR[i2, j] + ha[i2] * aR[i1, j]
            rb = hb[j1] * aR[i, j2] + hb[j2] * aR[i, j1]
            if abs(t[i2] * R[i1, j] - t[i1] * R[i2, j]) > ra + rb:
                return False
    return True


def _box_from_mask(depth, mask, name, obb=None):
    """Non-target reconstruction: an oriented-bounding-box stand-in mesh.

    The OBB comes from _cleaned_obb_from_mask (pass a precomputed `obb` tuple
    to skip recomputing the point cleanup). Returns (box Trimesh, transform
    entry) or None when too few points survive to box anything. The box is
    authored Y-up like the sam3d GLBs, because the IK/Isaac servers re-apply
    the Y-up->Z-up fix to every GLB mesh; the entry carries the OBB pose with
    scale 1.
    """
    from .geometry import R_MESH_YUP_TO_ZUP

    if obb is None:
        obb = _cleaned_obb_from_mask(depth, mask, name)
    if obb is None:
        return None
    center, R, extent, n_pts = obb

    box = trimesh.creation.box(extents=extent)
    to_yup = np.eye(4)
    to_yup[:3, :3] = R_MESH_YUP_TO_ZUP.T             # author the mesh Y-up (see above)
    box.apply_transform(to_yup)

    M = np.eye(4)
    M[:3, :3] = R
    quat_wxyz = trimesh.transformations.quaternion_from_matrix(M)
    entry = {"translation": center.tolist(),
             "rotation_wxyz": [float(q) for q in quat_wxyz],
             "scale": 1.0}
    print(f"[reconstruct_3d] {name!r}: boxed {n_pts} pts -> extents "
          f"{np.round(extent, 3).tolist()} m at {np.round(center, 3).tolist()}")
    return box, entry


def objects_by_name(transformations):
    """Normalize a transforms dict so "objects" is keyed by name.

    The sam3d server now emits objects keyed by name directly
    ({"objects": {name: {translation, rotation_wxyz, scale}}}), which passes
    through unchanged. LEGACY files store a list of dicts, each carrying its
    own "name" field; those are converted — the name moves into the key and is
    dropped from the entry (every other field rides through untouched):

        {"objects": [{"name": n, ...}, ...]} -> {"objects": {n: {...}, ...}}

    List order is preserved (dicts are insertion-ordered)."""
    objs = transformations.get("objects", [])
    if isinstance(objs, dict):
        return transformations
    by_name = {}
    for o in objs:
        entry = dict(o)
        name = entry.pop("name")
        if name in by_name:
            raise ValueError(f"duplicate object name {name!r} in transforms")
        by_name[name] = entry
    out = dict(transformations)
    out["objects"] = by_name
    return out


def reconstruct_3d(rgb, depth, objects, target=None, full_scene=False,
                   target_clearance=None):
    """Per-object 3D reconstruction. Returns (meshes {name: Trimesh}, transforms
    dict with "objects" keyed by name — see objects_by_name).

    `full_scene` picks the reconstruction mode independently of `target` (which
    keeps identifying the target object for callers downstream, e.g.
    grasp.load_meshes / VLM prompts, even when it isn't None):

    - full_scene=False (default): ONLY the target's mask goes to the sam3d
      server for a full mesh; every other object becomes an oriented-
      bounding-box stand-in built locally from its masked depth points
      (_box_from_mask) — the non-targets only matter as collision context, so
      a box is enough and skips the expensive server round-trip. With
      target=None too, this degrades to sending every mask to sam3d (no
      target to single out).
    - full_scene=True: every mask is sent to sam3d for a real mesh (no boxes)
      — scripts/cache_meshes.py wants real meshes for everything.

    `target_clearance` (metres) turns on the crowding check in either mode:
    every object's OBB is built from its masked depth points, the TARGET's OBB
    is inflated by `target_clearance` on every side, and any non-target whose
    OBB intersects it is PROMOTED into the eval group — recorded in the
    returned transforms dict as `"eval_objects": [target, *colliding]` so the
    caller can spawn those same objects in the Isaac eval world. In target-only
    mode a promoted neighbor also gets a full sam3d reconstruction instead of a
    box (it sits close enough to matter as real geometry during the grasp);
    with `full_scene=True` everything is reconstructed anyway, so the check
    only decides the eval group.

    Offline (USE_LIVE_SAM3D=False) the cached test meshes are returned
    unchanged regardless of these flags."""
    if not USE_LIVE_SAM3D:
        mesh_dir = TEST_DIR / "meshes"
        meshes = {f.stem: trimesh.load(f) for f in mesh_dir.glob("*.glb")}
        with open(mesh_dir / "transforms.json") as f:
            return meshes, objects_by_name(json.load(f))

    if target is not None and target not in objects:
        raise ValueError(f"target {target!r} not among the detected objects "
                         f"{list(objects)}; nothing would be sent to sam3d")
    sam3d_names = list(objects) if (full_scene or target is None) else [target]

    # Crowding check: does anything sit inside the target's clearance envelope?
    # OBBs are built locally from masked depth (no server round-trip), so this
    # runs BEFORE the sam3d request and decides what that request contains.
    obbs, eval_objects = {}, None
    if target_clearance is not None and target is not None:
        for name, mask in objects.items():
            obbs[name] = _cleaned_obb_from_mask(depth, mask, name)
        target_obb = obbs[target]
        eval_objects = [target]
        if target_obb is None:
            print(f"[reconstruct_3d] no OBB for target {target!r}; "
                  f"skipping the clearance check")
        else:
            colliding = [name for name, obb in obbs.items()
                         if name != target and obb is not None
                         and _obbs_collide(target_obb, obb, grow_a=target_clearance)]
            if colliding:
                print(f"[reconstruct_3d] {colliding} within "
                      f"{target_clearance * 1000:.0f} mm of target {target!r}; "
                      f"eval world will carry {[target] + colliding}")
                eval_objects = [target] + colliding
            else:
                print(f"[reconstruct_3d] no OBB within {target_clearance * 1000:.0f} mm "
                      f"of target {target!r}; eval world carries the target alone")
        if not full_scene:
            sam3d_names = list(eval_objects)   # promoted neighbors get real meshes

    # Request: image.png + pointcloud.ply + masks.zip (mask filenames become the
    # object names server-side) — sam3d reconstructs exactly the masks it gets.
    img_buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb)).save(img_buf, format="PNG")
    ply_bytes = _depth_to_ply_bytes(depth)
    masks_buf = io.BytesIO()
    with zipfile.ZipFile(masks_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sam3d_names:
            m = np.asarray(objects[name])
            if m.dtype != np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            png = io.BytesIO()
            Image.fromarray(m).save(png, format="PNG")
            zf.writestr(f"{name}.png", png.getvalue())

    zip_bytes = _sam3d_request(img_buf.getvalue(), ply_bytes, masks_buf.getvalue())

    # Reply: a zip of <name>.glb files plus transforms.json — same shape as offline.
    meshes = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open("transforms.json") as jf:
            transformations = objects_by_name(json.load(jf))
        for info in zf.infolist():
            if info.filename.lower().endswith(".glb"):
                with zf.open(info) as gf:
                    meshes[info.filename[:-4]] = trimesh.load(
                        io.BytesIO(gf.read()), file_type="glb", force="mesh")

    # Promoted neighbors sam3d was ASKED for but returned no mesh for still
    # belong in the Isaac eval world (they met the target's clearance criterion).
    # Log them here, box them in the loop below (which already handles this
    # fallback), then keep them in eval_objects — the eval group is finalized
    # AFTER boxing so those box stand-ins count.
    if eval_objects is not None:
        dropped_by_sam3d = [n for n in eval_objects
                            if n != target and n not in meshes]
        if dropped_by_sam3d:
            print(f"[reconstruct_3d] sam3d returned no mesh for {dropped_by_sam3d}; "
                  f"spawning box stand-ins for them in the eval world")

    # Non-targets: OBB box stand-ins from their own masked depth points (reusing
    # the clearance check's OBBs when they were computed). A promoted neighbor
    # sam3d dropped is boxed here too, so it still has geometry to spawn.
    for name, mask in objects.items():
        if name in sam3d_names:
            promoted_fallback = (eval_objects is not None and name != target
                                 and name not in meshes)
            if not promoted_fallback:
                continue
        boxed = _box_from_mask(depth, mask, name, obb=obbs.get(name))
        if boxed is not None:
            meshes[name], transformations["objects"][name] = boxed

    # Finalize the eval group AFTER boxing: every promoted neighbor that now has
    # geometry — its sam3d mesh, or the box stand-in above — goes to the Isaac
    # eval world. Only a neighbor with too few depth points to even box is left
    # out. The target keeps its own hard-failure path.
    if eval_objects is not None:
        no_geometry = [n for n in eval_objects if n not in meshes and n != target]
        if no_geometry:
            print(f"[reconstruct_3d] no mesh or box for {no_geometry}; "
                  f"excluding them from the eval world")
        transformations["eval_objects"] = [n for n in eval_objects
                                           if n in meshes or n == target]
    return meshes, transformations


def _match_name(text, names):
    """Resolve a free-text object description to one of `names`: exact match,
    then substring either way, then any shared word. None if nothing hits."""
    text = text.lower()
    for name in names:
        if text == name.lower():
            return name
    for name in names:
        if text in name.lower() or name.lower() in text:
            return name
    text_words = set(text.split())
    for name in names:
        if text_words & set(name.lower().split()):
            return name
    return None


def find_target_name(task, names):
    """Which of the detected objects the robot should interact with FIRST for
    `task`. `names` is the list from get_pertinent_objects; the return value is
    always one of those names (or None if nothing fits).

    Gemini Robotics ER makes the call — the task's direct object, or the thing
    being carried rather than the destination, or the tool, or whatever is in
    the way (see config/target_object_selection_prompt.yaml). Its answer is
    matched back onto `names` so a paraphrase still resolves; if the model
    fails or answers off-list, fall back to matching the task text itself."""
    from .vlm import query_target_object

    if not names:
        return None
    try:
        choice = query_target_object(task, names)
    except Exception as e:
        print(f"[find_target_name] VLM selection failed ({e}); "
              f"falling back to matching the task text")
        choice = None

    if choice is not None:
        target = _match_name(choice, names)
        if target is not None:
            return target
        print(f"[find_target_name] VLM chose {choice!r}, which is not in "
              f"{names}; falling back to matching the task text")
    return _match_name(task, names)


# ══════════════════════════════════════════════════════════════════════════════
#  VLM grasp proposal: WHERE to grasp (xyz) and HOW to approach (rpy),
#  each in two stages — a semantic phrase first, then the numbers it conditions.
# ══════════════════════════════════════════════════════════════════════════════

def _target_mesh(meshes, object_name):
    """Accept either the full {name: mesh} dict or a single target mesh."""
    return meshes[object_name] if isinstance(meshes, dict) else meshes


def VLM_Predict_XYZ_Semantic(task, target, meshes=None, transformations=None):
    """WHERE to grasp, in words: the specific part/region of `target` the
    gripper should take for `task` (e.g. "outer edge of the bowl rim").

    Feed the phrase to VLM_Predict_XYZ to turn it into metric points, and to
    VLM_Predict_RPY / VLM_Predict_Gripper_Width as `grasp_part` so every stage
    aims at the same part. `meshes`/`transformations` are accepted for
    signature parity with the numeric stages; the choice is text-only today.
    """
    from .vlm import query_grasp_semantics
    return query_grasp_semantics(task, target)


def VLM_Predict_RPY_Semantic(task, target, meshes=None, transformations=None,
                             grasp_part=None):
    """HOW to approach, in words: a short phrase describing the direction the
    gripper should come from and how the jaws align on the part (e.g. "from
    directly above, jaws closing across the rim edge").

    `grasp_part` is the phrase from VLM_Predict_XYZ_Semantic so the approach is
    chosen for the same part. Feed the result to VLM_Predict_RPY as
    `approach_hint` to condition the arrow vote. `meshes`/`transformations` are
    accepted for signature parity with the numeric stages; the choice is
    text-only today.
    """
    from .vlm import query_approach_semantics
    return query_approach_semantics(task, target, grasp_part)


def VLM_Predict_XYZ(task, object_name, meshes, transforms, grasp_part=None,
                    save_dir=None):
    """Return (points, grasp_part): every valid grasp candidate as an (N, 3)
    array in the robot base frame, plus the grasp-semantics phrase (the specific
    part/region of the target to grasp, e.g. "outer edge of the bowl rim") that
    the VLM was pointed at — feed it to VLM_Predict_RPY so the approach choice
    targets the same part.

    `meshes` may be the full {name: mesh} dict or the single target mesh.
    `grasp_part` supplies the part phrase when the semantic stage already ran
    (VLM_Predict_XYZ_Semantic); when omitted it is derived here (legacy
    single-call path) and returned either way.
    `save_dir` writes the whole stage there, in the order it happens:
        photo_NN_<label>.png     the views sent to the VLM
        pointed_NN_<label>.png   the same views with the VLM's red dot on them
        cluster_NN_<signs>.png   the clustered 3D result, as red spheres on the
                                 object, from 20 viewpoints
    plus a contact sheet for each of the last two.

    Renders multiple views of the in-memory target mesh, asks the VLM to point
    at the graspable feature in each, unprojects those 2D points to 3D (DBSCAN
    gives one candidate per cluster plus the overall mean), then maps them
    GLB -> robot.
    """
    from .rendering import render_views
    from .vlm import query_vlm_views
    from .geometry import unproject_predictions, glb_points_to_robot

    mesh = _target_mesh(meshes, object_name)
    metadata = render_views(mesh)
    if save_dir is not None:
        _dump_vlm_stage(save_dir, metadata["views"])
        print(f"[VLM_Predict_XYZ] {len(metadata['views'])} photos -> {save_dir}")
    predictions, grasp_part = query_vlm_views(metadata, task, object_name, grasp_part)
    if save_dir is not None:
        n = _dump_pointed_views(save_dir, metadata, predictions)
        print(f"[VLM_Predict_XYZ] {len(predictions)} point(s) over {n}/"
              f"{len(metadata['views'])} view(s) -> {save_dir}/pointed_*.png")
    glb_points, _ = unproject_predictions(predictions, metadata)

    obj_entry = transforms["objects"][object_name]
    points = glb_points_to_robot(glb_points, obj_entry, mesh=mesh)
    if save_dir is not None:
        _dump_cluster_views(save_dir, mesh, points, transforms, object_name)
        print(f"[VLM_Predict_XYZ] {len(points)} clustered candidate(s) "
              f"(last row = mean) -> {save_dir}/cluster_*.png")
    return points, grasp_part


def VLM_Predict_RPY(task, target, meshes, transformations_settle, grasp_part=None,
                    approach_hint=None, save_dir=None):
    """Pick a gripper-approach orientation (rpy) for `target` by arrow voting.

    Renders the target mesh from 20 OBB viewpoints with 5 colored approach
    arrows drawn in (one per kept OBB face, tips on the face centers). Each
    photo goes to the VLM in its own call, asking which arrow color is the best
    approach for `grasp_part` (falls back to the target name). The per-photo
    color choices are tallied; the most common color's arrow gives the returned
    rpy — the gripper orientation main.py feeds to IK.

    `meshes` may be the full {name: mesh} dict or the single target mesh.
    `approach_hint` is the phrase from VLM_Predict_RPY_Semantic ("from directly
    above, jaws closing across the rim edge"); when given, the vote prefers the
    arrow matching it. `task` is shown to the VLM so the chosen approach
    orientation suits what happens after the grasp (e.g. keeping an opening
    clear to pour, or a handle clear to hand the object off).
    `save_dir` writes every photo sent, the prompt, the replies and a contact
    sheet there (see _dump_vlm_stage).

    Returns the winning arrow's rpy (3,) list, or None if no recognized color
    appeared in any VLM output.
    """
    from .rendering import render_obb_corners, ARROW_COLOR_NAMES

    # 1) Render the 20 photos with the approach arrows drawn in.
    # standoff 3.36 = 2.8 zoomed out 20%, so the arrows are not clipped at the
    # frame edge on elongated objects. arrow_length 0.05 (was 0.04) makes each
    # arrow 25% longer RELATIVE TO THE OBJECT — shaft radius follows it, since
    # render_obb_corners derives shaft_r from arrow_length. The two changes work
    # against each other on screen: 1.25 x longer seen from 1.2 x further away
    # nets only ~4% more pixels, so raise arrow_length further if the arrows
    # still read too small in the photos.
    records, arrows = render_obb_corners(
        _target_mesh(meshes, target), standoff=3.36, draw_obb=False,
        transform=transformations_settle,
        object_name=target, draw_arrows=True, arrow_length=0.05,
    )

    # 2) Ask the VLM, once per photo, which arrow color approaches grasp_part best.
    cfg = _load_prompt("approach_direction_prompt.yaml")
    prompt = cfg["system_prompt"].strip() + "\n\n" + cfg["select_prompt_template"].format(
        task=task or "grasp the object",
        grasp_part=grasp_part or target,
        approach_hint=approach_hint or "(none given — judge from the image alone)",
        colors=", ".join(ARROW_COLOR_NAMES))
    print(f"[VLM_Predict_RPY] prompt (one call per photo):\n{prompt}\n")
    if save_dir is not None:
        _dump_vlm_stage(save_dir, records, prompt)
        print(f"[VLM_Predict_RPY] {len(records)} photos + prompt -> {save_dir}")

    def ask(rec):
        try:
            text = _ask_gemini(
                cv2.cvtColor(rec["image"], cv2.COLOR_RGB2BGR), prompt,
                model=cfg.get("model", "gemini-robotics-er-2-preview"),
                temperature=cfg.get("temperature", 0.3))
            text = " ".join(str(text).split())
        except Exception as e:
            text = f"(unavailable: {e})"
        return rec["index"], rec["signs"], text

    # Blocking sync requests in threads so the network waits overlap; map keeps
    # the results in photo order.
    with ThreadPoolExecutor(max_workers=len(records)) as pool:
        outputs = list(pool.map(ask, records))
    for idx, signs, text in outputs:
        print(f"[VLM_Predict_RPY] photo {idx} ({signs}): {text}")

    _show_photos_with_results(records, outputs)

    # 3) Tally the color votes; the winning color's arrow carries the rpy.
    #    Each reply reasons then ends with 'FINAL: <color>' (or 'FINAL: none' to
    #    abstain on an unreadable view) -- _parse_arrow_vote reads that decision.
    votes = Counter()
    for _idx, _signs, text in outputs:
        name = _parse_arrow_vote(text, ARROW_COLOR_NAMES)
        if name:
            votes[name] += 1

    print("\n=== color votes ===")
    for name, n in votes.most_common():
        print(f"{name}: {n}")
    if not votes:
        print("\nno recognized color in any VLM output")
        if save_dir is not None:
            _dump_vlm_stage(save_dir, records, prompt, outputs, votes, "none")
        return None

    # Antiparallel arrows (e.g. the two side approaches onto the same thin
    # dimension) are the same grasp family — same closing line, opposite
    # approach side — and the per-photo votes split between them depending on
    # which side each camera happens to see. Tally by that axis pair first so a
    # split good-axis vote can't lose to a single bad arrow, then take the
    # stronger color within the winning pair.
    groups, used = [], set()
    for a in arrows:
        if a["color"] in used:
            continue
        partner = next(
            (b for b in arrows if b["color"] not in used and b is not a
             and np.allclose(np.asarray(a["approach"]),
                             -np.asarray(b["approach"]), atol=1e-6)),
            None)
        group = (a["color"],) if partner is None else (a["color"], partner["color"])
        used.update(group)
        groups.append(group)
    axis_votes = {g: sum(votes.get(c, 0) for c in g) for g in groups}
    print("=== axis votes (antiparallel arrows pooled) ===")
    for g, n in sorted(axis_votes.items(), key=lambda kv: -kv[1]):
        print(f"{'+'.join(g)}: {n}")
    best_group = max(groups, key=lambda g: (axis_votes[g],
                                            max(votes.get(c, 0) for c in g)))
    best_color = max(best_group, key=lambda c: votes.get(c, 0))
    count = votes[best_color]
    arrow = next(a for a in arrows if a["color"] == best_color)
    print(f"\nwinning axis: {'+'.join(best_group)} "
          f"({axis_votes[best_group]}/{len(outputs)} photos) "
          f"-> color {best_color} ({count})")
    print(f"approach rpy ({arrow['frame']} frame): {arrow['rpy']}")
    print(f"rpy candidates [base, twist]: {arrow['rpy_candidates']}")
    print(f"approach direction / outward normal: {arrow['approach']} / {arrow['normal']}")
    if save_dir is not None:
        _dump_vlm_stage(save_dir, records, prompt, outputs, votes,
                        f"{best_color} arrow -> rpy {arrow['rpy']}")
        print(f"[VLM_Predict_RPY] results + contact sheet -> {save_dir}")
    return arrow["rpy"]


def VLM_Predict_Gripper_Width(task, target, meshes, transformations, grasp_part=None,
                              save_dir=None):
    """How wide the gripper must open to take `grasp_part` -> width in METERS
    (or None if no photo produced a usable color) — get_grasp_width behind the
    same task-first signature as the other VLM_Predict_* stages. No grasp point
    is passed, so the photos carry no sphere and the scale bars anchor at the
    OBB center.

    `meshes` may be the full {name: mesh} dict or the single target mesh.
    `grasp_part` is the phrase from VLM_Predict_XYZ_Semantic. See
    get_grasp_width for how the scale-bar vote works and what `save_dir` dumps.
    """
    return get_grasp_width(
        _target_mesh(meshes, target), transformations, target,
        grasp_part=grasp_part, task=task, save_dir=save_dir)


def _vlm_result_contact_sheet(records, outputs, cols=4, tile_w=480):
    """BGR contact sheet pairing each photo SENT to the VLM with the result of
    that photo's call: corner index/signs on top, the wrapped response text at
    the bottom."""
    from .rendering import tile_grid

    by_index = {idx: (signs, text) for idx, signs, text in outputs}
    first = np.asarray(records[0]["image"])
    tile_h = int(round(first.shape[0] * tile_w / first.shape[1]))
    tiles = []
    for rec in records:
        bgr = cv2.cvtColor(np.asarray(rec["image"]), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (tile_w, tile_h))
        signs, text = by_index.get(rec["index"], (rec["signs"], ""))
        label = f"{rec['index']} ({signs})"
        cv2.putText(bgr, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(bgr, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        lines = textwrap.wrap(text or "(no response)", width=48)[-7:] or ["(no response)"]
        y = tile_h - 16 * len(lines) - 10
        for line in lines:
            cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
        cv2.rectangle(bgr, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 2)
        tiles.append(bgr)
    return tile_grid(tiles, cols)


def _show_photos_with_results(records, outputs, tag="VLM_Predict_RPY"):
    """Diagnostic pop-up: the photos sent to the VLM next to their per-call
    results (press any key to continue). Headless runs skip entirely — with no
    display, OpenCV's Qt backend hard-aborts the process inside imshow (qFatal),
    so a try/except never gets the chance to fire."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print(f"[{tag}] no display; skipping the {len(records)}-photo contact sheet")
        return
    win = f"{tag}: photos sent to VLM + results"
    print(f"[{tag}] showing {len(records)} photos sent to the VLM "
          f"(press any key in the '{win}' window to continue)")
    # Never let a missing/closed GUI window (e.g. the user closing it via its X
    # button so destroyWindow has nothing to destroy) take down the pipeline.
    try:
        cv2.imshow(win, _vlm_result_contact_sheet(records, outputs))
        cv2.waitKey(0)
        cv2.destroyWindow(win)
    except cv2.error as e:
        print(f"[{tag}] skipping photo display (no usable GUI): {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Debug visualization
# ══════════════════════════════════════════════════════════════════════════════

def show_grasp_on_mesh(points, mesh, sphere_radius=0.02, transform=None, object_name=None):
    """Display robot-frame xyz points as red spheres on the object mesh in
    trimesh's viewer.

    The mesh is shown untouched in its native GLB frame; `transform` (the
    standard GLB->robot conversion) is inverted to map the points back into GLB
    space so they line up. It may be a single object's entry (the dict with
    "translation", "rotation_wxyz", "scale") or a full transforms dict
    ({"objects": {name: entry}}), resolved via `object_name` (or the sole
    entry). If `transform` is omitted the points are assumed to already be
    GLB-frame. `sphere_radius` is in robot-frame meters."""
    from .geometry import _build_glb_to_robot_matrix, resolve_obj_entry

    # Accept a path or an in-memory mesh. Copy in-memory meshes so the grasp
    # spheres added below never leak back into the caller's object (e.g.
    # meshes[target]), which would otherwise get uploaded to Isaac later.
    if isinstance(mesh, (str, Path)):
        scene = trimesh.load(str(mesh), force="scene")
    else:
        scene = mesh.copy()
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    # A full transforms dict ({"objects": {name: entry}}) -> a single entry.
    transform = resolve_obj_entry(transform, object_name)

    # Map the points into GLB space rather than moving the mesh: the viewer
    # auto-places its lights from the scene bbox, so moving the mesh would
    # change the lighting. The conversion's rotation block is scale*R, so
    # robot->glb divides lengths by scale — shrink the sphere radius the same way.
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    radius = sphere_radius
    if transform is not None:
        glb_to_robot = _build_glb_to_robot_matrix(transform, mesh=scene)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts = (np.linalg.inv(glb_to_robot) @ pts_h.T).T[:, :3]
        radius = sphere_radius / float(transform["scale"])

    for pt in pts:
        sphere = trimesh.creation.uv_sphere(radius=radius, count=[16, 16])
        sphere.apply_translation(pt)
        sphere.visual.face_colors = [255, 0, 0, 200]  # red
        scene.add_geometry(sphere)

    # render_views() runs pyrender with PYOPENGL_PLATFORM=egl, which permanently
    # pins this process to a core-profile GL context; pyglet's viewer needs a
    # compatibility profile, so an in-process scene.show() dies. Show it in a
    # fresh subprocess (PYOPENGL_PLATFORM cleared); the scene crosses via an
    # in-memory GLB on stdin — no file written.
    glb = scene.export(file_type="glb")
    viewer = (
        "import sys, io, trimesh; "
        "trimesh.load(io.BytesIO(sys.stdin.buffer.read()), file_type='glb').show()"
    )
    env = dict(os.environ)
    env.pop("PYOPENGL_PLATFORM", None)
    subprocess.run([sys.executable, "-c", viewer], input=glb, env=env)


def _dump_vlm_stage(save_dir, records, prompt=None, outputs=None, votes=None,
                    answer=None):
    """Write out what one VLM stage actually sent and got back, so a bad pick can
    be traced to the images and wording that produced it:

        photo_00_<tag>.png ...   every photo the stage sent
        prompt.txt               the text, identical for all photos, written once
        results.txt              per-photo replies, the vote tally, the answer
        contact_sheet.png        each photo paired with its own reply

    `records` may be either photo shape the pipeline renders — render_obb_corners
    records (RGB array in "image", corner "signs") or render_views views (PNG
    bytes in "image_png", a view "label"). Voting stages (VLM_Predict_RPY,
    get_grasp_width) call this twice: once with photos+prompt BEFORE the VLM runs,
    so a hung or failed call still leaves the inputs on disk, then again with the
    results. Stages that only want their photos kept pass records alone.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if prompt is not None:
        (save_dir / "prompt.txt").write_text(prompt)
    for rec in records:
        tag = rec.get("signs") or rec.get("label") or "view"
        path = save_dir / f"photo_{rec['index']:02d}_{tag}.png"
        if "image_png" in rec:              # render_views: already encoded
            path.write_bytes(rec["image_png"])
        else:                               # render_obb_corners: RGB array
            cv2.imwrite(str(path), cv2.cvtColor(rec["image"], cv2.COLOR_RGB2BGR))

    if outputs is None:
        return
    lines = [f"photo {idx} ({signs}): {text}" for idx, signs, text in outputs]
    if votes is not None:
        lines += ["", "votes: " + (", ".join(f"{n}: {c}" for n, c in votes.most_common())
                                   or "none")]
    if answer is not None:
        lines.append(f"answer: {answer}")
    (save_dir / "results.txt").write_text("\n".join(lines) + "\n")
    cv2.imwrite(str(save_dir / "contact_sheet.png"),
                _vlm_result_contact_sheet(records, outputs))


def _dump_pointed_views(save_dir, metadata, predictions):
    """VLM_Predict_XYZ's raw answer drawn back onto the photo it came from: a red
    dot per predicted point, on the view that produced it -> pointed_NN_<label>.png
    plus a contact sheet of all of them.

    Views the VLM returned no point for are still written, unmarked, so the
    numbering lines up with the photo_NN_* files the stage was sent — a view
    missing a dot IS the diagnostic (the VLM declined, or its point missed the
    depth map and got dropped in unprojection).
    """
    from .geometry import point_to_pixel
    from .rendering import tile_grid
    from ..config import COORD_SCALE

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    by_view = {}
    for p in predictions:
        by_view.setdefault(p["view_index"], []).append(p)

    w, h = metadata["image_width"], metadata["image_height"]
    tiles = []
    for view in metadata["views"]:
        bgr = cv2.imdecode(np.frombuffer(view["image_png"], np.uint8), cv2.IMREAD_COLOR)
        for p in by_view.get(view["index"], []):
            u, v = point_to_pixel(p["point"], COORD_SCALE, w, h)
            center = (int(round(u)), int(round(v)))
            cv2.circle(bgr, center, 8, (0, 0, 255), -1)        # the red dot
            cv2.circle(bgr, center, 8, (255, 255, 255), 1)     # so it reads on a red object
            cv2.putText(bgr, str(p.get("label", "")), (center[0] + 12, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(save_dir / f"pointed_{view['index']:02d}_{view['label']}.png"), bgr)
        tiles.append(bgr)

    tile_w = 400
    tile_h = int(round(tiles[0].shape[0] * tile_w / tiles[0].shape[1]))
    cv2.imwrite(str(save_dir / "pointed_contact_sheet.png"),
                tile_grid([cv2.resize(t, (tile_w, tile_h)) for t in tiles], 4))
    return len(by_view)


def _dump_cluster_views(save_dir, mesh, points, transform, object_name):
    """The stage's OUTPUT photographed: the clustered grasp candidate(s) drawn as
    red spheres on the object from the 20 OBB viewpoints -> cluster_NN_<signs>.png
    plus a contact sheet.

    The per-view dots above are what the VLM said; this is what those votes
    actually resolved to in 3D after unprojection and DBSCAN, seen from every
    side. `points` is the (n_clusters + 1, 3) robot-frame candidate set —
    the cluster centroids, then their overall mean as the last row.
    """
    from .rendering import render_obb_corners, corner_contact_sheet

    # mesh_alpha: translucent shell, so a candidate that unprojected to a point
    # INSIDE the object (a depth hit on the far wall, a hollow mug interior) is
    # visible instead of hidden behind the surface. Output diagnostic only — the
    # photos sent to the VLM above are rendered solid, untouched.
    records, _ = render_obb_corners(
        mesh, standoff=2.8, draw_obb=False, grasp_points=points,
        transform=transform, object_name=object_name, uniform_radius=True,
        mesh_alpha=0.35)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        cv2.imwrite(str(save_dir / f"cluster_{rec['index']:02d}_{rec['signs']}.png"),
                    cv2.cvtColor(rec["image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "cluster_contact_sheet.png"), corner_contact_sheet(records))
    return records


def get_grasp_width(mesh, transform=None, object_name=None, grasp_part=None,
                    grasp_xyz=None, task=None, save_dir=None):
    """Pick how wide the gripper must open to grasp `grasp_part` -> width in
    METERS: the length of the winning scale bar (rendering.RULER_BARS — red 8 cm,
    green 5 cm or blue 2 cm), or None if no photo produced a usable color.

    Renders the same 20 OBB viewpoints as VLM_Predict_RPY, with the grasp point
    as a red sphere — but instead of the approach arrows it draws three stacked
    scale bars under the object: red 8 cm on top, green 5 cm, blue 2 cm at the
    bottom. The bars carry no text; the prompt supplies the color->length
    mapping. The bars are what make the answer metric. A VLM cannot know how big an object is from a bare render — nothing
    in the picture has a size — but it can judge which of three known lengths in
    the same picture best matches the span it is asked about: a CHOICE, which
    VLMs are good at, instead of a measurement, which they are bad at.

    Each bar is drawn flat in the image plane, at the grasp point's own depth,
    fully below the object's silhouette (rendering._ruler_rows), so on every
    photo each bar shows exactly what its length looks like at the object — a bar
    that sat nearer the camera, slanted into the screen, or hid behind the object
    would lie about its size and poison the choice.

    Each photo goes to the VLM in its own call asking which bar best matches the
    opening the fingers need. The per-photo answers are tallied like
    VLM_Predict_RPY's arrow votes, and the most common color's length is
    returned. Ties break toward the LONGER bar (red over green over blue): a
    touch too wide still closes onto the part, a touch too narrow jams into it.

    `transform` is the transforms dict (or a single object entry) placing the
    mesh in the robot frame — required, since the bars are absolute metric
    lengths. `grasp_xyz` is the grasp point from VLM_Predict_XYZ ((3,) or (1,3),
    robot frame); only the first point is used. `task` and `grasp_part` tell the
    VLM which part it is choosing for. `save_dir` writes every photo sent, the
    prompt, the replies and a contact sheet there (see scripts/
    grasp_width_test.py).
    """
    from .rendering import render_obb_corners, RULER_BARS, RULER_COLOR_NAMES

    # 1) Render the 20 photos with the grasp sphere + the three bars drawn in.
    grasp_point = None
    if grasp_xyz is not None:
        grasp_point = np.asarray(grasp_xyz, dtype=float).reshape(-1, 3)[:1]
    records, _ = render_obb_corners(
        mesh, standoff=2.8, draw_obb=False, grasp_points=grasp_point,
        transform=transform, object_name=object_name, draw_ruler=True,
        uniform_radius=True,   # every view must show all three bars, or the choice
                               # they offer grounds nothing
    )

    # 2) Ask the VLM, once per photo, which bar matches the needed opening.
    cfg = _load_prompt("grasp_width_prompt.yaml")
    prompt = cfg["system_prompt"].strip() + "\n\n" + cfg["select_prompt_template"].format(
        task=task or "grasp the object",
        grasp_part=grasp_part or object_name or "the object")
    print(f"[get_grasp_width] prompt (one call per photo):\n{prompt}\n")
    if save_dir is not None:
        _dump_vlm_stage(save_dir, records, prompt)
        print(f"[get_grasp_width] {len(records)} photos + prompt -> {save_dir}")

    def ask(rec):
        try:
            text = _ask_gemini(
                cv2.cvtColor(rec["image"], cv2.COLOR_RGB2BGR), prompt,
                model=cfg.get("model", "gemini-robotics-er-2-preview"),
                temperature=cfg.get("temperature", 0.3))
            text = " ".join(str(text).split())
        except Exception as e:
            text = f"(unavailable: {e})"
        return rec["index"], rec["signs"], text

    with ThreadPoolExecutor(max_workers=len(records)) as pool:
        outputs = list(pool.map(ask, records))
    for idx, signs, text in outputs:
        print(f"[get_grasp_width] photo {idx} ({signs}): {text}")

    _show_photos_with_results(records, outputs, tag="get_grasp_width")

    # 3) Tally the color votes. RULER_BARS is ordered longest first, so on a tied
    # count the first name to reach the top count wins — red over green over blue.
    votes = Counter()
    for _idx, _signs, text in outputs:
        name = next((c for c in RULER_COLOR_NAMES if c in text.lower()), None)
        if name:
            votes[name] += 1

    print("\n=== color votes ===")
    for name, n in votes.most_common():
        print(f"{name}: {n}")

    width_m = None
    if votes:
        top = max(votes.values())
        best = next(n for n in RULER_COLOR_NAMES if votes[n] == top)
        width_m = next(m for name, m, _ in RULER_BARS if name == best)
        print(f"\nmost common color: {best} ({votes[best]}/{len(outputs)} photos)")
        print(f"grasp width: {100.0 * width_m:g} cm  ({width_m:.3f} m)")
        if width_m >= GRIPPER_MAX_WIDTH_M:
            print("note: this is the gripper's full opening — no clearance going in")
    else:
        print("\nno recognized color in any VLM output")

    if save_dir is not None:
        answer = ("none" if width_m is None
                  else f"{100.0 * width_m:g} cm ({width_m:.3f} m)")
        _dump_vlm_stage(save_dir, records, prompt, outputs, votes, answer)
        print(f"[get_grasp_width] results + contact sheet -> {save_dir}")
    return width_m