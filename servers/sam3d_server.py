# sam3d_server.py
#
# SAM-3D reconstruction server. This is a DEPLOY ARTIFACT: it runs inside the
# SAM-3D repo on the GPU box (tc-gpu003), not in semantic_grasp — it imports
# `notebook.inference` and reads `checkpoints/`, which live there. Copy it over
# as that repo's `server.py`.
#
# Output format: transforms.json keys objects BY NAME — the same shape the
# pipeline uses in memory, so nothing downstream converts anything:
#
#     {"objects": {
#         "blue mug": {"translation": [x, y, z],
#                      "rotation_wxyz": [w, x, y, z],
#                      "scale": 1.5},
#         ...
#     }}
#
# (The old list-of-dicts format with per-entry "name"/"glb" fields is legacy;
# `objects_by_name()` in semantic_grasp still converts old files on load. Each
# object's mesh is the sibling "<name>.glb" in the reply zip.)
#
# The transform is the object's rigid pose in the ROBOT-BASE frame
# (translation, rotation_wxyz, scale), with the camera->robot C2R extrinsics
# baked in HERE. Downstream (semantic_grasp) consumes it as-is — no service
# re-applies C2R, so C2R lives in exactly one place. Point C2R_PATH (env var)
# at the same C2R.npy the pipeline uses.
import json
import os
import io
import tempfile
import zipfile
import traceback

import numpy as np
import open3d as o3d
import torch
import zmq
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import quaternion_to_matrix

from notebook.inference import Inference, load_image, load_mask

# ── Config ─────────────────────────────────────────────────────────────────────
SAM3D_CONFIG = "checkpoints/hf/pipeline.yaml"
SEED = 42
BIND_ADDR = "tcp://0.0.0.0:5561"   # for IPC transport use e.g. "ipc:///tmp/sam3d.ipc"

# ── Femto Mega colour intrinsics (1920×1080, rotated frame) ───────────────────
# MUST match semantic_grasp/config.py — the pipeline back-projects the PLY it
# sends us with these, so ply_to_pointmap only lands each point on the pixel it
# came from if both ends agree. Duplicated rather than imported because this
# file deploys into the SAM-3D repo, where semantic_grasp is not importable.
#
# (These were previously FX/FY 1125.7/1126.2, CX/CY 942.6/509.2 — a stale read
# that disagreed with config.py by 18 px in x and 65 px in y, smearing the
# pointmap off the RGB by that much everywhere.)
FX, FY = 1128.720, 1128.520
CX, CY = 960.501, 574.412
WIDTH, HEIGHT = 1920, 1080

# Colour lens distortion, OpenCV order (k1,k2,p1,p2,k3,k4,k5,k6), rotated frame
# = config.CAMERA_DIST_CV. The lens is not a pinhole and the colour stream is
# never rectified, so projecting without this misplaces points by up to ~20 px
# toward the frame edges — zero at the principal point, growing radially.
DIST = (0.080802, -0.106045, 0.000479, -0.000196, 0.043203, 0.0, 0.0, 0.0)

# ── Coordinate-frame conversion ───────────────────────────────────────────────
# SAM-3D camera pose -> "blender" axis convention.
_C = np.array([[-1, 0, 0],
               [ 0, 0, 1],
               [ 0, 1, 0]], dtype=np.float64)

# ── C2R: "blender" camera frame -> robot base frame ───────────────────────────
# The calibrated 4x4 camera->robot extrinsics. Its rotation block is right-
# multiplied by R_BL_TO_CV so it consumes poses in the blender convention that
# `_C` above produces (blender -> CV -> robot). This is exactly the chain the
# pipeline's downstream services used to each run themselves; it now runs once,
# here, so the emitted pose is directly robot-frame usable.
C2R_PATH = os.environ.get("C2R_PATH", "C2R.npy")
_R_BL_TO_CV = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]], dtype=np.float64)
_C2R_RAW = np.load(C2R_PATH)
_C2R = _C2R_RAW.copy()
_C2R[:3, :3] = _C2R_RAW[:3, :3] @ _R_BL_TO_CV
print(f"Loaded C2R from {C2R_PATH}")

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading SAM3D model...")
model = Inference(SAM3D_CONFIG, compile=False)
print("Model ready.")


def project(x, y, z):
    """Camera-frame points (metres) -> float pixel coords, distortion included.

    The forward half of semantic_grasp.camera_model's model, written out in
    numpy so this file keeps its only-numpy dependency footprint. Must stay the
    exact inverse of the ray table the pipeline back-projects the PLY with.
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = DIST
    xn, yn = x / z, y / z
    r2 = xn * xn + yn * yn
    radial = ((1 + r2 * (k1 + r2 * (k2 + r2 * k3)))
              / (1 + r2 * (k4 + r2 * (k5 + r2 * k6))))
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return FX * xd + CX, FY * yd + CY


def ply_to_pointmap(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    pointmap = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    pu, pv = project(x, y, z)
    u = np.round(pu).astype(np.int32)
    v = np.round(pv).astype(np.int32)

    in_bounds = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    u, v = u[in_bounds], v[in_bounds]
    x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]

    pointmap[v, u] = np.stack([-x, -y, z], axis=1)

    filled = np.count_nonzero(pointmap[:, :, 2])
    print(f"[pointmap] {len(points):,} pts → {filled:,}/{WIDTH*HEIGHT:,} pixels "
          f"({100*filled/(WIDTH*HEIGHT):.1f}% filled)")
    return pointmap


def transform_to_robot(result: dict) -> dict:
    """SAM-3D result -> object rigid pose in the ROBOT-BASE frame.

    Two composed steps (matching what the pipeline used to do, split across its
    downstream services — now folded into one place):
      1. SAM-3D camera pose -> "blender" axis convention (the `_C` remap).
      2. blender -> robot base via the calibrated C2R extrinsics.
    The returned translation / rotation_wxyz / scale are directly usable by the
    grasp pipeline: nothing downstream re-applies a camera->robot conversion.
    """
    q_sam = result["rotation"].squeeze().cpu()
    t_cam = result["translation"].squeeze().cpu().numpy().astype(np.float64)
    s_sam = result["scale"].squeeze().cpu().numpy().astype(np.float64)

    R_row = quaternion_to_matrix(q_sam).numpy().astype(np.float64)
    R_col = R_row.T

    # 1) camera -> blender
    R_blender = _C @ R_col
    t_blender = _C @ t_cam

    # 2) blender -> robot base (C2R extrinsics)
    R_robot = _C2R[:3, :3] @ R_blender
    t_robot = (_C2R @ np.array([t_blender[0], t_blender[1], t_blender[2], 1.0]))[:3]

    q_xyzw = Rotation.from_matrix(R_robot).as_quat()
    q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

    scale = float(s_sam.mean())
    return {
        "translation": t_robot.tolist(),
        "rotation_wxyz": q_wxyz,
        "scale": scale,
    }


def handle_predict(image_bytes: bytes, ply_bytes: bytes, masks_zip_bytes: bytes) -> bytes:
    """
    Runs inference and returns the result .zip as bytes.
    Inputs:
      - image_bytes: RGB capture (.png)
      - ply_bytes: point cloud (.ply)
      - masks_zip_bytes: a .zip containing mask PNGs
    """
    tmpdir = tempfile.mkdtemp()

    image_path = os.path.join(tmpdir, "image.png")
    ply_path = os.path.join(tmpdir, "pointcloud.ply")
    masks_dir = os.path.join(tmpdir, "masks")
    os.makedirs(masks_dir)

    with open(image_path, "wb") as f:
        f.write(image_bytes)
    with open(ply_path, "wb") as f:
        f.write(ply_bytes)

    masks_zip = os.path.join(tmpdir, "masks.zip")
    with open(masks_zip, "wb") as f:
        f.write(masks_zip_bytes)
    with zipfile.ZipFile(masks_zip, "r") as z:
        z.extractall(masks_dir)

    # Build pointmap
    pointmap_np = ply_to_pointmap(ply_path)
    pointmap_tensor = torch.from_numpy(pointmap_np)
    pointmap_tensor[pointmap_tensor.sum(dim=-1) == 0] = float("nan")

    # Load image
    img = load_image(image_path)

    # Process each mask. Objects are keyed BY NAME — transform_to_robot()'s
    # {translation, rotation_wxyz, scale} dict is the whole entry; the mesh is
    # the sibling "<name>.glb" in the zip.
    mask_files = sorted(os.listdir(masks_dir))
    scene_json = {"objects": {}}
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir)

    for mask_file in mask_files:
        if not mask_file.lower().endswith(".png"):
            continue
        name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(masks_dir, mask_file)
        print(f"\n── {name} ──")

        try:
            mask = load_mask(mask_path)
            result = model(img, mask, seed=SEED, pointmap=pointmap_tensor)

            glb = result.get("glb")
            if glb is None:
                print(f"  WARNING: no GLB for {name}")
                continue

            glb_path = os.path.join(output_dir, f"{name}.glb")
            glb.export(glb_path)
            print(f"  saved: {glb_path}")

            scene_json["objects"][name] = transform_to_robot(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # Write transforms.json
    json_path = os.path.join(output_dir, "transforms.json")
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=2)

    # Zip everything up
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            zf.write(os.path.join(output_dir, fname), fname)
    buf.seek(0)
    return buf.getvalue()


def recv_request(sock):
    """
    Wire protocol (multipart REQ→REP):
      frame 0: command ("predict" or "health")
      for "predict":
        frame 1: image bytes
        frame 2: ply bytes
        frame 3: masks zip bytes
    """
    frames = sock.recv_multipart()
    cmd = frames[0].decode("utf-8")
    return cmd, frames[1:]


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND_ADDR)
    print(f"ZMQ REP server listening on {BIND_ADDR}")

    while True:
        try:
            cmd, payload = recv_request(sock)

            if cmd == "health":
                sock.send_multipart([b"ok", json.dumps(
                    {"status": "ok", "model": "loaded"}
                ).encode("utf-8")])
                continue

            if cmd == "predict":
                image_bytes, ply_bytes, masks_zip_bytes = payload[0], payload[1], payload[2]
                result_zip = handle_predict(image_bytes, ply_bytes, masks_zip_bytes)
                # reply: status frame + zip bytes
                sock.send_multipart([b"ok", result_zip])
                continue

            # Unknown command
            sock.send_multipart([b"error", f"unknown command: {cmd}".encode("utf-8")])

        except Exception as e:
            traceback.print_exc()
            # REP sockets MUST reply to keep the state machine happy
            try:
                sock.send_multipart([b"error", str(e).encode("utf-8")])
            except zmq.ZMQError:
                pass


if __name__ == "__main__":
    main()
