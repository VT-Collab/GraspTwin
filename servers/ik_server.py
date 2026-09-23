#!/usr/bin/env python3
"""ZMQ collision-aware IK + trajectory server (cuRobo v2, v0.8+ API) — pickle I/O.

What changed vs the old version
-------------------------------
  * Wire format is now PICKLE, not a JSON-header + raw-bytes multipart. A request
    is a single pickled dict; the reply is a single pickled dict. NumPy arrays
    travel as plain lists in BOTH directions (a pickled ndarray embeds a
    numpy-version-specific module path and breaks across the 1.x/2.x split);
    array-typed reply fields below describe what IKClient rebuilds on receipt.
  * No more transforms.json / mesh-dir globbing / collect_mesh_paths. The scene
    is handed in directly: each request carries the list of objects to place in
    the collision world. Nothing is searched for on disk.
  * NEW: a "solve_traj" command plans a full collision-free joint trajectory from
    a start config to a single end-effector goal pose (cuRobo v2 MotionPlanner)
    and returns it resampled to a fixed rate (default 20 Hz) for direct replay on
    the robot. (The original default request still just batch-solves IK.)

Request — batch IK (default; pickled dict):
    {
      "objects": [...],                       # collision scene (see below)
      "poses":      np.ndarray(N, 6) float,   # [x,y,z,roll,pitch,yaw] ROBOT frame
      "batch_size": int,   # optional cap; effective batch = min(batch_size, N)
      "num_seeds":  int,   # optional (def 32 = cuRobo default)
      "table":      {"pose": [x,y,z], "dims": [lx,ly,lz]},  # optional override
    }

Request — full trajectory (pickled dict):
    {
      "cmd":          "solve_traj",
      "objects":      [...],                   # same collision-scene format
      "pose":         [x,y,z,roll,pitch,yaw],  # single EE goal, ROBOT frame
      "start_state":  [q0..q6],   # optional 7-dof start; default = planner home
      "hz":           float,      # optional, output sampling rate (def 20)
      "max_attempts": int,        # optional, plan_pose retries (def 10)
      "table":        {...},      # optional override
      "ignore_collisions": bool,  # optional (also on viz_traj): plan with NO world
                                  # obstacles (objects + table dropped). Self-
                                  # collision and joint limits still apply.
    }
    Or a ping:  {"cmd": "ping"}

  Object entry (used by both):
        {
          "name":        str,            # label
          "mesh_bytes":  bytes,          # an encoded mesh file (in-request)
          "mesh_format": str,            # "glb" / "obj" / ...   (with mesh_bytes)
          # ── or, instead of mesh_bytes/mesh_format ──
          # "vertices":  (V,3),  "faces": (F,3),
          "translation": [x, y, z],      # ROBOT-BASE frame
          "rotation":    [w, x, y, z],   # ROBOT-BASE frame (wxyz)
          "scale":       float,
        }

Response — batch IK (pickled dict):
    { "status": "ok",
      "joints":     np.ndarray(N, 7) float32,   # NaN rows = infeasible
      "success":    np.ndarray(N,) bool,
      "n_feasible": int, "n_total": int, "elapsed_s": float }

Response — full trajectory (pickled dict):
    { "status": "ok",
      "success":     bool,                       # False = no plan found
      "positions":   np.ndarray(T, 7) float32,   # joint trajectory @ `hz`
      "times":       np.ndarray(T,)  float32,    # seconds, 0 .. (T-1)*dt
      "dt": float, "hz": float,
      "n_waypoints": int, "duration_s": float, "elapsed_s": float }
    Or on failure (either path):  { "status": "error", "message": str }

Object translation/rotation arrive already in the ROBOT-BASE frame (the sam3d
server bakes in the camera->robot C2R), so they are used as-is — no conversion
here. IK / goal `poses` are likewise ROBOT-BASE frame — i.e. relative to the
robot base, NOT the world/env frame Isaac uses.

IK targets are at the cuRobo tool frame `gripper_tcp` — a virtual link 0.20407 m
along panda_hand's +z (i.e. 8.6 mm behind the jaw fingertip, the same tip-to-TCP
distance as the stock Franka; was 0.19267 m / 20 mm until 2026-09-05). Isaac's diff IK
controls panda_hand directly, so when comparing an IK target against an Isaac-
measured EE pose, account for that offset (or use the _tcp_pose_* helpers).

Run:
  python ik_server.py            # start the server
  python ik_server.py --test     # fire a quick random-pose IK client (table-only)
"""

import io
import os
import sys
import time
import pickle
import hashlib
import logging
import threading
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R_scipy

import zmq

# ─── cuRoboV2 public API ──────────────────────────────────────────────────────
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Scene, Cuboid, Mesh as CuMesh
from curobo.types import Pose, GoalToolPose, JointState
from curobo._src.util_file import join_path, load_yaml
from curobo.content import get_robot_configs_path
# Motion planner (cuRobo v2) — matches the getting_started.motion_planning API:
#   MotionPlannerCfg.create(robot=, scene_model=) -> MotionPlanner
#   planner.warmup(...); planner.plan_pose(goal, q_start); result.get_interpolated_plan()
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo.content import get_assets_path

# ─── cuRobo version and local edits ──────────────────────────────────────────
# This server was developed and the paper's experiments were run against cuRobo
# v2 (https://github.com/NVlabs/curobo) at commit ca941586c33b8482ed9c0e74d60f23efd64b516a
# (2026-04-18, "remove dates"), installed editable, with three local edits.
# The full diff is servers/curobo_local_edits.diff; apply it with
#     cd <curobo checkout> && git checkout ca94158 && git apply <this repo>/servers/curobo_local_edits.diff
# In words:
#   1. curobo/_src/curobolib/backends/__init__.py — a failed `cuda.core` backend
#      import is demoted from log_and_raise to log_warn, so cuRobo falls through
#      to its other backends instead of aborting at import time.
#   2. curobo/_src/geom/data/data_mesh.py — `wp.torch.device_from_torch` renamed
#      to `wp.device_from_torch` (Warp API change), and the mesh SDF query radius
#      in compute_local_sdf and compute_local_sdf_with_grad becomes
#      max(half bounding-box diagonal, obs_set.max_dist) instead of just the half
#      diagonal. Without this, points farther from a mesh than half its bounding
#      box report no distance, so collision costs vanish for small meshes (the
#      reconstructed objects here are a few cm across) and the IK solver plans
#      straight through them.
#   3. curobo/_src/perception/mapper/mesh_extractor.py — the same
#      `wp.torch.device_from_torch` -> `wp.device_from_torch` rename, 7 places.
# Edits 1 and 3 are compatibility fixes for the installed cuda/Warp versions.
# Edit 2 changes collision behaviour and is required to reproduce our results.


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit these for your rig
# ══════════════════════════════════════════════════════════════════════════════

ZMQ_PORT            = 5561
# batch_size x num_seeds is the real GPU footprint per launch. cuRobo's own
# intent is a solver sized to the problem (its cfg default is max_batch_size=1),
# so the handler clamps the effective batch to min(batch_size, #poses in the
# request) — batch_size is only a chunking CAP for large requests.
DEFAULT_BATCH_SIZE  = 512       # max IK poses per GPU launch (chunking cap)
DEFAULT_NUM_SEEDS   = 32        # IK seeds per pose (cuRobo v2 default)

DEFAULT_TRAJ_HZ     = 20.0      # output sampling rate of returned trajectories (Hz)
DEFAULT_MAX_ATTEMPTS = 10       # plan_pose retries before giving up (cuRobo def 5)
# MotionPlanner internal seed counts (NOT the batch-IK DEFAULT_NUM_SEEDS above).
# Cranked up from cuRobo defaults (32 / 4) to find paths for hard grasp goals
# (near table / joint limits, where stock seeds failed); costs more VRAM +
# warmup time per plan.
DEFAULT_PLANNER_IK_SEEDS     = 128   # cuRobo default 32
DEFAULT_PLANNER_TRAJOPT_SEEDS = 24   # cuRobo default 4

DEFAULT_VIZ_PORT    = 8082      # Viser web-viewer port (browse at http://host:PORT).
                                # Not 8080/8081: BCM's cmd.service owns those on
                                # cluster nodes; 8090/8091 are the robot bridge.
                                # 8082 is where viser's port auto-bump already
                                # landed, so existing tunnels keep working.
_VIZ_SPHERE_RADIUS  = 0.015     # candidate-point icosphere radius (m)
_VIZ_SPHERE_COLOR   = (255, 50, 50)

_MESH_MAX_FACES       = 5000                # decimation target per object mesh

# Keep these in lock-step with isaac_server.py / baseline_server.py so the IK
# collision table lines up exactly with the USD table Isaac spawns.
_ROBOT_BASE_POS_WORLD    = (0.0, 0.0, 1.05)   # robot base placement in world
_TABLE_TRANSLATION_WORLD = (0.55, 0.0, 1.05)  # USD table pivot in world
_TABLETOP_OFFSET_FROM_PRIM_Z = -0.003          # collider top relative to table prim
_TABLE_DIMS              = (2.0, 2.0, 0.05)
# SeattleLabTable's box collider tops out 3 mm below its prim origin, so its
# physical top is robot-z = -0.003 (world-z = 1.047). Place the planner cuboid's
# centre another half-thickness below that surface.
_TABLE_TOP_Z_ROBOT = (
    _TABLE_TRANSLATION_WORLD[2] - _ROBOT_BASE_POS_WORLD[2]
    + _TABLETOP_OFFSET_FROM_PRIM_Z
)
_TABLE_POSE = (
    _TABLE_TRANSLATION_WORLD[0] - _ROBOT_BASE_POS_WORLD[0],
    _TABLE_TRANSLATION_WORLD[1] - _ROBOT_BASE_POS_WORLD[1],
    _TABLE_TOP_Z_ROBOT - _TABLE_DIMS[2] / 2.0,
)
# "ignore_collisions" world: no objects, and the table shrunk to 1 mm and sunk
# far below the workspace. A degenerate-but-present cuboid (rather than an empty
# Scene) keeps the collision checker's world non-empty so nothing in cuRobo has
# to handle the zero-obstacle case; the arm (~0.85 m reach) can never touch it.
_IGNORE_COLL_TABLE_POSE = (0.0, 0.0, -100.0)
_IGNORE_COLL_TABLE_DIMS = (0.001, 0.001, 0.001)
# 2026-09-05: changed 0.19267 -> 0.20407 so the TCP sits 8.6 mm behind the rubber jaw
# tips, the same tip-to-TCP distance as the stock Franka (fingertip 0.112 m, TCP
# 0.1034 m from panda_hand). Every pose.json under baseline/baseline_results/
# generated before this date was produced with the old 20 mm value. Copies of
# this number that were changed with it: servers/isaac_server.py,
# semantic_grasp/robot.py, baseline/scripts/deploy_pose_no_collision.py; the
# 20 mm tip distance also lived in lerf_sim.py TCP_BEHIND_CONTACT_M and
# servers/baseline_server.py _FORK_TIP_Z (both now 0.0086).
_GRIPPER_TCP_OFFSET   = 0.20407             # panda_hand → TCP along +z (m): 0.07 standoff + 0.14267 jaw tip − 0.0086, i.e. TCP sits 8.6 mm behind the jaw fingertip
# cuRobo robot config for the rubber parallel-jaw gripper (Franka arm + new jaws).
# Collision spheres are baked into the yml. Mesh paths in the yml and the URDF
# are written with two placeholders that _resolve_urdf() fills in at load time:
#   {CUROBO_ASSETS}  cuRobo's bundled asset dir (the stock Franka link meshes)
#   {GRIPPER_DIR}    config/grippers/franka_rubber/ in this repo (palm + jaws)
# Repo root (this file lives in servers/) — derive asset paths from it.
# Repo root is the PARENT of servers/ (these scripts were moved into servers/;
# `.parent` alone points at servers/). Needed both for the config paths below
# and so `from semantic_grasp...` resolves when run as `python servers/ik_server.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ROBOT_CFG_YML  = str(_REPO_ROOT / "config/grippers/franka_rubber/franka_rubber.yml")
_ROBOT_URDF_TEMPLATE = _REPO_ROOT / "config/grippers/franka_rubber/franka_rubber.urdf"
_ROBOT_URDF = None   # set by _resolve_urdf() on first use


def _resolve_urdf() -> str:
    """Fill the {CUROBO_ASSETS} / {GRIPPER_DIR} placeholders in the gripper URDF
    and return the path of the resolved copy (written next to the template as
    franka_rubber.resolved.urdf, gitignored)."""
    global _ROBOT_URDF
    if _ROBOT_URDF is None:
        text = _ROBOT_URDF_TEMPLATE.read_text()
        text = text.replace("{CUROBO_ASSETS}", str(get_assets_path()))
        text = text.replace("{GRIPPER_DIR}", str(_ROBOT_URDF_TEMPLATE.parent))
        out = _ROBOT_URDF_TEMPLATE.with_suffix(".resolved.urdf")
        out.write_text(text)
        _ROBOT_URDF = str(out)
    return _ROBOT_URDF
# Default finger collision-sphere inflation (>1.0 inflates). This is only the
# FALLBACK now — every client method can override it per request via
# `gripper_sphere_scale`; see build_robot_cfg() and _gripper_scale().
_GRIPPER_SPHERE_SCALE = 1.2
# Whole-robot collision-sphere inflation, applied to EVERY link's spheres (arm
# included, not just the gripper). Raise it when the REAL arm trips its
# self-collision reflex on plans cuRobo accepted — the sphere model is too
# tight and needs margin everywhere; 1.1–1.2 gives 10–20 %. Set with the
# `--sphere-scale` CLI arg; on the gripper links it COMPOUNDS with
# `gripper_sphere_scale`. Server-wide, so changing it means restarting.
_ROBOT_SPHERE_SCALE = 1.0
# Extra padding (metres) added to every link's self_collision_buffer. Unlike
# --sphere-scale this makes ONLY the self-collision check more conservative --
# it costs no table/world clearance, so plans near the table keep working.
# Reach for this when the REAL arm's self-collision reflex trips on plans the
# server accepted (the reflex is a conservative joint-space boundary of the
# bare robot; the sphere model can be geometrically right and still cross it).
# Set with `--self-collision-pad`; attached_object is left alone (its buffer
# is part of grasp handling, not arm geometry).
_SELF_COLLISION_PAD = 0.0

# Y-up (GLB / glTF) → Z-up (robot) : +90° about X
_R_YUP_ZUP = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
])
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ik_server")


# ── helpers ───────────────────────────────────────────────────────────────────
def rot_matrix_from_quat(wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion → 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return R_scipy.from_quat([x, y, z, w]).as_matrix()


def euler_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    """(M, 3) extrinsic xyz Euler → (M, 4) wxyz quaternions (cuRobo convention)."""
    quat_xyzw = R_scipy.from_euler("xyz", rpy).as_quat()
    return np.concatenate(
        [quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1
    ).astype(np.float32)


def _resample_positions(pos: np.ndarray, src_dt: float, dst_dt: float) -> np.ndarray:
    """Linearly resample a (T_src, dof) joint-position trajectory from `src_dt`
    spacing to `dst_dt` spacing. The final planned config is kept as the last row
    so the robot still lands exactly on the goal. The source plan is already a
    smooth interpolation, so per-joint linear interp introduces no real error."""
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    T = pos.shape[0]
    if T < 2 or abs(src_dt - dst_dt) < 1e-9:
        return pos
    src_t = np.arange(T, dtype=np.float64) * src_dt
    duration = float(src_t[-1])
    n_dst = int(np.floor(duration / dst_dt)) + 1
    dst_t = np.arange(n_dst, dtype=np.float64) * dst_dt
    out = np.empty((n_dst, pos.shape[1]), dtype=np.float32)
    for j in range(pos.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, pos[:, j])
    if dst_t[-1] < duration - 1e-9:                    # keep the exact final config
        out = np.vstack([out, pos[-1][None, :]])
    return out


# ── scene build (decimated low-poly meshes + table cuboid) ─────────────────────
# cuRobo V2's mesh SDF (curobo/_src/geom/data/data_mesh.py, compute_local_sdf)
# only searches out to HALF THE MESH'S BOUNDING-BOX DIAGONAL, and when nothing
# is found within that radius it returns that radius AS the distance. The
# collision kernel then flags penetration = sphere_radius - distance > 0. So an
# obstacle whose half-diagonal is smaller than the largest robot sphere (0.06 m,
# panda_link1/3) puts those spheres in collision EVERYWHERE, and every IK pose
# in the workspace is rejected — regardless of where the obstacle is (measured
# 2026-09-06: a 6 cm cube anywhere -> 0/140 feasible; 8 cm cube -> fine). The
# pipeline hits this through perception's OBB box stand-ins for small flat
# neighbours (a spoon boxes to ~5x5x1 cm), which killed IK for every task in
# any scene containing one. Two defences, both here so the client stays dumb:
#   * box stand-ins (8-vertex / 12-face meshes) become cuRobo Cuboid primitives —
#     exact for a box, and the cuboid SDF has no search cutoff;
#   * any other mesh whose half-diagonal is below _MESH_MIN_HALF_DIAG gets two
#     UNREFERENCED padding vertices so its bounding box (which cuRobo takes from
#     the raw vertex list) is large enough; no face references them, so the
#     collision geometry itself is untouched.
_MESH_MIN_HALF_DIAG = 0.10   # m; > largest robot sphere (0.06) + activation margin
_BOX_PAD            = 0.06   # m; +-pad on every axis -> half-diagonal >= 0.104


def _box_standin_dims(v_local: np.ndarray, faces: np.ndarray):
    """If `v_local` (N,3) + `faces` describe an axis-aligned box in its own frame
    (perception's OBB stand-ins: trimesh.creation.box = 8 corners, 12 faces),
    return (dims, center); else None."""
    if len(v_local) != 8 or len(faces) != 12:
        return None
    lo, hi = v_local.min(axis=0), v_local.max(axis=0)
    dims, center = hi - lo, (hi + lo) / 2.0
    if np.any(dims <= 0):
        return None
    # every vertex must sit on a corner: |v - c| == dims/2 on all three axes
    if not np.allclose(np.abs(v_local - center), dims / 2.0, rtol=1e-5, atol=1e-7):
        return None
    return dims, center


def build_scene_from_objects(objects: list[dict],
                             table_pose=_TABLE_POSE,
                             table_dims=_TABLE_DIMS) -> Scene:
    """Build a fresh cuRobo Scene from objects handed in by the request. Each
    object's pose is already in the robot base frame and is placed as-is (only
    the GLB Y-up -> Z-up mesh fix and uniform scale are applied). Box stand-ins
    become Cuboid primitives and undersized meshes get bounding-box padding —
    see the note above build_scene_from_objects."""
    scene_meshes, scene_cuboids = [], []
    for obj in objects:
        name        = obj["name"]
        translation = obj["translation"]
        rotation    = obj["rotation"]
        scale       = obj["scale"]

        # Poses arrive already in the robot base frame (sam3d bakes in C2R).
        t_robot = np.array(translation, dtype=float)
        R_obj   = rot_matrix_from_quat(np.array(rotation, dtype=float))

        # Geometry rides along in the request — load it from in-memory bytes
        # (preferred) or raw vertices/faces, never from disk.
        if obj.get("mesh_bytes") is not None:
            mesh_format = str(obj.get("mesh_format", "glb")).lstrip(".").lower()
            m = trimesh.load(io.BytesIO(bytes(obj["mesh_bytes"])),
                             file_type=mesh_format, force="mesh")
        elif obj.get("vertices") is not None and obj.get("faces") is not None:
            m = trimesh.Trimesh(
                vertices=np.asarray(obj["vertices"], dtype=float).reshape(-1, 3),
                faces=np.asarray(obj["faces"], dtype=np.int64).reshape(-1, 3),
                process=False,
            )
        else:
            raise ValueError(f"object {name!r} has no mesh_bytes or vertices/faces")
        n_in = len(m.faces)
        if n_in > _MESH_MAX_FACES:
            m = m.simplify_quadric_decimation(face_count=_MESH_MAX_FACES)
        # Clean up degenerate / orphaned geometry that can crash Warp's BVH
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        m.merge_vertices()

        # Object frame, Z-up, metres (Y-up fix + scale, no rotation yet).
        v_local = (_R_YUP_ZUP @ np.asarray(m.vertices, dtype=float).T).T * scale
        faces_np = np.asarray(m.faces, dtype=int)

        box = _box_standin_dims(v_local, faces_np)
        if box is not None:
            # Exact cuboid at the box's own pose: request rotation (wxyz) is the
            # OBB frame, request translation is the OBB centre (the box mesh is
            # authored centred, so `center` is ~0; keep it for safety).
            dims, center = box
            q = np.asarray(rotation, dtype=float)
            q = q / (np.linalg.norm(q) + 1e-12)
            p = t_robot + R_obj @ center
            scene_cuboids.append(Cuboid(
                name=name.replace(" ", "_"),
                dims=[float(d) for d in dims],
                pose=[float(p[0]), float(p[1]), float(p[2]),
                      float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            ))
            log.info(f"  scene '{name}': box stand-in -> Cuboid "
                     f"{np.round(dims, 3).tolist()} m @ "
                     f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            continue

        verts = (R_obj @ v_local.T).T
        faces = faces_np.flatten().tolist()
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        half_diag = 0.5 * float(np.linalg.norm(hi - lo))
        padded = ""
        if half_diag < _MESH_MIN_HALF_DIAG:
            # Unreferenced padding vertices: they widen the bounding box cuRobo
            # derives its SDF search radius from, and nothing else.
            c = (hi + lo) / 2.0
            pad = np.array([[_BOX_PAD, _BOX_PAD, _BOX_PAD]])
            verts = np.vstack([verts, c + pad, c - pad])
            padded = (f" (half-diagonal {half_diag * 100:.1f} cm < "
                      f"{_MESH_MIN_HALF_DIAG * 100:.0f} cm: bbox padded)")
        scene_meshes.append(CuMesh(
            name=name.replace(" ", "_"),
            vertices=verts.tolist(),
            faces=faces,
            pose=[float(t_robot[0]), float(t_robot[1]), float(t_robot[2]),
                  1.0, 0.0, 0.0, 0.0],
        ))
        log.info(f"  scene '{name}': {n_in} → {len(faces_np)} faces, "
                 f"{len(verts)} verts @ "
                 f"({t_robot[0]:.3f}, {t_robot[1]:.3f}, {t_robot[2]:.3f}){padded}")

    table = Cuboid(
        name="table",
        dims=list(table_dims),
        pose=[table_pose[0], table_pose[1], table_pose[2], 1.0, 0.0, 0.0, 0.0],
    )
    return Scene(mesh=scene_meshes, cuboid=[table] + scene_cuboids)


# ── robot config: gripper TCP extra link + finger sphere inflation ─────────────
def build_robot_cfg(gripper_sphere_scale: float = _GRIPPER_SPHERE_SCALE) -> dict:
    """Build the cuRobo robot config. `gripper_sphere_scale` multiplies the radius
    of every collision sphere on the gripper links (panda_hand / leftfinger /
    rightfinger) — >1.0 inflates them for more conservative clearance. It's a plain
    argument so callers can set it per request (clients pass it through); the
    default is the module-level _GRIPPER_SPHERE_SCALE. On top of that,
    _ROBOT_SPHERE_SCALE (CLI `--sphere-scale`) inflates EVERY link's spheres."""
    rcfg = load_yaml(_ROBOT_CFG_YML)
    kin = rcfg["robot_cfg"]["kinematics"]
    kin["urdf_path"] = _resolve_urdf()   # placeholders in the yml/URDF filled here
    kin.setdefault("extra_links", {})
    kin["extra_links"]["gripper_tcp"] = {
        "link_name":        "gripper_tcp",
        "parent_link_name": "panda_hand",
        "joint_name":       "gripper_tcp_joint",
        "joint_type":       "FIXED",
        "fixed_transform":  [0, 0, _GRIPPER_TCP_OFFSET, 1, 0, 0, 0],
    }
    kin["tool_frames"] = ["gripper_tcp"]
    if _SELF_COLLISION_PAD != 0.0:
        buf = kin.get("self_collision_buffer", {})
        for link in buf:
            if link != "attached_object":
                buf[link] = float(buf[link]) + _SELF_COLLISION_PAD
    spheres = kin.get("collision_spheres", {})
    if isinstance(spheres, dict):
        # Whole-robot inflation first (global, read at call time so the CLI
        # assignment is seen), then the per-request gripper inflation on top.
        if _ROBOT_SPHERE_SCALE != 1.0:
            for link_spheres in spheres.values():
                for s in link_spheres:
                    s["radius"] *= _ROBOT_SPHERE_SCALE
        if gripper_sphere_scale != 1.0:
            for link in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
                for s in spheres.get(link, []):
                    s["radius"] *= gripper_sphere_scale
    return rcfg


def _gripper_scale(req: dict) -> float:
    """Per-request finger collision-sphere inflation. Falls back to the module
    default _GRIPPER_SPHERE_SCALE when the client doesn't send one, so a client
    that never sets it keeps the previous behaviour."""
    return float(req.get("gripper_sphere_scale", _GRIPPER_SPHERE_SCALE))


def _validate_gripper_scale(req) -> tuple[bool, str | None]:
    g = req.get("gripper_sphere_scale")
    if g is not None and not (isinstance(g, (int, float)) and g > 0):
        return False, "'gripper_sphere_scale' must be a positive number"
    return True, None


def make_ik_solver(scene: Scene, rcfg: dict, num_seeds: int, batch_size: int):
    config = InverseKinematicsCfg.create(
        robot=rcfg,
        scene_model=scene,
        num_seeds=num_seeds,
        max_batch_size=batch_size,
        self_collision_check=True,
    )
    return InverseKinematics(config)


def _objects_digest(objects: list[dict], table_pose, table_dims,
                    num_seeds: int, batch_size: int,
                    gripper_sphere_scale: float) -> str:
    """Stable content hash of everything baked into the IK solver at create() time:
    the object geometry/poses, the table, the seed/batch sizes, and the gripper
    sphere inflation (which changes the robot's collision model). Two requests with
    the same digest yield an identical solver, so a cached one can be reused instead
    of rebuilt (which would recapture CUDA graphs — the slow part).

    Conservative on ordering: a different object order hashes differently and just
    forces a rebuild; it never reuses a solver for a scene that isn't bit-identical.
    """
    h = hashlib.sha256()
    h.update(repr((tuple(table_pose), tuple(table_dims),
                   int(num_seeds), int(batch_size),
                   float(gripper_sphere_scale),
                   float(_ROBOT_SPHERE_SCALE),
                   float(_SELF_COLLISION_PAD))).encode())
    for obj in objects:
        h.update(str(obj.get("name", "")).encode())
        h.update(np.asarray(obj.get("translation", []), dtype=np.float64).tobytes())
        h.update(np.asarray(obj.get("rotation", []), dtype=np.float64).tobytes())
        h.update(np.float64(obj.get("scale", 1.0)).tobytes())
        if obj.get("mesh_bytes") is not None:
            h.update(b"mb")
            h.update(bytes(obj["mesh_bytes"]))
        else:
            h.update(b"vf")
            h.update(np.ascontiguousarray(obj.get("vertices", []), dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(obj.get("faces", []), dtype=np.int64).tobytes())
    return h.hexdigest()


def make_motion_planner(scene: Scene, rcfg: dict,
                        num_ik_seeds: int = DEFAULT_PLANNER_IK_SEEDS,
                        num_trajopt_seeds: int = DEFAULT_PLANNER_TRAJOPT_SEEDS):
    """Build a collision-aware MotionPlanner for this scene and warm it up.

    num_ik_seeds / num_trajopt_seeds are the planner's INTERNAL seed counts
    (cuRobo defaults 32 / 4). More seeds = higher chance of finding a path for a
    hard goal, at the cost of more VRAM and longer warmup/plan time.

    NOTE the warmup cost: warmup() pre-compiles CUDA graphs (~seconds). Because
    the scene arrives per request we rebuild + warm up every call. If you only
    ever move (not add/remove) obstacles, build the planner ONCE and instead call
    planner.scene_collision_checker.update_obstacle_pose(name, Pose) per request
    — far faster. Set enable_graph=False below to skip graph capture if you'd
    rather trade per-plan speed for lower build latency.
    """
    config = MotionPlannerCfg.create(
        robot=rcfg, scene_model=scene,
        num_ik_seeds=num_ik_seeds, num_trajopt_seeds=num_trajopt_seeds,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    return planner


# ── core IK sweep over a (N, 6) xyz-rpy matrix ─────────────────────────────────
def solve_orientations(ik, target_link, xyz: np.ndarray, quat_wxyz: np.ndarray,
                       batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Chunked collision-free IK. Returns (joints (N,7) float32, success (N,) bool).
    Infeasible rows in `joints` are NaN.
    """
    N        = int(xyz.shape[0])
    n_chunks = int((N + batch_size - 1) // batch_size)

    joints  = np.full((N, 7), np.nan, dtype=np.float32)
    success = np.zeros(N, dtype=bool)

    pos_t  = torch.as_tensor(xyz,       device="cuda", dtype=torch.float32)
    quat_t = torch.as_tensor(quat_wxyz, device="cuda", dtype=torch.float32)

    for ci in range(n_chunks):
        s = ci * batch_size
        e = min(s + batch_size, N)
        n = e - s

        # Pad the (possibly short) final chunk up to batch_size so a single CUDA
        # graph captured at max_batch_size covers every launch.
        if n < batch_size:
            positions = torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32)
            positions[:n] = pos_t[s:e]
            quaternions = torch.zeros((batch_size, 4), device="cuda", dtype=torch.float32)
            quaternions[:, 0] = 1.0                       # identity quaternion
            quaternions[:n] = quat_t[s:e]
        else:
            positions   = pos_t[s:e].contiguous()
            quaternions = quat_t[s:e].contiguous()

        goal_poses = Pose(position=positions, quaternion=quaternions)
        result = ik.solve_pose(
            GoalToolPose.from_poses({target_link: goal_poses}, num_goalset=1)
        )
        ok  = result.success.view(-1).cpu().numpy()[:n].astype(bool)
        sol = result.solution.view(-1, 7).cpu().numpy()[:n]   # best seed per pose

        idx = np.arange(s, e)[ok]
        joints[idx]      = sol[ok]
        success[s:e]     = ok

    return joints, success


# ── core trajectory solve: start config → single EE goal ───────────────────────
def solve_trajectory(planner, xyz: np.ndarray, quat_wxyz: np.ndarray,
                     start_q, max_attempts: int):
    """Plan one collision-free trajectory from `start_q` to the goal EE pose.

    start_q : length-7 start config, or None to use planner.default_joint_state.
    Returns the interpolated joint-position trajectory (T, dof) at the planner's
    native interpolation_dt, or None if no plan was found. Caller trims to 7 dof
    and resamples to the requested rate.
    """
    target_link = planner.tool_frames[0]

    goal = Pose(
        position=torch.as_tensor(xyz.reshape(1, 3),         device="cuda", dtype=torch.float32),
        quaternion=torch.as_tensor(quat_wxyz.reshape(1, 4), device="cuda", dtype=torch.float32),
    )
    goal_pose = GoalToolPose.from_poses({target_link: goal}, num_goalset=1)

    if start_q is None:
        q_start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
    else:
        q = np.asarray(start_q, dtype=np.float32).reshape(1, -1)
        q_start = JointState.from_position(
            torch.as_tensor(q, device="cuda", dtype=torch.float32),
            joint_names=planner.joint_names,
        )

    result = planner.plan_pose(goal_pose, q_start, max_attempts=max_attempts)
    if result is None or not bool(result.success.any()):
        return None

    interp = result.get_interpolated_plan()            # JointState @ interpolation_dt
    pos = interp.position.detach().cpu().numpy().astype(np.float32)
    return pos.reshape(-1, pos.shape[-1])              # (T, dof)


# ── core penetration query: robot collision spheres vs scene obstacles ─────────
def solve_penetration(ik, joints: np.ndarray, activation_distance: float = 0.0):
    """Total world-collision penetration of the robot at each joint config.

    For each config we run FK to get the robot's collision spheres — the SAME
    sphere model IK / the motion planner collision-check against, including the
    inflated gripper spheres and the gripper_tcp link from build_robot_cfg() — and
    sum, over every sphere, how far it pokes into any scene obstacle (table +
    objects). This is robot-vs-SCENE only; it does NOT include self-collision.

    joints : (N, 7) arm configs (e.g. IK solutions). Non-finite rows return NaN.
    Returns penetration (N,) float32: 0.0 = collision-free, larger = deeper. With
    activation_distance > 0 the cost also ramps up within that margin OUTSIDE the
    obstacle (useful as a soft clearance signal), so it's no longer pure depth.
    """
    scene_coll = ik.scene_collision_checker
    device = "cuda"

    q_np = np.ascontiguousarray(joints, dtype=np.float32).reshape(-1, 7)
    finite = np.isfinite(q_np).all(axis=1)
    # Zero out non-finite rows so the FK kernel doesn't choke; overwritten w/ NaN.
    q_safe = np.where(finite[:, None], q_np, 0.0).astype(np.float32)

    q = torch.as_tensor(q_safe, device=device, dtype=torch.float32)
    state = ik.compute_kinematics(JointState.from_position(q))
    spheres = state.robot_spheres                      # (N, horizon, n_spheres, 4)

    buf    = CollisionBuffer.from_shape(spheres.shape, scene_coll.device_cfg)
    weight = torch.ones(1, device=device, dtype=torch.float32)
    act    = torch.full((1,), float(activation_distance), device=device, dtype=torch.float32)

    d = scene_coll.get_sphere_distance(state, buf, weight, act, return_loss=False)
    pen = d.sum(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)   # (N,)
    pen[~finite] = np.nan
    return pen


# ── request validation ─────────────────────────────────────────────────────────
def _validate_objects(objs: list) -> tuple[bool, str | None]:
    """Shared object-list checks for both the IK and trajectory paths."""
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            return False, f"objects[{i}] is not a dict"
        if not o.get("name"):
            return False, f"objects[{i}] missing 'name'"
        mb = o.get("mesh_bytes")
        has_bytes  = isinstance(mb, (bytes, bytearray, memoryview)) and len(mb) > 0
        has_arrays = o.get("vertices") is not None and o.get("faces") is not None
        if not (has_bytes or has_arrays):
            return False, (f"objects[{i}] needs in-request geometry: "
                           f"'mesh_bytes' (+ 'mesh_format') or 'vertices'+'faces'")
        if has_bytes and not o.get("mesh_format"):
            return False, f"objects[{i}] 'mesh_bytes' requires 'mesh_format' (e.g. 'glb')"
        if o.get("translation") is None or len(o["translation"]) != 3:
            return False, f"objects[{i}] 'translation' must be length-3"
        if o.get("rotation") is None or len(o["rotation"]) != 4:
            return False, f"objects[{i}] 'rotation' must be length-4 (wxyz)"
        if o.get("scale") is None:
            return False, f"objects[{i}] missing 'scale'"
    return True, None


def validate_request(req) -> tuple[bool, str | None]:
    if not isinstance(req, dict):
        return False, "request is not a dict"
    ok, err = _validate_gripper_scale(req)
    if not ok:
        return False, err
    objs = req.get("objects")
    if objs is None or not isinstance(objs, list):
        return False, "'objects' must be a list (may be empty for a table-only scene)"
    ok, err = _validate_objects(objs)
    if not ok:
        return False, err
    try:
        poses = np.asarray(req.get("poses"), dtype=np.float32)
    except Exception as e:
        return False, f"'poses' not array-like: {e}"
    if poses.ndim != 2 or poses.shape[1] != 6:
        return False, f"'poses' must be (N,6) [x,y,z,r,p,y], got {poses.shape}"
    return True, None


def validate_traj_request(req) -> tuple[bool, str | None]:
    if not isinstance(req, dict):
        return False, "request is not a dict"
    ok, err = _validate_gripper_scale(req)
    if not ok:
        return False, err
    objs = req.get("objects")
    if objs is None or not isinstance(objs, list):
        return False, "'objects' must be a list (may be empty for a table-only scene)"
    ok, err = _validate_objects(objs)
    if not ok:
        return False, err
    try:
        pose = np.asarray(req.get("pose"), dtype=np.float32).reshape(-1)
    except Exception as e:
        return False, f"'pose' not array-like: {e}"
    if pose.shape[0] != 6:
        return False, f"'pose' must be length-6 [x,y,z,r,p,y], got {pose.shape}"
    ss = req.get("start_state")
    if ss is not None:
        try:
            ss = np.asarray(ss, dtype=np.float32).reshape(-1)
        except Exception as e:
            return False, f"'start_state' not array-like: {e}"
        if ss.shape[0] != 7:
            return False, f"'start_state' must be length-7, got {ss.shape}"
    hz = req.get("hz", DEFAULT_TRAJ_HZ)
    if not (isinstance(hz, (int, float)) and hz > 0):
        return False, "'hz' must be a positive number"
    return True, None


# ── trajectory request handler (builds scene + planner, plans, packs reply) ────
def run_traj_request(req: dict) -> dict:
    ok, err = validate_traj_request(req)
    if not ok:
        return {"status": "error", "message": err}

    rcfg         = build_robot_cfg(_gripper_scale(req))
    hz           = float(req.get("hz", DEFAULT_TRAJ_HZ))
    dst_dt       = 1.0 / hz
    max_attempts = int(req.get("max_attempts", DEFAULT_MAX_ATTEMPTS))
    ignore_coll  = bool(req.get("ignore_collisions", False))
    tbl          = req.get("table") or {}
    table_pose   = tuple(tbl.get("pose", _TABLE_POSE))
    table_dims   = tuple(tbl.get("dims", _TABLE_DIMS))

    pose = np.asarray(req["pose"], dtype=np.float32).reshape(6)
    xyz  = pose[:3]
    quat = euler_to_wxyz(pose[3:6].reshape(1, 3))[0]
    start_q = req.get("start_state")                   # list or None

    t0 = time.time()
    if ignore_coll:
        # Plan against an empty world: objects dropped, table degenerate + banished.
        # Self-collision and joint limits still apply (baked into the robot model).
        log.info("solve_traj: ignore_collisions=True — planning with no world obstacles")
        scene = build_scene_from_objects([], _IGNORE_COLL_TABLE_POSE,
                                         _IGNORE_COLL_TABLE_DIMS)
    else:
        scene = build_scene_from_objects(req["objects"], table_pose, table_dims)
    t_scene = time.time()

    planner = make_motion_planner(scene, rcfg)
    src_dt = float(planner.trajopt_solver.config.interpolation_dt)
    t_planner = time.time()

    try:
        pos = solve_trajectory(planner, xyz, quat, start_q, max_attempts)
    finally:
        del planner, scene
        torch.cuda.empty_cache()
    t_plan = time.time()

    if pos is None:
        log.info(f"traj plan FAILED · scene {t_scene-t0:.2f}s | "
                 f"planner {t_planner-t_scene:.2f}s | plan {t_plan-t_planner:.2f}s")
        return {
            "status":      "ok",
            "success":     False,
            "positions":   np.zeros((0, 7), dtype=np.float32),
            "times":       np.zeros((0,),   dtype=np.float32),
            "dt": dst_dt, "hz": hz,
            "n_waypoints": 0,
            "duration_s":  0.0,
            "elapsed_s":   round(float(t_plan - t_planner), 4),
        }

    # Trim finger cols (keep the 7 arm joints) and resample to the requested rate.
    positions = _resample_positions(pos[:, :7], src_dt, dst_dt)
    T     = int(positions.shape[0])
    times = (np.arange(T, dtype=np.float32) * dst_dt)
    log.info(f"traj OK · {T} waypoints @ {hz:.0f}Hz ({(T-1)*dst_dt:.2f}s, "
             f"native dt {src_dt:.3f}s) · scene {t_scene-t0:.2f}s | "
             f"planner {t_planner-t_scene:.2f}s | plan {t_plan-t_planner:.2f}s")
    return {
        "status":      "ok",
        "success":     True,
        "positions":   np.ascontiguousarray(positions),
        "times":       np.ascontiguousarray(times),
        "dt": dst_dt, "hz": hz,
        "n_waypoints": T,
        "duration_s":  round(float((T - 1) * dst_dt), 4),
        "elapsed_s":   round(float(t_plan - t_planner), 4),
    }


# ── penetration request handler (builds scene + solver, FK, collision query) ───
def validate_penetration_request(req) -> tuple[bool, str | None]:
    if not isinstance(req, dict):
        return False, "request is not a dict"
    ok, err = _validate_gripper_scale(req)
    if not ok:
        return False, err
    objs = req.get("objects")
    if objs is None or not isinstance(objs, list):
        return False, "'objects' must be a list (may be empty for a table-only scene)"
    ok, err = _validate_objects(objs)
    if not ok:
        return False, err
    try:
        joints = np.asarray(req.get("joints"), dtype=np.float32)
    except Exception as e:
        return False, f"'joints' not array-like: {e}"
    if joints.ndim != 2 or joints.shape[1] != 7:
        return False, f"'joints' must be (N,7), got {joints.shape}"
    ad = req.get("activation_distance", 0.0)
    if not (isinstance(ad, (int, float)) and ad >= 0):
        return False, "'activation_distance' must be a non-negative number"
    return True, None


def run_penetration_request(req: dict) -> dict:
    ok, err = validate_penetration_request(req)
    if not ok:
        return {"status": "error", "message": err}

    rcfg       = build_robot_cfg(_gripper_scale(req))
    num_seeds  = int(req.get("num_seeds",  DEFAULT_NUM_SEEDS))
    tbl        = req.get("table") or {}
    table_pose = tuple(tbl.get("pose", _TABLE_POSE))
    table_dims = tuple(tbl.get("dims", _TABLE_DIMS))
    act_dist   = float(req.get("activation_distance", 0.0))

    joints = np.asarray(req["joints"], dtype=np.float32)

    t0 = time.time()
    scene = build_scene_from_objects(req["objects"], table_pose, table_dims)
    t_scene = time.time()
    # We only need FK + the scene collision checker, not solve_pose, so the IK
    # solver's batch size is irrelevant here (compute_kinematics resizes FK
    # buffers to whatever batch we hand it).
    ik = make_ik_solver(scene, rcfg, num_seeds, DEFAULT_BATCH_SIZE)
    t_solver = time.time()
    try:
        pen = solve_penetration(ik, joints, act_dist)
    finally:
        del ik, scene
        torch.cuda.empty_cache()
    t_pen = time.time()

    in_collision = np.isfinite(pen) & (pen > 0.0)
    log.info(f"penetration · {joints.shape[0]} configs · in-collision "
             f"{int(in_collision.sum())}/{joints.shape[0]} · "
             f"scene {t_scene - t0:.2f}s | solver {t_solver - t_scene:.2f}s | "
             f"query {t_pen - t_solver:.2f}s")
    return {
        "status":       "ok",
        "penetration":  np.ascontiguousarray(pen),
        "in_collision": np.ascontiguousarray(in_collision),
        "n_total":      int(joints.shape[0]),
        "elapsed_s":    round(float(t_pen - t_solver), 4),
    }


# ── Viser scene visualization ──────────────────────────────────────────────────
# A single long-lived viewer is kept alive across requests (viser serves it from a
# background thread, so the server keeps handling requests while the page stays up).
_VIZ        = None          # the live ViserVisualizer
_VIZ_META   = None          # (port, host, show_spheres, gripper_scale) viewer was built with
_VIZ_PTS    = []            # icosphere handles we added, so we can clear them
_VIZ_OBJS   = []            # /obstacles/* node names added, so we can clear them


def validate_viz_request(req) -> tuple[bool, str | None]:
    if not isinstance(req, dict):
        return False, "request is not a dict"
    ok, err = _validate_gripper_scale(req)
    if not ok:
        return False, err
    objs = req.get("objects", [])
    if not isinstance(objs, list):
        return False, "'objects' must be a list (may be empty for a table-only scene)"
    if objs:                                           # objects are optional here
        ok, err = _validate_objects(objs)
        if not ok:
            return False, err
    pts = req.get("points")
    if pts is not None:
        try:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        except Exception as e:
            return False, f"'points' must be (N,3): {e}"
    jc = req.get("joint_config")
    if jc is not None:
        try:
            q = np.asarray(jc, dtype=np.float32).reshape(-1)
        except Exception as e:
            return False, f"'joint_config' not array-like: {e}"
        if q.shape[0] != 7:
            return False, f"'joint_config' must be length-7, got {q.shape}"
    return True, None


def run_viz_request(req: dict) -> dict:
    """Launch (or refresh) a Viser web view of the cuRobo collision scene. With
    show_spheres=True the robot's collision-sphere model is drawn (this is the same
    sphere decomposition IK collision-checks against). Optional `points` (N,3,
    ROBOT frame) are scattered as icospheres — handy for showing IK targets or
    grasp candidates. Optional `joint_config` (7,) poses the robot so the spheres
    sit at a real configuration."""
    global _VIZ, _VIZ_META, _VIZ_PTS, _VIZ_OBJS

    ok, err = validate_viz_request(req)
    if not ok:
        return {"status": "error", "message": err}

    try:
        from curobo.viewer import ViserVisualizer
    except Exception as exc:
        return {"status": "error",
                "message": f"viser/viewer not available in this env: {exc}"}

    objs         = req.get("objects", [])
    show_spheres = bool(req.get("show_spheres", False))
    port         = int(req.get("port", DEFAULT_VIZ_PORT))
    host         = str(req.get("host", "0.0.0.0"))
    scale        = _gripper_scale(req)
    tbl          = req.get("table") or {}
    table_pose   = tuple(tbl.get("pose", _TABLE_POSE))
    table_dims   = tuple(tbl.get("dims", _TABLE_DIMS))

    # Build the same collision scene solve() would (poses are robot-frame as-is).
    scene = build_scene_from_objects(objs, table_pose, table_dims)

    # visualize_robot_spheres is a constructor flag, so changing it (or the port,
    # or the gripper sphere scale, which bakes into the robot's sphere model) means
    # rebuilding the viewer. Best-effort close the old one first.
    need_new = (_VIZ is None) or (_VIZ_META != (port, host, show_spheres, scale))
    if need_new:
        if _VIZ is not None:
            for closer in ("stop", "close", "shutdown"):
                try:
                    getattr(_VIZ, closer)()
                    break
                except Exception:
                    pass
            _VIZ = None
        try:
            # Pass the scaled robot_cfg dict (not a ContentPath) so the viewer
            # sees the per-request gripper sphere scale; ContentPath would re-read
            # franka.yml from disk and bypass build_robot_cfg's sphere inflation.
            _VIZ = ViserVisualizer(
                content_path=build_robot_cfg(scale),
                connect_ip=host,
                connect_port=port,
                add_control_frames=False,
                visualize_robot_spheres=show_spheres,
            )
        except Exception as exc:
            _VIZ = _VIZ_META = None
            return {"status": "error",
                    "message": f"could not launch Viser on {host}:{port} "
                               f"(port in use? try a different port): {exc}"}
        _VIZ_META = (port, host, show_spheres, scale)
        _VIZ_PTS = []
        _VIZ_OBJS = []

    viz = _VIZ

    # Drop the obstacle nodes from the previous visualize() call before re-adding —
    # otherwise add_scene stacks new meshes on top of the old ones (and viser's
    # persistent-message replay shows the cumulative pile to every new browser).
    for node_name in _VIZ_OBJS:
        try:
            viz._server.scene.remove_by_name(node_name)
        except Exception:
            pass
    _VIZ_OBJS = []

    try:
        viz.add_scene(scene)
        # Mirror what add_scene names each obstacle so we can clean them up next call.
        for m in scene.mesh:
            _VIZ_OBJS.append("/obstacles/" + m.name)
    except Exception as exc:
        log.warning(f"viz.add_scene failed: {exc}")

    # Pose the robot at a specific config so the spheres reflect it (best-effort).
    jc = req.get("joint_config")
    if jc is not None:
        try:
            from curobo.types import JointState
            q = np.asarray(jc, dtype=np.float32).reshape(1, -1)
            viz.set_joint_state(JointState.from_position(
                torch.as_tensor(q, device="cuda", dtype=torch.float32)))
        except Exception as exc:
            log.warning(f"viz.set_joint_state failed (robot left at default): {exc}")

    # Clear any candidate spheres from a previous call, then scatter the new ones.
    for h in _VIZ_PTS:
        try:
            h.remove()
        except Exception:
            pass
    _VIZ_PTS = []
    n_pts = 0
    pts = req.get("points")
    if pts is not None:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        for pi, pt in enumerate(pts):
            try:
                h = viz._server.scene.add_icosphere(
                    f"/candidate_{pi}",
                    radius=_VIZ_SPHERE_RADIUS, color=_VIZ_SPHERE_COLOR,
                    position=(float(pt[0]), float(pt[1]), float(pt[2])),
                )
                _VIZ_PTS.append(h)
                n_pts += 1
            except Exception as exc:
                log.warning(f"add_icosphere {pi} failed: {exc}")

    url = f"http://localhost:{port}"
    log.info(f"Viser running — connect at {url}  "
             f"(robot_spheres={'on' if show_spheres else 'off'}, "
             f"objects={len(objs)}, points={n_pts})")
    return {
        "status":       "ok",
        "url":          url,
        "port":         port,
        "show_spheres": show_spheres,
        "n_objects":    len(objs),
        "n_points":     n_pts,
    }


# ── viz_traj: plan a trajectory and loop-play it in the viewer ────────────────
# Module-level state for the looping playback thread. We keep exactly one
# playback running at a time — a new viz_traj request cancels the previous loop
# before swapping the trajectory in.
_VIZ_PLAY_THREAD = None
_VIZ_PLAY_STOP   = None


def _stop_playback():
    global _VIZ_PLAY_THREAD, _VIZ_PLAY_STOP
    if _VIZ_PLAY_STOP is not None:
        _VIZ_PLAY_STOP.set()
    if _VIZ_PLAY_THREAD is not None and _VIZ_PLAY_THREAD.is_alive():
        _VIZ_PLAY_THREAD.join(timeout=2.0)
    _VIZ_PLAY_THREAD = None
    _VIZ_PLAY_STOP   = None


def _start_playback(viz, positions: np.ndarray, dt: float, joint_names: list[str]):
    """Spawn a daemon thread that cycles `positions` through `viz.set_joint_state`
    forever, sleeping `dt` between frames. `joint_names` must match
    `positions.shape[1]` — viser's set_joint_state iterates joint_state.joint_names,
    so passing None there raises 'NoneType is not iterable'. Missing joints (e.g.
    fingers when positions is arm-only) are backfilled by cuRobo's get_full_js."""
    global _VIZ_PLAY_THREAD, _VIZ_PLAY_STOP
    _stop_playback()

    T = int(positions.shape[0])
    if T == 0:
        return

    stop_event = threading.Event()
    pos_cuda = torch.as_tensor(positions, device="cuda", dtype=torch.float32)

    def loop():
        i = 0
        while not stop_event.is_set():
            try:
                viz.set_joint_state(JointState.from_position(
                    pos_cuda[i:i + 1], joint_names=joint_names))
            except Exception as exc:
                log.warning(f"viz_traj playback set_joint_state failed: {exc}")
                return
            i = (i + 1) % T
            if stop_event.wait(timeout=dt):
                return

    _VIZ_PLAY_STOP   = stop_event
    _VIZ_PLAY_THREAD = threading.Thread(target=loop, name="viz_traj_playback", daemon=True)
    _VIZ_PLAY_THREAD.start()


def _arm_joint_names(rcfg: dict, n: int = 7) -> list[str]:
    """First n joint names from the cspace — Franka has 7 arm joints then 2 fingers."""
    return list(rcfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"][:n])


def validate_viz_traj_request(req) -> tuple[bool, str | None]:
    ok, err = validate_traj_request(req)
    if not ok:
        return False, err
    pts = req.get("points")
    if pts is not None:
        try:
            np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        except Exception as e:
            return False, f"'points' must be (N,3): {e}"
    return True, None


def run_viz_traj_request(req: dict) -> dict:
    """Plan a trajectory to a single EE goal (same path as solve_traj) and replay
    it in a loop inside the Viser viewer. Optional `points` and `show_spheres`
    match run_viz_request semantics."""
    global _VIZ, _VIZ_META, _VIZ_PTS, _VIZ_OBJS

    ok, err = validate_viz_traj_request(req)
    if not ok:
        return {"status": "error", "message": err}

    try:
        from curobo.viewer import ViserVisualizer
    except Exception as exc:
        return {"status": "error",
                "message": f"viser/viewer not available in this env: {exc}"}

    scale        = _gripper_scale(req)
    rcfg         = build_robot_cfg(scale)
    hz           = float(req.get("hz", DEFAULT_TRAJ_HZ))
    dst_dt       = 1.0 / hz
    max_attempts = int(req.get("max_attempts", DEFAULT_MAX_ATTEMPTS))
    ignore_coll  = bool(req.get("ignore_collisions", False))
    tbl          = req.get("table") or {}
    table_pose   = tuple(tbl.get("pose", _TABLE_POSE))
    table_dims   = tuple(tbl.get("dims", _TABLE_DIMS))
    show_spheres = bool(req.get("show_spheres", True))
    port         = int(req.get("port", DEFAULT_VIZ_PORT))
    host         = str(req.get("host", "0.0.0.0"))

    pose = np.asarray(req["pose"], dtype=np.float32).reshape(6)
    xyz  = pose[:3]
    quat = euler_to_wxyz(pose[3:6].reshape(1, 3))[0]
    start_q = req.get("start_state")
    objs    = req.get("objects", [])

    # ── plan first; if no path exists, fail before touching the viewer ────────
    # `scene` is what the viewer shows — always the REAL objects + table, so with
    # ignore_collisions you can watch the trajectory sweep through them; only the
    # planner's world (`plan_scene`) is emptied.
    t0      = time.time()
    scene   = build_scene_from_objects(objs, table_pose, table_dims)
    if ignore_coll:
        log.info("viz_traj: ignore_collisions=True — planning with no world obstacles")
        plan_scene = build_scene_from_objects([], _IGNORE_COLL_TABLE_POSE,
                                              _IGNORE_COLL_TABLE_DIMS)
    else:
        plan_scene = scene
    t_scene = time.time()
    planner = make_motion_planner(plan_scene, rcfg)
    src_dt  = float(planner.trajopt_solver.config.interpolation_dt)
    t_planner = time.time()
    try:
        pos = solve_trajectory(planner, xyz, quat, start_q, max_attempts)
    finally:
        del planner
        torch.cuda.empty_cache()
    t_plan = time.time()

    if pos is None:
        return {"status": "error",
                "message": ("trajectory planning failed (goal unreachable — "
                            "collisions were ignored)" if ignore_coll else
                            "trajectory planning failed (no collision-free path)")}

    positions = _resample_positions(pos[:, :7], src_dt, dst_dt).astype(np.float32)

    # Stop any prior playback before swapping the scene out from under it.
    _stop_playback()

    # ── viewer setup (mirrors run_viz_request) ────────────────────────────────
    need_new = (_VIZ is None) or (_VIZ_META != (port, host, show_spheres, scale))
    if need_new:
        if _VIZ is not None:
            for closer in ("stop", "close", "shutdown"):
                try:
                    getattr(_VIZ, closer)()
                    break
                except Exception:
                    pass
            _VIZ = None
        try:
            _VIZ = ViserVisualizer(
                content_path=build_robot_cfg(scale),
                connect_ip=host,
                connect_port=port,
                add_control_frames=False,
                visualize_robot_spheres=show_spheres,
            )
        except Exception as exc:
            _VIZ = _VIZ_META = None
            return {"status": "error",
                    "message": f"could not launch Viser on {host}:{port}: {exc}"}
        _VIZ_META = (port, host, show_spheres, scale)
        _VIZ_PTS  = []
        _VIZ_OBJS = []

    viz = _VIZ

    for node_name in _VIZ_OBJS:
        try:
            viz._server.scene.remove_by_name(node_name)
        except Exception:
            pass
    _VIZ_OBJS = []
    try:
        viz.add_scene(scene)
        for m in scene.mesh:
            _VIZ_OBJS.append("/obstacles/" + m.name)
    except Exception as exc:
        log.warning(f"viz.add_scene failed: {exc}")

    for h in _VIZ_PTS:
        try:
            h.remove()
        except Exception:
            pass
    _VIZ_PTS = []
    n_pts = 0
    pts = req.get("points")
    if pts is not None:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        for pi, pt in enumerate(pts):
            try:
                h = viz._server.scene.add_icosphere(
                    f"/candidate_{pi}",
                    radius=_VIZ_SPHERE_RADIUS, color=_VIZ_SPHERE_COLOR,
                    position=(float(pt[0]), float(pt[1]), float(pt[2])),
                )
                _VIZ_PTS.append(h)
                n_pts += 1
            except Exception as exc:
                log.warning(f"add_icosphere {pi} failed: {exc}")

    _start_playback(viz, positions, dst_dt, _arm_joint_names(rcfg, positions.shape[1]))

    url = f"http://localhost:{port}"
    log.info(f"viz_traj OK · {positions.shape[0]} waypoints @ {hz:.0f}Hz · "
             f"scene {t_scene-t0:.2f}s | planner {t_planner-t_scene:.2f}s | "
             f"plan {t_plan-t_planner:.2f}s | url={url}")
    return {
        "status":      "ok",
        "url":         url,
        "port":        port,
        "n_waypoints": int(positions.shape[0]),
        "hz":          hz,
        "dt":          dst_dt,
        "n_objects":   len(objs),
        "n_points":    n_pts,
        "elapsed_s":   round(float(t_plan - t0), 4),
    }


# ── ZMQ wire helpers (single pickled frame each way) ───────────────────────────
def _wire(obj):
    """numpy -> builtins, recursively. A pickled ndarray/scalar embeds this
    numpy's module layout (numpy._core on >=2.0) and fails to unpickle in a
    client on the other side of the 1.x/2.x split (e.g. Isaac apps, whose
    bundled numpy 1.x shadows the env's). Same rule as requests: lists on the
    wire; IKClient rebuilds arrays on receipt."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_wire(v) for v in obj]
    return obj


def _send(sock, obj):
    """REP requires exactly one reply per request. Always send one pickled dict."""
    sock.send(pickle.dumps(_wire(obj)))


# ── server ─────────────────────────────────────────────────────────────────────
def main(endpoint=None):
    # The robot config now carries a per-request gripper_sphere_scale, so it's
    # rebuilt inside each handler (build_robot_cfg(_gripper_scale(req))) rather than
    # once here. It's cheap (a YAML load + a few radius multiplies — no CUDA); the
    # expensive solver/planner build is what the caches below guard.

    # ---- NOTE on per-request rebuild ------------------------------------------
    # The IK solver and the MotionPlanner bake the scene in at create() time, and
    # the scene arrives with every request. Rebuilding recaptures CUDA graphs
    # (~seconds; planner.warmup() adds more), so the batch-IK path below caches the
    # last solver keyed on a content hash of the scene + seed/batch sizes: a repeat
    # request with an identical scene (e.g. a cost-function loop sweeping poses)
    # reuses it and skips the rebuild entirely. The cache holds ONE solver — any
    # scene change evicts it (frees VRAM) and builds a fresh one. The traj /
    # penetration / viz paths still rebuild per call. If you only ever MOVE
    # obstacles you could go further and use
    #   ik.scene_collision_checker.update_obstacle_pose(name, Pose)
    # per request instead. (cuRobo v2 exposes this — see the Viser example.)
    # ---------------------------------------------------------------------------

    # Single-entry solver cache for the batch-IK path (see NOTE above).
    solver_cache = {"key": None, "ik": None, "scene": None}

    from semantic_grasp.config import IK, bind
    endpoint = endpoint or IK
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.REP)
    # ipc:// — co-located, no network. pickle.loads on a hostile payload is RCE;
    # a Unix socket is reachable only by local users (was 127.0.0.1 TCP).
    bind(sock, endpoint)
    log.info(f"IK server listening on {endpoint}")

    while True:
        try:
            raw = sock.recv()
        except KeyboardInterrupt:
            log.info("shutting down")
            break
        except Exception as exc:                       # socket-level, rare
            log.warning(f"recv failed: {exc}")
            continue

        # REP requires exactly one reply per request — guarantee it inside try.
        try:
            try:
                req = pickle.loads(raw)
            except Exception as exc:
                _send(sock, {"status": "error", "message": f"could not unpickle: {exc}"})
                continue

            if isinstance(req, dict) and req.get("cmd") == "ping":
                _send(sock, {"status": "ok"})
                continue

            # ── full-trajectory planning path ─────────────────────────────────
            if isinstance(req, dict) and req.get("cmd") == "solve_traj":
                _send(sock, run_traj_request(req))
                continue

            # ── per-config penetration query (robot spheres vs scene) ─────────
            if isinstance(req, dict) and req.get("cmd") == "penetration":
                _send(sock, run_penetration_request(req))
                continue

            # ── Viser scene visualization ─────────────────────────────────────
            if isinstance(req, dict) and req.get("cmd") == "visualize":
                _send(sock, run_viz_request(req))
                continue

            # ── plan a traj and loop-play it in the viewer ────────────────────
            if isinstance(req, dict) and req.get("cmd") == "viz_traj":
                _send(sock, run_viz_traj_request(req))
                continue

            # ── default: batch IK ─────────────────────────────────────────────
            ok, err = validate_request(req)
            if not ok:
                _send(sock, {"status": "error", "message": err})
                continue

            num_seeds     = int(req.get("num_seeds",  DEFAULT_NUM_SEEDS))
            gripper_scale = _gripper_scale(req)
            tbl        = req.get("table") or {}
            table_pose = tuple(tbl.get("pose", _TABLE_POSE))
            table_dims = tuple(tbl.get("dims", _TABLE_DIMS))

            poses = np.asarray(req["poses"], dtype=np.float32)
            # Never build a solver bigger than the request: cuRobo sizes the
            # solver to the problem (its cfg default is max_batch_size=1), and
            # short chunks get padded up to batch_size for the CUDA graph — so
            # an oversized batch_size is pure wasted VRAM (batch x seeds).
            batch_size = max(1, min(int(req.get("batch_size", DEFAULT_BATCH_SIZE)),
                                    poses.shape[0]))
            xyz   = poses[:, :3]
            rpy   = poses[:, 3:6]
            quat  = euler_to_wxyz(rpy)

            cache_key = _objects_digest(req["objects"], table_pose, table_dims,
                                        num_seeds, batch_size, gripper_scale)

            t_req = time.time()

            if solver_cache["key"] == cache_key and solver_cache["ik"] is not None:
                # Identical scene + solver params as the last call — reuse the baked
                # solver and skip the scene build + CUDA-graph recapture entirely.
                ik      = solver_cache["ik"]
                scene   = solver_cache["scene"]
                t_scene = t_solver = time.time()
                reused  = True
            else:
                # Scene (or batch/seed sizes) changed — evict the stale solver first
                # so VRAM doesn't accumulate, then build a fresh one for this scene.
                if solver_cache["ik"] is not None:
                    solver_cache.update(key=None, ik=None, scene=None)
                    torch.cuda.empty_cache()

                scene = build_scene_from_objects(req["objects"], table_pose, table_dims)
                t_scene = time.time()
                # build the IK solver for this scene (recaptures CUDA graphs); the
                # robot config carries this request's gripper sphere inflation.
                rcfg = build_robot_cfg(gripper_scale)
                ik = make_ik_solver(scene, rcfg, num_seeds, batch_size)
                t_solver = time.time()
                solver_cache.update(key=cache_key, ik=ik, scene=scene)
                reused = False

            target_link = ik.tool_frames[0]

            # ── solve ──────────────────────────────────────────────────────────
            joints, success = solve_orientations(ik, target_link, xyz, quat, batch_size)
            t_solve = time.time()

            build_str = ("REUSED solver" if reused else
                         f"scene {t_scene - t_req:.2f}s | solver {t_solver - t_scene:.2f}s")
            log.info(f"solved {poses.shape[0]} poses · feasible "
                     f"{int(success.sum())}/{poses.shape[0]} · "
                     f"{build_str} | ik {t_solve - t_solver:.2f}s")

            _send(sock, {
                "status":     "ok",
                "joints":     np.ascontiguousarray(joints),
                "success":    np.ascontiguousarray(success),
                "n_feasible": int(success.sum()),
                "n_total":    int(success.shape[0]),
                "elapsed_s":  round(float(t_solve - t_solver), 4),
            })

            # Solver/scene are kept in solver_cache for the next call (evicted on a
            # scene change above), so don't free them here.

        except Exception as exc:
            log.exception("solve failed")
            try:
                _send(sock, {"status": "error", "message": str(exc)})
            except Exception:
                pass


# ── quick client for sanity testing ───────────────────────────────────────────
def _client_test(addr=None, n=200, batch_size=64):
    if addr is None:
        from semantic_grasp.config import IK as addr
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(addr)

    poses = np.zeros((n, 6), dtype=np.float32)
    poses[:, :3] = np.random.uniform([0.30, -0.30, 0.10],
                                     [0.60,  0.30, 0.50], (n, 3))
    poses[:, 3:] = np.random.uniform(-np.pi, np.pi, (n, 3))

    # Empty object list -> table-only collision scene; lets IK run without meshes.
    req = {"objects": [], "poses": poses, "batch_size": batch_size}
    sock.send(pickle.dumps(req))

    resp = pickle.loads(sock.recv())
    if resp.get("status") != "ok":
        print("ERROR:", resp.get("message"))
        return
    joints  = np.asarray(resp["joints"], dtype=np.float32)   # replies carry lists
    success = np.asarray(resp["success"], dtype=bool)
    print(f"feasible {resp['n_feasible']}/{resp['n_total']} in {resp['elapsed_s']}s")
    print("first feasible joints:\n", joints[success][:3])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="run a random-pose client")
    ap.add_argument("--port", type=int, default=None,
                    help=f"Viser web-viewer port (default {DEFAULT_VIZ_PORT}); "
                         "requests that carry their own 'port' still win")
    ap.add_argument("--addr", default=None)  # None -> ipc endpoint (config.IK)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sphere-scale", type=float, default=None,
                    help="inflate EVERY robot collision sphere by this factor "
                         "(e.g. 1.15 = 15%% margin everywhere). Use when the real "
                         "arm trips self-collision on plans the server accepted. "
                         "Compounds with the per-request gripper_sphere_scale on "
                         "the gripper links.")
    ap.add_argument("--self-collision-pad", type=float, default=None,
                    help="pad every link's self_collision_buffer by this many "
                         "metres (e.g. 0.02). Makes ONLY the self-collision check "
                         "stricter -- no table-clearance cost, unlike "
                         "--sphere-scale. Use when the real arm trips its "
                         "self-collision reflex on plans the server accepted.")
    args = ap.parse_args()

    if args.sphere_scale is not None:
        if args.sphere_scale <= 0:
            ap.error("--sphere-scale must be positive")
        _ROBOT_SPHERE_SCALE = args.sphere_scale
        print(f"[ik_server] ALL robot collision spheres inflated x{_ROBOT_SPHERE_SCALE}")

    if args.self_collision_pad is not None:
        if args.self_collision_pad < 0:
            ap.error("--self-collision-pad must be >= 0")
        _SELF_COLLISION_PAD = args.self_collision_pad
        print(f"[ik_server] self_collision_buffer padded +{_SELF_COLLISION_PAD} m per link")

    if args.port is not None:
        # Both viz handlers fall back to this global when a request carries no
        # "port" of its own (the IKClient default since port went optional).
        DEFAULT_VIZ_PORT = args.port

    if args.test:
        _client_test(args.addr, args.n, args.batch)
    else:
        main(args.addr)
