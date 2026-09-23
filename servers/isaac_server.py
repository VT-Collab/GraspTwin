"""
Isaac Lab grasp-lift test — Persistent ZMQ Server (pickle I/O, per-request scene).

What changed vs the old version
-------------------------------
  * Wire format is now PICKLE, not JSON. A request is a single pickled dict; the
    reply is a single pickled dict. NumPy arrays (the (N,7) joints) travel as-is,
    with no list round-trip.
  * There is no more transforms.json / orientation_results.json / resolve_target.
    Each request carries the scene itself, INCLUDING the mesh bytes (the mesh is
    not read from disk — Isaac's MeshConverter needs a file, so the bytes are
    written to a temp file, converted, then deleted):
        {
          "target":      str,             # object name (label only)
          "mesh_bytes":  bytes,           # an encoded mesh file
          "mesh_format": str,             # "glb" / "obj" / "ply" ...
          "translation": [x, y, z],       # ROBOT-BASE frame
          "rotation":    [w, x, y, z],    # ROBOT-BASE frame (wxyz)
          "scale":       float,
          "joints":      np.ndarray(N,7), # arm joint sets to evaluate
          "mode":        str,             # "try" (default) or "visualize"
          "viz_steps":   int,             # OPTIONAL, only for "visualize"
          "extra_objects": [ {name, mesh_bytes, mesh_format, translation,
                              rotation, scale}, ... ],
                                          # OPTIONAL (try/visualize/load): the
                                          # clearance-promoted neighbors, spawned
                                          # beside the target in every env. Part
                                          # of the scene identity. Per-env
                                          # scene_poses move ONLY the target.
        }
    translation/rotation arrive already in the robot base frame (the sam3d
    server bakes in the camera->robot C2R), so no frame conversion happens here —
    only the ROBOT_BASE_POS env offset is added before they reach Isaac Lab.

    "mode" controls what happens with the joints:
      * "try"        -> the original behavior: run every joint set through the
                        full snap -> grip-close -> lift pipeline and return the
                        per-grasp metrics. This path is UNCHANGED.
      * "visualize"  -> take ONLY the first joint row, place the robot in that
                        configuration, and just render it so you can eyeball how
                        the arm/gripper sits relative to the object. The gripper
                        stays open and there is NO lift. Rendering is forced on.
      * "settle"     -> step physics on a multi-object scene with the robot held
                        at home, return settled robot-base-frame poses.
      * "rollout"    -> loop the full snap -> close -> lift -> hold -> reset
                        pipeline for ONE joint configuration (the FIRST row of
                        `joints`) FOREVER, until the process is killed or the
                        viewer is closed. Launch the server with --render to
                        watch it live. Mirrors evaluate()'s motion exactly.
                        Unlike the grasp modes, rollout spawns the WHOLE scene:
                        the request carries an "objects" list (same shape as
                        settle) and a "target" naming which of them is grasped /
                        lifted; the rest are spawned as surrounding clutter.

Why it still doesn't hang
-------------------------
Rebuilding the scene graph / re-initializing the PhysX GPU views *live* (after the
SimulationContext has been sitting idle behind a blocking socket.recv) deadlocks.
So the scene is built EXACTLY ONCE per process, at startup, in the tight
synchronous sequence Isaac Lab expects. The serve loop never rebuilds:

  * Request's scene inputs match the live scene  -> just evaluate the new joints.
  * Request's scene inputs differ               -> dump those inputs to a temp
    pickle and os.execv into a fresh process that builds them at startup.
  * Cold start (no scene yet) -> the first request triggers that same re-exec,
    so a live build never happens inside the loop.

Pass --bootstrap <pkl> at launch (a pickled request dict) to build immediately and
skip the first-request restart.

Robustness fixes in THIS revision
---------------------------------
  * restart_with_request no longer closes the socket with LINGER=0 right after
    queueing the "restarting" reply — zmq sends are async, and a zero linger
    could DISCARD the queued reply before the execv, leaving the client blocked
    forever on recv(). The socket now lingers long enough to flush.
  * validate_request now actually LOADS the mesh bytes (trimesh) before the
    request is allowed to trigger a re-exec. Previously a corrupt/empty mesh
    passed the cheap checks, forced a restart, and the NEW process died during
    the world build — a permanent outage with a client stuck waiting on a
    "retry: true" promise that could never be honored.
  * A failed world build no longer kills the process. The server remembers the
    failed scene identity + error, keeps serving, replies with a clear error to
    any request for that same scene, and still re-execs normally for a
    DIFFERENT scene. (Previously: silent exit; clients timed out — or, for the
    timeout-disabled rollout call, hung forever.)
  * settle() is now idempotent: every settle call first resets all objects back
    to their REQUESTED (build-time) poses and the robot to home, so resending
    the identical request (e.g. after a client timeout) returns the same
    answer instead of compounding another settle_steps of physics.
  * evaluate() with per-env `scene_poses` now PRE-SETTLES each batch (robot
    parked at home) before capturing the snap-drift baseline. Previously the
    baseline was the raw teleported pose: a pose floating a few mm falls
    ~3.4 cm under gravity in the 5 snap steps, and one slightly inside the
    table depenetrates at up to 1 m/s (~8 cm) — both tripping the 5 cm
    interpenetration_on_snap check spuriously.
  * Converted USDs are cached by (mesh-bytes hash, format, scale, resting
    up-vector). Identical geometry — which the BO loop resends every round —
    reuses the cooked USD instead of re-decimating/re-cooking, and
    /tmp/sim_grasp_usd growth is now bounded by the number of DISTINCT meshes
    (x the handful of resting orientations they are seen in), not the number
    of requests.
  * Resting-base flattening (BASE_FLATTEN): the collision mesh is planed flat
    below its lowest point along the object's world-up so its base is a
    statically stable flat cap instead of the hallucinated dome that made
    tall objects rock/walk on the table indefinitely (and re-rock after every
    teleport reset). Render mesh untouched; the cut is skipped when no cut
    within the cap would be stable (the object is going to topple anyway).
    Planed objects also SPAWN square on their cap (_upright_env_pose): the
    perception tilt (3-15 deg is typical) is rotated out and the cap set on
    the tabletop, so nothing has to topple into place — the settle reply is
    the upright pose.
  * PhysX device is now explicit (--physics-device, default = --device).
    Isaac Sim 4.5's GPU narrowphase pumps energy into DISTURBED trimesh-vs-box
    resting contacts (a kicked object never stops rocking; no solver knob
    helps — see _physics_device); the CPU path damps them but buzzes hollow
    objects at rest and is 3.4x slower for 145 envs. Planed objects spawn at
    rest and stay at rest on the GPU, so the GPU stays the default.
  * The fixed Y-up -> Z-up conversion rotation is now applied ONLY to glTF
    family formats (glb/gltf); obj/ply/stl pass through unrotated. (The client
    always sends glb, so this only changes behavior for formats that were
    previously being silently rotated 90 degrees.)
  * rollout / rollout_traj worlds are built with a batch size of ONE. They only
    ever demo row 0 broadcast across the batch; building --num-envs (default
    145) fully-cluttered environments and rendering all of them to watch a
    single demo was pure waste (and USD-cloning at that scale is minutes).
  * Successful results now always carry "fail_reason": None, so clients can
    read the key unconditionally.
  * 'try' mode rejects an empty (0,7) joints array instead of returning [].
  * The bootstrap pickle (which holds full mesh bytes) is deleted even when the
    build fails, instead of leaking in /tmp.
"""

# ── App must launch before any sim imports ───────────────────────────────────
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab grasp-lift ZMQ server (pickle I/O)")
parser.add_argument("--bootstrap", type=str, default=None,
                    help="Path to a pickled request dict to build the scene from at startup.")
# 145 = one full candidate-scene block per chunk in main_batched's BO round
# (9*16+1 pose candidates). Measured on the RTX 3080 Ti 16GB: 145 envs run a
# 493-step rollout chunk in ~6s vs ~46s for 10 chunks of 15 (7.4x), with <1GB
# extra VRAM and ~15s warm world rebuilds. 725 (one whole round) was tested and
# rejected: USD cloning made every scene-switch rebuild take >10 minutes.
parser.add_argument("--num-envs", type=int, default=145, help="Batch size for parallel evaluation.")
parser.add_argument("--lift-height", type=float, default=0.5)
parser.add_argument("--render", action="store_true",
                    help="Render the viewport every step (slow). Off by default for a server.")
parser.add_argument("--viz-steps", type=int, default=600,
                    help="Render steps to hold a 'visualize' pose (~10s @ 60Hz). "
                         "A per-request 'viz_steps' value overrides this.")
parser.add_argument("--physics-device", type=str, default=None,
                    help="PhysX device ('cpu' or 'cuda:N'); default: same as --device. "
                         "Isaac Sim 4.5's GPU narrowphase pumps energy into resting "
                         "trimesh-vs-box contacts once they are disturbed (objects rock/walk "
                         "for ever, no solver knob helps); the CPU path damps them, but buzzes "
                         "hollow objects (the mug) at rest instead and is ~3.4x slower for the "
                         "145-env grasp world (measured 47 vs 14 ms/tick; faster for one env). "
                         "With BASE_FLATTEN objects spawn at rest and stay at rest on either, "
                         "so the GPU stays the default; 'cpu' is the knob if a rollout leaves "
                         "bumped neighbours rocking.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Safe to import sim modules now ───────────────────────────────────────────
import hashlib
import io
import json
import os
import pickle
import tempfile
import types

import zmq
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# ==============================================================================
#  HARDCODED CONFIGS
# ==============================================================================
ROBOT_BASE_POS    = (0.0, 0.0, 1.05)
TABLE_TRANSLATION = (0.55, 0.0, 1.05)
# The SeattleLabTable collider (Collisions/Cube in table.usd) is a box whose
# top face sits 3 mm BELOW the table prim's origin, so with the table spawned
# at TABLE_TRANSLATION the physical tabletop is at env z = 1.047 (robot-base
# z = -0.003). Used to spawn planed objects resting exactly on it.
TABLE_TOP_Z_ENV      = TABLE_TRANSLATION[2] - 0.003
SPAWN_TABLE_CLEARANCE = 0.0005   # m; planed cap starts this far above the top
SPAWN_TABLE_SNAP_MAX  = 0.03     # m; only snap caps already within this of the top
ENV_SPACING       = 4.0
SIM_DT            = 1.0 / 60.0   # CONTROL tick — targets/IK/captures update at 60 Hz
# Physics substeps per control tick. 60 Hz single-stepping was a root cause of
# the contact glitchiness (objects buzzing in the closed jaws, popping out at
# the depenetration cap): at the fingers' 0.2 m/s approach one 60 Hz step
# buries a pad ~3 SDF voxels deep before the solver ever sees the contact, and
# the undamped pinch re-excited itself every step. 4 substeps -> 240 Hz physics
# keeps per-step penetration under one SDF voxel. Every *_STEPS constant below
# is STILL a 60 Hz tick count (wall-clock semantics unchanged); phys_tick()
# hides the substepping, so control/IK/render cadence stays 60 Hz.
PHYS_STEPS_PER_TICK = 4
PHYS_DT             = SIM_DT / PHYS_STEPS_PER_TICK
USD_DIR           = "/tmp/sim_grasp_usd"

# panda_hand -> gripper_tcp offset along panda_hand's local +z. Matches the
# virtual link ik_server.py adds to cuRobo's kinematics (the frame IK targets
# are solved at — i.e. 8.6 mm behind the jaw fingertip). Keep in lock-step with
# ik_server._GRIPPER_TCP_OFFSET.
# NOTE: this constant and hand_to_tcp() below are REFERENCE/documentation for
# that lock-step contract — nothing in this file calls them. If you delete
# them, the only remaining statement of the TCP convention lives in ik_server.
# 2026-09-05: 0.19267 -> 0.20407 to match ik_server (TCP now 8.6 mm behind the jaw
# tips, the stock-Franka tip-to-TCP distance, instead of 20 mm).
GRIPPER_TCP_OFFSET = 0.20407   # 0.07 standoff + 0.14267 jaw tip − 0.0086 (8.6 mm behind fingertip)

# Rubber parallel-jaw gripper on the Franka arm. Same arm actuators/init as the
# Panda HIGH_PD cfg (link/joint names preserved), only the spawn USD is swapped
# to the converted franka_rubber.usd (see convert_gripper_usd.py).
# Repo root (this file lives at the top of the repo) — derive asset paths from it
# so the "<repo-root>" prefix isn't hardcoded.
# Repo root is the PARENT of servers/ (these scripts were moved into servers/;
# `.parent` alone points at servers/). Needed both for the config paths below
# and so `from semantic_grasp...` resolves when run as `python servers/isaac_server.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
RUBBER_GRIPPER_USD = str(_REPO_ROOT / "config/grippers/franka_rubber/franka_rubber.usd")
# Self-collisions OFF (stock FRANKA_PANDA_CFG has them on): the jaws' VHACD
# colliders sit ~3 mm proud of the real pad faces, so with self-collisions the
# fingers "touch" ~6 mm before the pads meet and an empty close reads 6 mm of
# width. Finger-finger contact adds nothing (the 0 joint limit already stops
# each pad at the closed plane) and arm self-collision is curobo's job at plan
# time, so drop it; object/table contact is unaffected.
RUBBER_GRIPPER_CFG = FRANKA_PANDA_HIGH_PD_CFG.replace(
    spawn=FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
        usd_path=RUBBER_GRIPPER_USD,
        articulation_props=FRANKA_PANDA_HIGH_PD_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=False,
        ),
    ),
)

# Reverted to the original 0.1 kg. It was briefly 0.5 (commit 7812293) on the
# reasoning that 0.1 made the sim too forgiving: with mu=1.0 and 20 N of grip
# effort per pad the jaws hold 2*mu*20 N = 40 N ~ 4 kg, so a 0.1 kg object comes
# up on 1/40th of a real squeeze and an edge/corner pinch can score as a clean
# lift. That tradeoff is real — if edge pinches start passing again, this is the
# knob — but 0.5 changes which grasps survive, so the two are not interchangeable
# and results from one mass are not comparable with the other.
# NOTE: applied via MassPropertiesCfg at spawn, not baked into the cached USD —
# no need to clear /tmp/sim_grasp_usd, but isaac_server must be RESTARTED.
OBJECT_MASS_KG   = 0.1
# Object physics stability (applied via _object_rigid_props). PhysX defaults
# leave per-body solver iterations low and depenetration uncapped, so concave
# SDF colliders jitter at rest and get ejected at high speed when the gripper or
# table grazes them ("freak out on touch"). These tame both.
OBJECT_SOLVER_POS_ITERS = 16    # up from the cooked default (~4): resolves
                                # resting contact on the concave SDF surface
OBJECT_SOLVER_VEL_ITERS = 4     # was 1 — friction needs velocity iterations to
                                # converge under the 40 N squeeze; at 1 the
                                # tangential solution ratcheted step to step
OBJECT_MAX_DEPEN_VEL    = 0.2   # m/s cap on solver depenetration. Was 1.0 —
                                # enough to stop launches, but every deep
                                # contact still exited at a visible 1 m/s pop
                                # (the "jumps out of the jaws" glitch). 0.2
                                # turns those events into sub-frame nudges.
# Resting-DRIFT tamers (separate from the ejection cap above). A thin SDF
# collider — a spoon handle is only ~2-3 voxels thick — makes noisy table
# contacts that keep a light body micro-jittering and slowly walking across the
# table. Damping bleeds that residual velocity; PhysX stabilization (enabled on
# the scene) quiets low-speed resting contacts. A gripper contact still wakes
# and moves the object normally, so grasp dynamics are unaffected.
OBJECT_LINEAR_DAMPING       = 0.2
OBJECT_ANGULAR_DAMPING      = 0.5     # thin objects rock about their long axis
OBJECT_STABILIZATION_THRESH = 0.0025  # mass-normalized speed^2 below which PhysX
                                      # stabilizes the body (needs enable_stabilization)
OBJECT_SLEEP_THRESH         = 5e-4    # mass-normalized KE below which the body
                                      # SLEEPS (~3 cm/s equivalent). The PhysX
                                      # default (5e-5) never triggered against
                                      # residual SDF contact noise, so settled
                                      # objects micro-wobbled forever instead of
                                      # going still. Deliberately mild: well
                                      # under any slip speed that matters during
                                      # a lift, and gripper contact wakes the
                                      # body normally.
# Torsional friction patch (spawn-time, on the collider prim). Captured bases
# are slightly convex — SAM-3D hallucinates the unseen underside — so at rest
# an object touches the table in a near-point contact. Coulomb friction carries
# no torque through a point, so the body rocks/spins indefinitely; the patch
# radius approximates the rotational friction of a real compressed contact
# area (PhysX best practice for bodies resting on small patches).
OBJECT_TORSIONAL_PATCH_RADIUS     = 0.02
OBJECT_MIN_TORSIONAL_PATCH_RADIUS = 0.005
# SDF colliders track the real (concave) mesh surface. convexDecomposition with
# default cooking bulged past the visual mesh and filled cavities (mug openings,
# handle holes), so "grasps" that only closed on empty space near the object
# still contacted the phantom hull volume and lifted it.
COLLISION_APPROX = "sdf"
SDF_RESOLUTION   = 128  # SDF samples along the mesh's largest AABB extent (~1mm
                        # voxels on a 12cm object — plenty for a 8cm-wide gripper)
# Perception meshes arrive at 500k+ triangles; PhysX SDF cooking AND per-step
# contact generation both scale with triangle density, which is what blew the
# per-evaluate time up. The budget applies to the COLLISION mesh only — an
# invisible purpose=guide prim (_build_collision_mesh / _author_collision_prim);
# the raw capture stays untouched as the render mesh, so this cap no longer
# low-fies what the cameras see. At this face budget the collider's surface
# error is far below one SDF voxel.
MAX_COLLISION_FACES = 20_000
# Thin-shell colliders (same constants and rationale as baseline_server.py):
# PhysX SDF contact breaks down on walls thinner than ~1 SDF cell (spoon
# bowls, ladles) — the resting contact kicks the object sideways every
# substep, a constant buzz that friction ratchets into a steady walk. Such
# colliders are rebuilt through the voxel remesh with the occupancy dilated
# THIN_SHELL_DILATE cells so every wall is >= 3 cells thick.
THIN_SHELL_FRAC     = 0.15
THIN_SHELL_SAMPLES  = 4000
THIN_SHELL_DILATE   = 1
# Floor for the adaptive watertight-remesh resolution (cells along the largest
# AABB extent). 48 keeps mug-handle / toaster-slot openings several voxels wide
# on every capture seen so far while staying near ~2x the face budget.
REMESH_MIN_RES = 48
# Resting-base flattening (_flatten_base). The captures' undersides are
# hallucinated — SAM-3D never sees the base, so it closes it with a dome/ridge —
# and a domed base makes the object a rocking toy: it touches the table at a
# point, its COM projects OUTSIDE that point contact, and PhysX has no rolling
# resistance, so it tips onto the next vertex, gets a depenetration kick there,
# and rocks/walks forever (measured: the tall bottles/containers in scene_7
# never sleep — 0.6-2 rad/s peaks after 12 s of free settling — and even a
# 0.7x/tick velocity bleed cannot quench them). No solver setting fixes this;
# it is geometry. So the COLLISION mesh (only — the render mesh is untouched)
# is planed flat below its lowest point along the object's world-up direction:
# the cut height is the smallest h >= H_MIN at which the cross-section's convex
# hull contains the COM projection with >= MARGIN (i.e. the flat base is
# statically stable), capped by H_MAX and H_FRAC * object height. Removing a
# few mm of hallucinated dome that would sit inside the table anyway changes
# no grasp: the base is on the table. Every real object here HAS a flat base;
# the flat cap is more faithful than the dome it replaces.
# The world-up direction is pose-dependent, so it is part of the USD cache key
# (quantized to BASE_UP_QUANT so the settled pose — a few mm/deg from the
# perception pose — hits the same cooked USD instead of recooking the SDF).
BASE_FLATTEN         = True
BASE_FLATTEN_H_MIN   = 0.002    # m; always plane off at least this much
BASE_FLATTEN_H_MAX   = 0.010    # m; never plane off more than this ...
BASE_FLATTEN_H_FRAC  = 0.10     # ... nor more than this fraction of the height
BASE_FLATTEN_MARGIN  = 0.008    # m; COM projection must sit this far inside the base hull
BASE_FLATTEN_H_STEP  = 0.0005   # m; cut-height search step
BASE_FLATTEN_MIN_STAB_DEG = 5.0  # deg; tilt the flat cap must tolerate before tipping
BASE_FLATTEN_MAX_TILT_DEG = 20.0 # deg; max re-uprighting away from the perception pose
BASE_UP_QUANT        = 0.05     # up-vector quantization for the cache key (~3 deg)

# Mesh formats whose source convention is Y-up (the glTF family). ONLY these
# get the fixed Y-up -> Z-up conversion rotation in convert_mesh(); obj/ply/stl
# etc. are passed through unrotated. (The old code rotated EVERYTHING, which
# silently turned any already-Z-up obj/ply capture 90 degrees. The pipeline
# client only ever sends glb, so for it nothing changes.)
YUP_MESH_FORMATS = {"glb", "gltf"}

WARMUP_STEPS     = 240  # initial object-settling (~4 s), run ONCE at startup
SNAP_STEPS       = 5    # after teleporting robot, let physics react
GRIP_STEPS       = 80   # closing the gripper
LIFT_STEPS       = 408  # IK-driven lift (~6.8 s @ 60 Hz) — slowed to reduce slip during the raise
LIFT_RAMP_FRAC   = 0.8   # fraction of LIFT_STEPS spent ramping z up; remainder holds at top to settle
SNAP_DRIFT_TOL_M = 0.05
SETTLE_STEPS     = 240  # default physics steps for the 'settle' mode (~4 s @ 60 Hz)
# Pre-settle steps for per-env `scene_poses` batches: the requested poses are
# teleported in raw, so they must be allowed to come to rest BEFORE the
# snap-drift baseline is captured (see evaluate()). ~0.75 s minimum for the
# few-mm float / sub-cm depenetration that survives an upstream settle pass;
# runs on to PRESETTLE_MAX_STEPS if the objects are still moving.
# Contract note: scene_poses SHOULD come from a prior 'settle' request; wildly
# unsettled poses (object mid-air) will still drift during the snap and can
# legitimately fail the interpenetration check.
PRESETTLE_STEPS  = 45
# Settle-phase velocity bleed. The captures' undersides are domed/ridged (the
# reconstruction never sees the base), so a settling object is a rocking toy:
# tipping is ROLLING contact — Coulomb friction dissipates nothing there, the
# torsional patch only damps spin about the vertical, and PhysX has no
# rolling-resistance model — so the rock outlives the settle window on angular
# damping alone. During settle-type stepping ONLY (warmup, 'settle' mode,
# per-batch pre-settle, rollout_traj's post-reset quench) every object's root
# velocity is scaled by this factor each 60 Hz control tick, and hard-zeroed
# once at the settle->eval handoff. Grasp/lift dynamics never see it, so
# evaluation results stay comparable. 0.7: heavy enough that the rock is
# overdamped (it creeps to the equilibrium instead of oscillating through it),
# so the pose the hard-zero freezes is one gravity will actually hold.
SETTLE_BLEED_FACTOR = 0.7
# Settle convergence test (settle_until_still). Every *_STEPS window below is a
# MINIMUM; stepping continues past it until every object in every env has been
# still for SETTLE_QUIET_TICKS consecutive ticks — pre-bleed root speed under
# the *_VEL_TOL (i.e. the physics itself has stopped generating motion, not
# merely that the bleed ate it) AND pose drift since the quiet window opened
# under the *_POSE_TOL — or until the phase's *_MAX_STEPS cap. Bleed+zero for a
# fixed count froze the pose wherever the rock happened to be, so a released
# object that was not at equilibrium simply started rocking again.
SETTLE_LIN_VEL_TOL  = 2e-3   # m/s
SETTLE_ANG_VEL_TOL  = 5e-2   # rad/s (~3 deg/s). Planed objects at rest on the GPU
                             # still report a 0.01-0.06 rad/s contact buzz with ZERO
                             # pose drift; 2e-2 kept the quench running to its cap
                             # every reset. The drift test below is the real gate.
SETTLE_POS_TOL      = 5e-4   # m drift over the quiet window
SETTLE_ROT_TOL      = 5e-3   # rad (~0.3 deg) over the quiet window
SETTLE_QUIET_TICKS  = 12     # ~0.2 s of stillness required
WARMUP_MAX_STEPS    = 900    # ~15 s cap on the one-time warmup / 'settle' mode
PRESETTLE_MAX_STEPS = 120    # ~2 s cap on the per-batch pre-settle
# Post-reset quench in rollout_traj: teleporting the objects back to their
# settled poses re-seats the SDF contacts with a small kick that restarts the
# rock, so bleed+zero (arm held at the trajectory start) until still — at
# least ~0.5 s, at most ~2 s — before the trajectory plays. Kept in lock-step
# with baseline_server.py.
RESET_SETTLE_STEPS     = 30
RESET_SETTLE_MAX_STEPS = 120

# rollout / rollout_traj only ever demo ONE configuration (row 0 broadcast), so
# their worlds are built single-env regardless of --num-envs. Building the full
# evaluation batch of cluttered scenes to watch one demo wasted minutes of USD
# cloning and rendered num_envs copies of the same motion.
ROLLOUT_BATCH_SIZE = 1


def _physics_device(world_kind):
    """PhysX device for a world (see --physics-device). Measured on this
    codebase (Isaac Sim 4.5 / PhysX 5.5.1, scene_7 settle world, no bleed):
    on the GPU pipeline an SDF-trimesh object on the table's box collider that
    gets ANY kick (spawned tilted, toppled, bumped) rocks at 1-2 rad/s for
    ever — no solver setting damps it (pos iters, PGS, contact/rest offsets,
    depenetration cap, mass, stabilization all tried); it is the box/convex-
    vs-triangle-mesh contact path (PhysX 5.10 changelog: wrong separation
    distances on GPU; convexDecomposition colliders on GPU settle instantly).
    On the CPU pipeline the same scene sleeps within 2 s — but the hollow mug
    buzzes at ~0.5-1 rad/s at rest there and does not on the GPU, and 145
    envs run 3.4x slower. Since planed objects spawn at rest and stay at rest
    on the GPU (measured 0.00-0.06 rad/s over 16 s incl. teleport resets), the
    GPU remains the default; the flag exists so a run can trade throughput
    for disturbance-robust resting."""
    dev = args_cli.physics_device or str(args_cli.device)
    dev = str(dev).strip().lower()
    print(f"[sim] {world_kind} world: PhysX on {dev} (--physics-device)", flush=True)
    return dev

RENDER = bool(args_cli.render)  # default False -> much faster for a headless server

# ==============================================================================
#  MATH HELPERS
# ==============================================================================
def _rot_matrix_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

def robot_pose_to_env(translation, rotation_wxyz):
    """ROBOT-BASE-frame (translation, rotation_wxyz) -> Isaac-Lab env (pos, quat).

    Object poses arrive already in the robot base frame (the sam3d server bakes
    in the camera->robot C2R extrinsics), so placing them in Isaac only needs
    the ROBOT_BASE_POS env offset. NOT the frame IK targets live in — those are
    robot-base already and need no offset."""
    env_pos = (
        ROBOT_BASE_POS[0] + translation[0],
        ROBOT_BASE_POS[1] + translation[1],
        ROBOT_BASE_POS[2] + translation[2],
    )
    return env_pos, tuple(float(v) for v in rotation_wxyz)

def env_pose_to_robot(env_pos, env_quat_wxyz):
    """Inverse of robot_pose_to_env: strip ROBOT_BASE_POS, keep the rotation."""
    pos = (
        env_pos[0] - ROBOT_BASE_POS[0],
        env_pos[1] - ROBOT_BASE_POS[1],
        env_pos[2] - ROBOT_BASE_POS[2],
    )
    return pos, tuple(float(v) for v in env_quat_wxyz)

def hand_to_tcp(pos_hand, quat_hand_wxyz):
    """Shift a panda_hand pose along its own local +z by GRIPPER_TCP_OFFSET.

    Returns (tcp_pos, tcp_quat_wxyz). The TCP orientation equals the
    panda_hand orientation (no rotation between them), so the quaternion is
    returned unchanged. (Reference implementation of the ik_server TCP
    convention — see the GRIPPER_TCP_OFFSET note; not called in this file.)"""
    q = np.asarray(quat_hand_wxyz, dtype=float)
    R = _rot_matrix_from_quat(q)
    pos_tcp = np.asarray(pos_hand, dtype=float) + R @ np.array([0.0, 0.0, GRIPPER_TCP_OFFSET])
    return tuple(pos_tcp.tolist()), tuple(q.tolist())

def _quat_wxyz_to_rpy_rad(q_wxyz_np):
    """Convert a wxyz numpy quaternion to roll/pitch/yaw in RADIANS (xyz order)."""
    return Rot.from_quat(
        [q_wxyz_np[1], q_wxyz_np[2], q_wxyz_np[3], q_wxyz_np[0]]
    ).as_euler("xyz", degrees=False)
# ==============================================================================
#  MESH -> USD CONVERSION
# ==============================================================================
def _set_collision_approximation(usd_path, approximation):
    from pxr import Usd, UsdPhysics, PhysxSchema
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return 0
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set(approximation)
            if approximation == "sdf":
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                sdf_api.CreateSdfResolutionAttr().Set(SDF_RESOLUTION)
            n += 1
    if n:
        stage.GetRootLayer().Save()
    return n

# μ=2.0 was masking bad grasps: with a 0.1 kg object (~1 N weight) and a 20 N
# grip effort, a single finger grazing the side at 0.25 N normal force was
# enough to hoist it. 1.0 is still grippy (rubber-pad-on-plastic territory) but
# requires an actual squeeze, not a touch.
OBJECT_STATIC_FRICTION  = 1.0
OBJECT_DYNAMIC_FRICTION = 1.0

def _bake_friction_into_usd(usd_path,
                            static_friction=OBJECT_STATIC_FRICTION,
                            dynamic_friction=OBJECT_DYNAMIC_FRICTION):
    """Write a physics material into the converted USD and bind it to all collision prims.

    The material must live INSIDE the layer's default prim: scenes consume this
    USD by REFERENCING that prim, and a reference cannot map relationship
    targets that point outside its scope. The old /World/ObjectFrictionMaterial
    path did exactly that — USD warned "refers to a path outside the scope of
    the reference ... Ignoring" on every scene build and silently DROPPED the
    binding, so referenced objects ran on PhysX default friction instead of the
    OBJECT_*_FRICTION tuning. (Delete /tmp/sim_grasp_usd when changing this:
    cached USDs keep whichever layout they were baked with.)"""
    from pxr import Usd, UsdPhysics, UsdShade
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return
    root = stage.GetDefaultPrim()
    if not root:                       # no default prim -> take the first root prim
        root = next(iter(stage.GetPseudoRoot().GetChildren()), None)
        if root is None:
            return
    mat = UsdShade.Material.Define(
        stage, root.GetPath().AppendChild("ObjectFrictionMaterial"))
    phys_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_api.CreateStaticFrictionAttr().Set(static_friction)
    phys_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_api.CreateRestitutionAttr().Set(0.0)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, materialPurpose="physics"
            )
    stage.GetRootLayer().Save()

def _materialize_mesh(mesh_bytes, mesh_format):
    """Write in-request mesh bytes to a temp file so Isaac's MeshConverter (which
    only reads files) can ingest it. Caller is responsible for deleting it."""
    suffix = "." + str(mesh_format).lstrip(".").lower()
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="grasp_mesh_")
    with os.fdopen(fd, "wb") as f:
        f.write(bytes(mesh_bytes))
    return path

def _voxel_remesh(mesh, resolution, dilate=0):
    """Rebuild `mesh` as a guaranteed-watertight surface: voxelize the surface
    at `resolution` cells along the largest AABB extent, decide solid/hollow for
    every enclosed air pocket by ray-parity against the source mesh, then run
    marching cubes over the occupancy. Through-holes (mug handles, toaster
    slots) survive because they connect to outside air; sealed interior shells
    (unseen volume the reconstruction closes off, wound inward) stay hollow
    because the parity test sees them as outside the solid — a plain flood-fill
    would weld them shut and hand PhysX a phantom-solid region. `dilate`
    grows the occupancy by that many cells on every side first (grid padded
    so nothing clips at the box) — the thin-shell fix, see THIN_SHELL_*."""
    import trimesh
    from scipy import ndimage

    pitch = float(mesh.extents.max()) / int(resolution)
    vox = mesh.voxelized(pitch)
    occ = np.asarray(vox.matrix, dtype=bool).copy()

    labels, n_labels = ndimage.label(~occ)
    border = set(int(b) for b in np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(),
        labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel()])))
    for lab in range(1, n_labels + 1):
        if lab in border:          # reaches the grid edge -> outside air
            continue
        region = labels == lab
        # Probe the voxel deepest inside the pocket, clear of surface noise.
        dist  = ndimage.distance_transform_cdt(region)
        idx   = np.unravel_index(int(np.argmax(dist)), dist.shape)
        point = vox.indices_to_points(np.array([idx], dtype=float))[0]
        try:
            solid = bool(mesh.contains([point])[0])
        except BaseException:
            solid = True           # unprobeable -> fill it (the pre-remesh behavior)
        if solid:
            occ |= region

    transform = np.asarray(vox.transform, dtype=float).copy()
    if dilate:
        occ = ndimage.binary_dilation(np.pad(occ, dilate), iterations=dilate)
        transform[:3, 3] -= transform[:3, :3] @ np.full(3, float(dilate))
    out = trimesh.voxel.VoxelGrid(occ).marching_cubes
    out.apply_transform(transform)  # marching_cubes returns voxel-index coords
    if out.volume < 0:
        out.invert()
    return out

def _drop_shard_bodies(mesh):
    """Drop tiny disconnected shards from a captured mesh; keep real parts.

    SAM-3D captures carry floating crumbs (a kettle capture had 40 bodies of
    <100 faces). They break the watertight check — forcing the destructive
    voxel-remesh fallback on an otherwise-clean surface — and a crumb under
    the base can prop the object up so it rocks at rest. Genuine separate
    parts survive the 2%-of-largest cut by orders of magnitude (a pan's lid
    is ~58% of the pan's face count). Returns (mesh, n_dropped)."""
    import trimesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    biggest = max(len(p.faces) for p in parts)
    keep = [p for p in parts if len(p.faces) >= max(64, 0.02 * biggest)]
    if len(keep) == len(parts):
        return mesh, 0
    return trimesh.util.concatenate(keep), len(parts) - len(keep)


def _o3d_decimate(mesh, max_faces):
    """Quadric decimation via open3d, which — unlike trimesh's
    fast_simplification backend — reliably preserves manifoldness and
    watertightness (verified on the benchmark captures: every res-128 remesh
    decimated clean, where fast_simplification broke every one). Used as the
    second-chance decimator and for shrinking an over-budget remesh; trimesh's
    is still tried first on raw captures since its result tracks the true
    surface at least as well and both are only accepted when clean."""
    import open3d as o3d
    import trimesh
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    d = m.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    return trimesh.Trimesh(np.asarray(d.vertices), np.asarray(d.triangles),
                           process=False)


def _thin_fraction(mesh, cell):
    """Fraction of the surface whose wall is thinner than `cell`: cast a ray
    inward from THIN_SHELL_SAMPLES surface points and measure the distance to
    the first exit. Samples whose ray misses (broken surfaces) are ignored."""
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, THIN_SHELL_SAMPLES)
    n = mesh.face_normals[fid]
    try:
        loc, idx, _ = mesh.ray.intersects_location(pts - n * 1e-5, -n, multiple_hits=False)
    except Exception:
        return 0.0
    if len(idx) == 0:
        return 0.0
    thick = np.linalg.norm(loc - pts[idx], axis=1)
    return float(np.mean(thick < cell))


def _build_collision_mesh(mesh_path, max_faces=MAX_COLLISION_FACES, scale=1.0):
    """Build the collision-only mesh for the capture at `mesh_path` WITHOUT
    touching the file: the raw capture stays the render mesh, and the mesh
    returned here becomes an invisible guide collider (_author_collision_prim).
    Returns (collision Trimesh in the SOURCE mesh frame, faces_in, how).

    PhysX derives the SDF's sign from the mesh surface, so the collider must
    be closed with consistent winding; an SDF cooked from a broken surface can
    mark free space near holes/cavities as deep-inside — a gripper finger
    snapped into a mug handle or toaster slot then reads as interpenetrating,
    and the depenetration impulse launches the object. Order of attack:

    0. a thin-shelled surface (THIN_SHELL_FRAC) skips straight to the
       dilated voxel remesh of step 3 — PhysX SDF contact on sub-cell walls
       buzzes at rest (see THIN_SHELL_*);
    1. drop floating shards (_drop_shard_bodies) — often all it takes;
    2. decimate to the face budget (trimesh, then the topology-safer o3d as a
       second chance); clean result -> done, the true surface survives;
    3. else voxel-remesh watertight at the SDF's own resolution, then
       taubin-smooth: the remesh marches cubes over BINARY occupancy, and its
       half-voxel stair-steps on the base made remeshed objects rock at rest;
    4. an over-budget remesh is shrunk with _o3d_decimate (clean in practice),
       with a coarser remesh as the last resort (faces scale ~resolution^2,
       hence the sqrt)."""
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    n_in = len(mesh.faces)
    mesh, n_dropped = _drop_shard_bodies(mesh)
    how = [f"dropped {n_dropped} shards"] if n_dropped else []
    cell = float(mesh.extents.max()) / SDF_RESOLUTION
    thin = _thin_fraction(mesh, cell)
    dilate = THIN_SHELL_DILATE if thin > THIN_SHELL_FRAC else 0

    out = mesh
    if not dilate and len(mesh.faces) > max_faces:
        out = mesh.simplify_quadric_decimation(face_count=max_faces)
        if not (out.is_watertight and out.is_winding_consistent):
            out = _o3d_decimate(mesh, max_faces)
        how.append("decimated")
    if not dilate and out.is_watertight and out.is_winding_consistent:
        return out, n_in, " + ".join(how) or "unchanged"

    res = SDF_RESOLUTION
    out = _voxel_remesh(mesh, res, dilate=dilate)
    trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=10)
    if dilate:
        print(f"  [mesh] thin shell ({thin * 100:.0f}% of the surface < 1 SDF cell = "
              f"{cell * scale * 1000:.1f} mm): collider voxel-remeshed @ res {res} and "
              f"dilated {dilate} cell — the RENDER mesh stays the raw capture", flush=True)
    how.append(f"voxel-remeshed @ res {res}"
               + (f" + dilated {dilate} cell" if dilate else "") + " + smoothed")
    if len(out.faces) > max_faces:
        dec = _o3d_decimate(out, max_faces)
        if dec.is_watertight and dec.is_winding_consistent:
            out = dec
            how.append("re-decimated")
        else:
            res = max(REMESH_MIN_RES, int(res * np.sqrt(max_faces / len(out.faces))))
            out = _voxel_remesh(mesh, res, dilate=dilate)
            trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=10)
            how.append(f"fallback remesh @ res {res}")
    return out, n_in, " + ".join(how)


def _conversion_rotation(mesh_format):
    """The fixed source->Isaac axis rotation convert_mesh() bakes into the USD
    (wxyz): Y-up -> Z-up for the glTF family, identity for everything else."""
    if str(mesh_format).lstrip(".").lower() in YUP_MESH_FORMATS:
        return (float(np.sqrt(2) / 2), float(np.sqrt(2) / 2), 0.0, 0.0)
    return (1.0, 0.0, 0.0, 0.0)


def _source_up_vector(rotation_wxyz, mesh_format):
    """World +Z expressed in the RAW SOURCE mesh frame (the frame the guide
    collider's vertices are authored in) for an object spawned with root
    rotation `rotation_wxyz`, quantized to BASE_UP_QUANT. Returns None when
    flattening is off or no rotation is given (-> no cut, legacy cache key)."""
    if not BASE_FLATTEN or rotation_wxyz is None:
        return None
    q = np.asarray(rotation_wxyz, dtype=float)
    r_obj = Rot.from_quat([q[1], q[2], q[3], q[0]])
    c = _conversion_rotation(mesh_format)
    r_conv = Rot.from_quat([c[1], c[2], c[3], c[0]])
    up = (r_obj * r_conv).inv().apply([0.0, 0.0, 1.0])
    return _quantize_up(up)


def _up_cache_tag(up_src):
    return "" if up_src is None else "_planed" + "".join(f"{v:+.2f}" for v in up_src)


def _quantize_up(up):
    """Snap a unit vector to the BASE_UP_QUANT grid (and renormalize) so equal
    resting directions produce equal cache tags. Renormalizing a grid vector
    then re-snapping it is stable (components move < half a bin)."""
    up = np.asarray(up, dtype=float)
    up = np.round(up / BASE_UP_QUANT) * BASE_UP_QUANT
    up[np.abs(up) < 1e-9] = 0.0          # no "-0.00" in the cache tag
    n = np.linalg.norm(up)
    if n < 1e-6:
        return None
    return tuple(float(v) for v in up / n)


def _flatten_base(coll_mesh, up_src, scale):
    """Plane the collision mesh flat below its lowest point so it rests on a
    statically stable flat cap (see the BASE_FLATTEN_* constants).

    `up_src` is the world-up direction in the collider's source frame at the
    requested pose (unit vector); `scale` converts source units to metres.
    Two resting directions are tried and the more stable one wins:
      1. `up_src` itself, and
      2. the "re-uprighted" direction: the outward normal of the convex-hull
         facet directly below the COM (the face the object would naturally
         settle onto), if it is within BASE_FLATTEN_MAX_TILT_DEG of `up_src`.
         Perception tilts of 5-10 deg are common; planing along the tilted
         world-up cuts a WEDGE that props the object up at that tilt with its
         COM near the base's edge (a leaning tower that rocks at the first
         nudge), whereas planing along the natural axis makes it stand square
         with the COM centred. Because the cap normal is what the object comes
         to rest on, the settled pose then reports exactly that up direction.
    A cut is accepted only if the cap contains the COM projection with a
    margin >= max(BASE_FLATTEN_MARGIN, com_height * tan(MIN_STAB_DEG)) — a
    tilt-tolerance the object can take before it tips over the cap edge onto
    the (still round) side and starts rocking again.
    Returns (mesh, info); on any failure / no stable cut the input mesh is
    returned unchanged with info["cut_mm"] = 0. info["up_used"] is the
    (quantized) direction the cap is perpendicular to."""
    import trimesh
    from scipy.spatial import ConvexHull

    up0 = np.asarray(up_src, dtype=float)
    up0 = up0 / np.linalg.norm(up0)
    V = np.asarray(coll_mesh.vertices, dtype=float) * float(scale)
    scaled = trimesh.Trimesh(V, coll_mesh.faces, process=False)
    com = np.asarray(scaled.center_mass, dtype=float)

    # -- candidate resting directions ---------------------------------------
    cands = [("as-posed", tuple(up0))]
    try:
        hull = scaled.convex_hull
        hits, _, fids = hull.ray.intersects_location(
            ray_origins=[com], ray_directions=[-up0], multiple_hits=False)
        if len(fids):
            up_c = -np.asarray(hull.face_normals[int(fids[0])], dtype=float)
            up_c = up_c / np.linalg.norm(up_c)
            tilt = float(np.degrees(np.arccos(np.clip(up_c @ up0, -1.0, 1.0))))
            if 0.5 < tilt <= BASE_FLATTEN_MAX_TILT_DEG:
                q = _quantize_up(up_c)
                if q is not None:
                    cands.append((f"re-uprighted {tilt:.1f} deg", q))
    except Exception:
        pass

    def search(up):
        """Smallest cut height with a stable cap along `up`. Returns
        (h, margin, area, zmin, need) — margin -inf if nothing qualifies."""
        up = np.asarray(up, dtype=float)
        height = V @ up
        zmin, zmax = float(height.min()), float(height.max())
        h_max = min(BASE_FLATTEN_H_MAX, BASE_FLATTEN_H_FRAC * (zmax - zmin))
        com_h = float(com @ up - zmin)
        need = max(BASE_FLATTEN_MARGIN,
                   com_h * float(np.tan(np.radians(BASE_FLATTEN_MIN_STAB_DEG))))
        if h_max < BASE_FLATTEN_H_MIN:
            return None, -np.inf, 0.0, zmin, need
        a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(up, a); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
        P = np.stack([e1, e2], axis=1)
        com_xy = com @ P
        best = (None, -np.inf, 0.0)
        h = BASE_FLATTEN_H_MIN
        while h <= h_max + 1e-9:
            sec = scaled.section(plane_origin=up * (zmin + h), plane_normal=up)
            m, area = -np.inf, 0.0
            if sec is not None and len(sec.vertices) >= 3:
                try:
                    ch = ConvexHull(np.asarray(sec.vertices) @ P)
                    d = -(ch.equations[:, :2] @ com_xy + ch.equations[:, 2])
                    m, area = float(d.min()), float(ch.volume)
                except Exception:
                    pass
            if m > best[1]:
                best = (h, m, area)
            if m >= need:
                break
            h += BASE_FLATTEN_H_STEP
        return best[0], best[1], best[2], zmin, need

    results = [(label, up, *search(up)) for label, up in cands]
    # prefer the candidate with the largest margin-over-requirement
    label, up, h, m, area, zmin, need = max(results, key=lambda r: r[3] - r[6])
    if h is None or m < need:
        why = "; ".join(f"{r[0]}: best margin {r[3]*100:+.1f} cm (need {r[6]*100:.1f})"
                        for r in results if r[2] is not None) or "object too short"
        # No cut makes this pose statically stable (upside-down / mid-topple
        # perception pose): the object will fall onto some OTHER face, so a
        # flat facet here would not be its resting base — and a sharp-edged
        # facet it tips over is worse than the dome (measured: an upside-down
        # spatula that settled in 2 s uncut thrashed at 4-5 rad/s with an
        # unstable cut). Leave it alone.
        return coll_mesh, {"cut_mm": 0.0, "why": f"no stable cut ({why})"}
    up = np.asarray(up, dtype=float)
    try:
        cut = trimesh.intersections.slice_mesh_plane(
            scaled, plane_normal=up, plane_origin=up * (zmin + h), cap=True)
    except Exception as e:
        return coll_mesh, {"cut_mm": 0.0, "why": f"slice failed: {e}"}
    if (cut is None or len(cut.faces) < 4 or not cut.is_watertight
            or not cut.is_winding_consistent or cut.volume <= 0):
        return coll_mesh, {"cut_mm": 0.0, "why": "cut not watertight"}
    # centre of the flat cap (source units): the point the spawn-pose
    # correction pivots about, and whose height it snaps to the table top.
    on_cap = np.abs(np.asarray(cut.vertices) @ up - (zmin + h)) < 1e-6
    cap_pts = np.asarray(cut.vertices)[on_cap] if on_cap.any() else np.asarray(cut.vertices)
    cap_center = cap_pts.mean(axis=0) / float(scale)
    cut = trimesh.Trimesh(np.asarray(cut.vertices) / float(scale), cut.faces, process=False)
    return cut, {"cut_mm": h * 1000.0, "margin_cm": m * 100.0, "need_cm": need * 100.0,
                 "base_cm2": area * 1e4, "how": label,
                 "up_used": tuple(float(v) for v in up),
                 "cap_center_src": tuple(float(v) for v in cap_center)}


def _author_collision_prim(usd_path, coll_mesh):
    """Author `coll_mesh` as an invisible collider prim in the converted USD.

    The standard Isaac/USD split: the full-resolution capture stays the render
    mesh (MeshConverter runs with NO collision props), and the collider is a
    separate `purpose=guide` Mesh — render passes skip guide prims, PhysX
    parses them regardless. The prim goes UNDER <defaultPrim>/geometry, whose
    xformOps carry MeshConverter's Y-up->Z-up rotation and uniform scale, so
    the vertices are written in RAW SOURCE-frame coordinates and inherit
    exactly the visual transform. Downstream helpers
    (_set_collision_approximation, _bake_friction_into_usd, spawn-time
    CollisionPropertiesCfg) all target prims that HAVE CollisionAPI — after
    this call, that is only this prim."""
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    stage = Usd.Stage.Open(usd_path)
    geom_path = stage.GetDefaultPrim().GetPath().AppendChild("geometry")
    mesh = UsdGeom.Mesh.Define(stage, geom_path.AppendChild("collision"))
    mesh.CreatePointsAttr(
        Vt.Vec3fArray.FromNumpy(np.asarray(coll_mesh.vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(coll_mesh.faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.asarray(coll_mesh.faces.ravel(), dtype=np.int32)))
    mesh.CreateExtentAttr(
        Vt.Vec3fArray.FromNumpy(np.asarray(coll_mesh.bounds, dtype=np.float32)))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()


def _finger_actuator_cfg():
    """Force-mode finger drive WITH viscous damping (stiffness=0 keeps the
    effort-target semantics: close is still a constant -20 N per finger).
    The damping is the point: with the drive fully undamped (old value 0.0)
    the finger-object-finger pinch had no dissipation anywhere but friction,
    so the 40 N squeeze on a ~100 g object chattered at the step rate — the
    in-jaw buzzing and "object pops out of the closed jaws" glitch.
    damping=1e2 adds a -100*qd [N] braking term inside the implicit solver:
    approach speed still tops out at the URDF-baked 0.2 m/s maxJointVelocity,
    but contact-cycle energy now bleeds off instead of accumulating.
    effort_limit_sim stays 200 N so the braking force is never clipped."""
    return ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        effort_limit_sim=200.0, stiffness=0.0, damping=1e2,
    )


def _object_rigid_props():
    """Rigid-body props shared by every spawned object, tuned for SDF-collider
    stability (no resting jitter, no contact 'explosions'). A factory (not a
    shared instance) so each cfg gets its own copy. See the OBJECT_* constants."""
    return RigidBodyPropertiesCfg(
        rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False,
        solver_position_iteration_count=OBJECT_SOLVER_POS_ITERS,
        solver_velocity_iteration_count=OBJECT_SOLVER_VEL_ITERS,
        max_depenetration_velocity=OBJECT_MAX_DEPEN_VEL,
        linear_damping=OBJECT_LINEAR_DAMPING,
        angular_damping=OBJECT_ANGULAR_DAMPING,
        stabilization_threshold=OBJECT_STABILIZATION_THRESH,
        sleep_threshold=OBJECT_SLEEP_THRESH,
    )

def _object_collision_props():
    """Spawn-time collider props shared by every spawned object (applied to the
    guide collider prim — the only prim with CollisionAPI). See the
    OBJECT_TORSIONAL_PATCH_* constants for why the patch radius matters for
    resting stability. A factory, matching _object_rigid_props."""
    return CollisionPropertiesCfg(
        collision_enabled=True,
        torsional_patch_radius=OBJECT_TORSIONAL_PATCH_RADIUS,
        min_torsional_patch_radius=OBJECT_MIN_TORSIONAL_PATCH_RADIUS,
    )

def convert_mesh(mesh_path, scale, out_name, mesh_format, content_hash, up_src=None):
    """Convert a single mesh file -> USD at the given (uniform) scale. Returns usd_path.

    The RAW mesh becomes the (collision-free) render mesh, and the sanitized/
    decimated collision mesh is authored beside it as an invisible guide prim
    (_build_collision_mesh / _author_collision_prim) — cameras see the
    full-resolution capture while PhysX cooks the SDF from the cheap collider.

    The output is CACHED by (content_hash, format, scale): identical geometry —
    which the BO loop resends every round, and which every re-exec rebuilds —
    reuses the already-cooked USD instead of re-decimating and re-cooking the
    SDF. This also bounds /tmp growth by the number of DISTINCT meshes.
    `content_hash` must be the sha256 of the ORIGINAL request bytes.
    `out_name` is only used to keep the cache directory human-readable.
    `up_src` (from _source_up_vector) selects the resting-base flattening cut
    and is part of the cache key; None -> no cut."""
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    fmt = str(mesh_format).lstrip(".").lower()
    s   = float(scale)
    safe_name   = str(out_name).replace(" ", "_")
    # "split_thick" keys this USD layout away from cached USDs cooked when the
    # decimated mesh doubled as the render mesh ("split" came in with the
    # visual/collision split, "_thick" with the thin-shell collider fix).
    obj_usd_dir = os.path.join(
        USD_DIR, f"{safe_name}_{content_hash[:12]}_{fmt}_split_thick{_up_cache_tag(up_src)}_{s:.6f}")
    usd_path    = os.path.join(obj_usd_dir, "object.usd")

    if os.path.isfile(usd_path):
        print(f"  [CONV] '{out_name}' cache hit → {usd_path}", flush=True)
        return usd_path

    coll_mesh, n_in, how = _build_collision_mesh(mesh_path, scale=s)
    alias_dir = None
    if up_src is not None:
        coll_mesh, info = _flatten_base(coll_mesh, up_src, s)
        if info["cut_mm"] > 0:
            how += (f" + base planed {info['cut_mm']:.1f} mm {info['how']} (COM margin "
                    f"{info['margin_cm']:+.1f} cm >= {info['need_cm']:.1f}, "
                    f"base {info['base_cm2']:.0f} cm2)")
            # A re-uprighted object settles onto the cap, so its SETTLED pose
            # reports up == info["up_used"]; alias that key to this USD so the
            # eval/rollout worlds built from the settled pose hit the cache.
            if _up_cache_tag(info["up_used"]) != _up_cache_tag(up_src):
                alias_dir = os.path.join(
                    USD_DIR, f"{safe_name}_{content_hash[:12]}_{fmt}_split_thick"
                             f"{_up_cache_tag(info['up_used'])}_{s:.6f}")
        else:
            how += f" + base NOT planed ({info.get('why', '?')})"
    print(f"  [MESH] '{out_name}' collider: {n_in} -> {len(coll_mesh.faces)} "
          f"faces ({how})", flush=True)

    print(f"  [CONV] '{out_name}' (scale={s:.4f}) ...", end=" ", flush=True)

    # Fixed axis-fix rotation for glTF-style (Y-up) source meshes -> Z-up.
    # Non-glTF formats (obj/ply/stl) are assumed already Z-up and pass through.
    conv_rot = _conversion_rotation(fmt)

    cfg = MeshConverterCfg(
        asset_path=mesh_path,         # RAW capture: the render mesh stays full-res
        usd_dir=obj_usd_dir,
        usd_file_name="object.usd",   # deterministic path so the cache check above works
        translation=(0.0, 0.0, 0.0),
        rotation=conv_rot,
        scale=(s, s, s),
        make_instanceable=False,
        rigid_props=_object_rigid_props(),
        # NO collision_props: the visual meshes stay collision-free; the guide
        # prim authored below is the only collider.
        mass_props=MassPropertiesCfg(mass=OBJECT_MASS_KG),
    )
    converter = MeshConverter(cfg)
    _author_collision_prim(converter.usd_path, coll_mesh)
    n = _set_collision_approximation(converter.usd_path, COLLISION_APPROX)
    _bake_friction_into_usd(converter.usd_path)
    print(f"→ {converter.usd_path}  ({n} collider(s) → {COLLISION_APPROX})")
    _write_planed_sidecar(obj_usd_dir, up_src, info if up_src is not None else None, s)
    if alias_dir is not None and not os.path.exists(alias_dir):
        try:
            os.symlink(obj_usd_dir, alias_dir)
        except OSError:
            pass
    return converter.usd_path

PLANED_SIDECAR = "planed.json"
_PLANED_CACHE = {}


def _write_planed_sidecar(obj_usd_dir, up_src, info, scale):
    """Record what the resting-base cut did next to the cooked USD, so a later
    process (cache hit) can still correct spawn poses (_upright_env_pose)."""
    rec = {"cut_mm": 0.0}
    if info is not None and info.get("cut_mm", 0.0) > 0:
        rec = {"cut_mm": info["cut_mm"], "up_src": list(up_src), "scale": float(scale),
               "up_used": list(info["up_used"]), "cap_center_src": list(info["cap_center_src"]),
               "how": info["how"]}
    try:
        with open(os.path.join(obj_usd_dir, PLANED_SIDECAR), "w") as f:
            json.dump(rec, f)
    except OSError:
        pass


def _planed_info(usd_path):
    """The planed.json record for a cooked USD (cached per path); {"cut_mm": 0}
    for USDs cooked without a cut / before sidecars existed."""
    rec = _PLANED_CACHE.get(usd_path)
    if rec is None:
        rec = {"cut_mm": 0.0}
        try:
            with open(os.path.join(os.path.dirname(usd_path), PLANED_SIDECAR)) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            pass
        _PLANED_CACHE[usd_path] = rec
    return rec


def _upright_env_pose(env_pos, env_rot_wxyz, usd_path, mesh_format):
    """Spawn-pose correction for a planed object: rotate the requested pose by
    the minimal rotation that makes the flat cap horizontal (pivoting about the
    cap centre so the footprint stays put) and set its height so the cap sits
    SPAWN_TABLE_CLEARANCE above the tabletop. The object then starts AT REST
    on its cap instead of toppling 5-15 deg onto it from the perception tilt
    (measured: that topple sent tall bottles walking into their neighbours and
    left leaning pairs jittering indefinitely). Identity for objects that were
    not planed. Returns (env_pos, env_rot_wxyz) as tuples."""
    rec = _planed_info(usd_path)
    if not rec.get("cut_mm", 0.0) or "up_used" not in rec:
        return tuple(float(v) for v in env_pos), tuple(float(v) for v in env_rot_wxyz)
    q = np.asarray(env_rot_wxyz, dtype=float)
    r_obj = Rot.from_quat([q[1], q[2], q[3], q[0]])
    c = _conversion_rotation(mesh_format)
    r_tot = r_obj * Rot.from_quat([c[1], c[2], c[3], c[0]])       # source -> env
    u_w = r_tot.apply(np.asarray(rec["up_used"], dtype=float))
    u_w = u_w / np.linalg.norm(u_w)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u_w, z)
    sin_a, cos_a = float(np.linalg.norm(axis)), float(np.clip(u_w @ z, -1.0, 1.0))
    ang = float(np.arctan2(sin_a, cos_a))
    r_delta = Rot.identity() if sin_a < 1e-8 else Rot.from_rotvec(axis / sin_a * ang)
    scale = float(rec.get("scale", 1.0))
    p = np.asarray(env_pos, dtype=float)
    cap_w = p + r_tot.apply(np.asarray(rec["cap_center_src"], dtype=float) * scale)
    p_new = cap_w + r_delta.apply(p - cap_w)
    # Snap the cap onto the tabletop — but only when the request already has it
    # near the table (perception z error is ~+-2 cm); a cap far above the table
    # is an object resting on something else, leave its height alone.
    dz = (TABLE_TOP_Z_ENV + SPAWN_TABLE_CLEARANCE) - cap_w[2]
    if abs(dz) <= SPAWN_TABLE_SNAP_MAX:
        p_new[2] += dz
    r_new = r_delta * r_obj
    x, y, zq, w = r_new.as_quat()
    return tuple(float(v) for v in p_new), (float(w), float(x), float(y), float(zq))


def _convert_object_entry(name, mesh_bytes, mesh_format, scale, rotation_wxyz=None):
    """materialize -> convert (cached) -> delete temp file. Returns usd_path.
    Shared by all three world builders so the hash/temp-file dance lives once.
    `rotation_wxyz` is the object's spawn rotation (robot/env frame, same
    thing): it fixes which side of the mesh is the resting base for the
    flattening cut (see BASE_FLATTEN)."""
    content_hash = hashlib.sha256(bytes(mesh_bytes)).hexdigest()
    mesh_path = _materialize_mesh(mesh_bytes, mesh_format)
    up_src = _source_up_vector(rotation_wxyz, mesh_format)
    try:
        return convert_mesh(mesh_path, scale, out_name=name,
                            mesh_format=mesh_format, content_hash=content_hash,
                            up_src=up_src)
    finally:
        try:
            os.remove(mesh_path)
        except OSError:
            pass

# ==============================================================================
#  SCENE CONFIG
# ==============================================================================
def _extra_slot_name(i):
    return f"extra_{i}"


def _warn_coincident_objects(named_poses, where):
    """Loud warning for two objects spawned at (near-)identical poses. The
    perception stage has been seen handing the SAME reconstruction to two
    labels (benchmark scene_6 'spray bottle'/'wipes container', scene_7 'glass
    bottle'/'spray bottle': identical translation, rotation and scale). Two
    fully overlapping rigid bodies fight the depenetration solver forever and
    thrash everything around them — it looks exactly like a physics bug but is
    a data bug, so name it. `named_poses`: [(name, translation_xyz), ...]."""
    for i in range(len(named_poses)):
        for j in range(i + 1, len(named_poses)):
            (na, pa), (nb, pb) = named_poses[i], named_poses[j]
            d = float(np.linalg.norm(np.asarray(pa, float) - np.asarray(pb, float)))
            if d < 0.01:
                print(f"[{where}] WARNING: {na!r} and {nb!r} are spawned {d*100:.1f} cm apart "
                      f"— almost certainly the same detection under two labels. The two "
                      f"bodies will interpenetrate and thrash; fix the perception output "
                      f"(dedupe masks) rather than the physics.", flush=True)

def build_scene_cfg(num_envs, env_spacing, usd_path, env_pos, env_rot,
                    extra_specs=None):
    @configclass
    class GraspScene(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 1.0)),
        )
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(2.0, 2.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_TRANSLATION),
        )
        robot: ArticulationCfg = RUBBER_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

    cfg = GraspScene(num_envs=num_envs, env_spacing=env_spacing)
    cfg.robot.init_state.pos = ROBOT_BASE_POS
    cfg.robot.actuators["panda_hand"] = _finger_actuator_cfg()

    cfg.target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=_object_rigid_props(),
            collision_props=_object_collision_props(),
            mass_props=MassPropertiesCfg(mass=OBJECT_MASS_KG),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=env_pos, rot=env_rot),
    )

    # Clearance-promoted neighbors: dynamic rigid bodies in EVERY env, sitting
    # at their fixed (settled) poses beside the target. Slot names are
    # positional, same convention as the settle scene.
    for i, spec in enumerate(extra_specs or []):
        setattr(cfg, _extra_slot_name(i), RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + _extra_slot_name(i),
            spawn=sim_utils.UsdFileCfg(
                usd_path=spec["usd_path"],
                rigid_props=_object_rigid_props(),
                collision_props=_object_collision_props(),
                mass_props=MassPropertiesCfg(mass=OBJECT_MASS_KG),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=spec["env_pos"], rot=spec["env_rot"]),
        ))
    return cfg

def _physx_cfg(num_envs):
    """PhysX GPU buffer sizes scaled to the parallel-env count.

    The collision stack, rigid contact/patch buffers and temp heap all grow
    ~linearly with the number of envs; undersizing any of them makes PhysX
    silently drop contacts ("PxGpuDynamicsMemoryConfig::collisionStackSize
    buffer overflow"). A 1000-env grasp batch measured ~0.29 MB/env of collision
    stack, so we budget 0.5 MB/env (~70% headroom) with floors matching the old
    small-batch defaults.

    These map onto USD *uint32* attributes on the PhysX scene, so every value
    must stay below 2**32 or Isaac aborts the build with "Type mismatch ...
    expected 'unsigned int', got 'long'". That hard-caps the byte-sized buffers
    at ~4 GB regardless of env count: at the measured ~0.29 MB/env that still
    covers ~13k envs before PhysX starts dropping contacts, past which you must
    shrink the batch (no USD knob can go higher)."""
    u32_max = (1 << 32) - 1
    return sim_utils.PhysxCfg(
        enable_stabilization=True,   # low-speed resting-contact stabilization (thin-SDF jitter)
        gpu_collision_stack_size=u32_max,  # PxU32 ceiling (~4.29 GB): multi-object envs need ~0.62 MB/env, the old 0.5 MB/env budget overflowed
        gpu_max_rigid_contact_count=max(2**22, num_envs * 4096),
        gpu_max_rigid_patch_count=max(2**18, num_envs * 256),
        gpu_heap_capacity=min(u32_max, max(256 * 1024 * 1024, num_envs * 256 * 1024)),         # ~0.25 MB/env, capped 4 GB
    )


def phys_tick(sim, scene, render=False):
    """Advance ONE 60 Hz control tick = PHYS_STEPS_PER_TICK physics steps at
    PHYS_DT. Joint targets persist in the asset buffers, so re-pushing them via
    write_data_to_sim() each substep is idempotent and keeps the implicit
    drives fed. Rendering (viewer/cameras) is one sim.render() on the last
    substep — a draw-only app update (Isaac Lab forces playSimulations off
    inside render()). NEVER sim.step(render=True): that goes through
    app.update() with physics live, which advances the timeline by
    rendering_dt and steps physics PHYS_STEPS_PER_TICK MORE times — a rendered
    tick used to run 7 substeps (29 ms sim time) vs the headless 4 (17 ms), so
    --render / visualize runs weren't comparable to headless evaluation.
    step(render=False) + render() is the split Isaac Lab's own envs use
    (manager_based_env.step)."""
    for i in range(PHYS_STEPS_PER_TICK):
        scene.write_data_to_sim()
        sim.step(render=False)
        if render and i == PHYS_STEPS_PER_TICK - 1:
            sim.render()
        scene.update(PHYS_DT)


def _bleed_velocities(objs, factor=None):
    """Scale every object's root velocity — the settle-phase rocking damper
    (see SETTLE_BLEED_FACTOR). Call once per control tick during settle-type
    stepping only; factor=0.0 hard-stops the objects at the settle->eval
    handoff so captured rest states are exactly at rest."""
    f = SETTLE_BLEED_FACTOR if factor is None else factor
    for obj in objs:
        obj.write_root_velocity_to_sim(obj.data.root_vel_w * f)


def _objects_still(objs, anchor):
    """One settle-convergence probe, taken right after a physics tick and BEFORE
    the bleed: True iff every object in every env has pre-bleed root speed under
    the SETTLE_*_VEL_TOL and (if `anchor` poses are given) has drifted less than
    SETTLE_POS_TOL / SETTLE_ROT_TOL from them. Also returns the peak speeds for
    the not-converged report."""
    lin_max = ang_max = 0.0
    still = True
    for k, obj in enumerate(objs):
        v = obj.data.root_vel_w
        lin = float(v[:, :3].norm(dim=-1).max())
        ang = float(v[:, 3:].norm(dim=-1).max())
        lin_max, ang_max = max(lin_max, lin), max(ang_max, ang)
        if lin > SETTLE_LIN_VEL_TOL or ang > SETTLE_ANG_VEL_TOL:
            still = False
        elif anchor is not None:
            p0, q0 = anchor[k]
            dp = float((obj.data.root_pos_w - p0).norm(dim=-1).max())
            dot = (obj.data.root_quat_w * q0).sum(dim=-1).abs().clamp(max=1.0)
            dq = float((2.0 * torch.acos(dot)).max())
            if dp > SETTLE_POS_TOL or dq > SETTLE_ROT_TOL:
                still = False
    return still, lin_max, ang_max


def settle_until_still(objs, tick, min_ticks, max_ticks, label="settle", verbose=False):
    """Settle-type stepping with a convergence test (see SETTLE_LIN_VEL_TOL).

    `tick()` is the caller's one-control-tick closure (hold the robot, then
    phys_tick); it may return False to abort (app closed). Each tick: step,
    probe stillness pre-bleed, bleed. Runs at least `min_ticks`, then stops as
    soon as the objects have been still for SETTLE_QUIET_TICKS consecutive
    ticks, or at `max_ticks`. Always ends with the hard zero, so the caller's
    captured rest state is exactly at rest. Returns (ticks_run, converged)."""
    quiet, anchor, n = 0, None, 0
    lin_max = ang_max = 0.0
    for n in range(1, max_ticks + 1):
        if tick() is False:
            break
        still, lin_max, ang_max = _objects_still(objs, anchor)
        if still:
            if anchor is None:   # quiet window opens: remember where we were
                anchor = [(o.data.root_pos_w.clone(), o.data.root_quat_w.clone())
                          for o in objs]
            quiet += 1
        else:
            quiet, anchor = 0, None
        _bleed_velocities(objs)
        if n >= min_ticks and quiet >= SETTLE_QUIET_TICKS:
            break
    _bleed_velocities(objs, factor=0.0)
    converged = quiet >= SETTLE_QUIET_TICKS
    if not converged:
        print(f"[{label}] WARNING: objects not still after {n} ticks (cap {max_ticks}); "
              f"residual pre-bleed speed lin={lin_max:.4f} m/s ang={ang_max:.4f} rad/s "
              f"— hard-zeroed anyway.", flush=True)
    elif verbose:
        print(f"[{label}] still after {n} ticks (min {min_ticks}, cap {max_ticks}).",
              flush=True)
    return n, converged


# ==============================================================================
#  ONE-TIME SCENE BUILD + WARMUP
# ==============================================================================
def build_world(target_name, mesh_bytes, mesh_format, translation, rotation, scale,
                batch_size, lift_height, extra_objects=None):
    """Create SimulationContext -> InteractiveScene -> reset -> warmup, ONCE.

    `extra_objects` (optional) is a list of settle-style dicts {name, mesh_bytes,
    mesh_format, translation, rotation, scale}: the clearance-promoted neighbors
    spawned beside the target in every env. They settle in the warmup with the
    target and evaluate() resets them to that settled state each batch; per-env
    scene_poses never move them (domain randomization is target-only)."""
    # MeshConverter only reads files, so stage the in-request bytes on disk
    # briefly, convert (content-hash cached), then drop the temp file.
    usd_path = _convert_object_entry(target_name, mesh_bytes, mesh_format, scale,
                                     rotation_wxyz=rotation)
    env_pos, env_rot = robot_pose_to_env(translation, rotation)
    env_pos, env_rot = _upright_env_pose(env_pos, env_rot, usd_path, mesh_format)
    _warn_coincident_objects([(target_name, translation)]
                             + [(o["name"], o["translation"]) for o in (extra_objects or [])],
                             "sim")

    extra_specs = []
    for obj in (extra_objects or []):
        e_usd = _convert_object_entry(obj["name"], obj["mesh_bytes"],
                                      obj["mesh_format"], obj["scale"],
                                      rotation_wxyz=obj["rotation"])
        e_pos, e_rot = robot_pose_to_env(obj["translation"], obj["rotation"])
        e_pos, e_rot = _upright_env_pose(e_pos, e_rot, e_usd, obj["mesh_format"])
        extra_specs.append({"name": obj["name"], "usd_path": e_usd,
                            "mesh_format": obj["mesh_format"],
                            "env_pos": e_pos, "env_rot": e_rot})

    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYS_DT,
        render_interval=PHYS_STEPS_PER_TICK,   # one rendered frame per 60 Hz tick
        physx=_physx_cfg(batch_size),
        device=_physics_device("grasp"),
    )

    # Tight, synchronous sequence — context -> scene -> reset. No idle gap.
    sim = SimulationContext(sim_cfg)
    scene_cfg = build_scene_cfg(batch_size, ENV_SPACING, usd_path, env_pos, env_rot,
                                extra_specs)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot:  Articulation = scene["robot"]
    target: RigidObject  = scene["target"]
    device = sim.device

    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_cfg.resolve(scene)
    arm_jids, hand_idx = arm_cfg.joint_ids, arm_cfg.body_ids[0]

    finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    finger_cfg.resolve(scene)
    finger_jids = finger_cfg.joint_ids
    ee_jacobi_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx

    # Position-only IK: the lift constrains EE x/y/z but leaves orientation (rpy)
    # free. Demanding a fixed EE orientation while raising straight up was driving
    # the wrist into singularities (the "shoot up fast" snaps); a 3-DOF task on the
    # 7-DOF arm has a large null space and stays well-conditioned.
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=batch_size, device=device)

    open_grip_pos  = torch.tensor([[0.04, 0.04]] * batch_size, device=device, dtype=torch.float32)
    grip_close_eff = torch.full((batch_size, 2), -20.0, device=device, dtype=torch.float32)
    grip_open_eff  = torch.full((batch_size, 2),  10.0, device=device, dtype=torch.float32)
    home_arm_pose  = robot.data.default_joint_pos[:, arm_jids].clone()

    # ── Settle warmup, ONCE ──────────────────────────────────────────────────
    print(f"[sim] One-time warmup settlement ({WARMUP_STEPS}..{WARMUP_MAX_STEPS} steps, "
          f"until still)...", flush=True)
    settle_objs = [target] + [scene[_extra_slot_name(i)] for i in range(len(extra_specs))]

    def _warmup_tick():
        robot.set_joint_position_target(home_arm_pose, joint_ids=arm_jids)
        robot.set_joint_effort_target(grip_open_eff, joint_ids=finger_jids)
        phys_tick(sim, scene, render=RENDER)
    settle_until_still(settle_objs, _warmup_tick, WARMUP_STEPS, WARMUP_MAX_STEPS,
                       label="sim warmup", verbose=True)

    init_target_state = torch.zeros((batch_size, 13), device=device, dtype=torch.float32)
    init_target_state[:, 0:3] = target.data.root_pos_w
    init_target_state[:, 3:7] = target.data.root_quat_w
    settled_pos_world = target.data.root_pos_w.detach().cpu().numpy().copy()

    # Post-warmup settled state of each extra object, so every batch can reset
    # them (a grasp can shove a neighbor; the next batch must not inherit that).
    extra_objs = []
    for i, spec in enumerate(extra_specs):
        ro: RigidObject = scene[_extra_slot_name(i)]
        st = torch.zeros((batch_size, 13), device=device, dtype=torch.float32)
        st[:, 0:3] = ro.data.root_pos_w
        st[:, 3:7] = ro.data.root_quat_w
        extra_objs.append({"name": spec["name"], "obj": ro, "init_state": st,
                           "usd_path": spec["usd_path"], "mesh_format": spec["mesh_format"]})
    if extra_objs:
        print(f"[sim] Grasp world carries {len(extra_objs)} extra object(s): "
              f"{[e['name'] for e in extra_objs]}", flush=True)
    print("[sim] Warmup complete. Scene is live and ready for requests.", flush=True)

    return types.SimpleNamespace(
        kind="grasp",
        usd_path=usd_path, mesh_format=mesh_format,
        sim=sim, scene=scene, robot=robot, target=target, device=device,
        arm_jids=arm_jids, hand_idx=hand_idx, finger_jids=finger_jids,
        ee_jacobi_idx=ee_jacobi_idx, ik=ik,
        open_grip_pos=open_grip_pos, grip_close_eff=grip_close_eff,
        grip_open_eff=grip_open_eff, home_arm_pose=home_arm_pose,
        init_target_state=init_target_state, settled_pos_world=settled_pos_world,
        batch_size=batch_size, lift_height=lift_height, target_name=target_name,
        extra_objects=extra_objs,
    )

# ==============================================================================
#  SETTLE SCENE BUILD (multi-object, single env)
# ==============================================================================
def _settle_slot_name(i):
    return f"obj_{i}"

def build_settle_world(objects_list):
    """Build a single-env scene with the Franka at home + N dynamic RigidObjects
    spawned at their (robot-base -> env) poses, then sim.reset(). Physics
    stepping happens in settle(); the build itself doesn't step.

    objects_list: list of dicts {name, mesh_bytes, mesh_format, translation,
    rotation, scale}. translation/rotation are in the ROBOT-BASE frame (same
    convention as the grasp request)."""
    # 1) Convert every mesh to USD and compute its env pose.
    _warn_coincident_objects([(o["name"], o["translation"]) for o in objects_list], "settle")
    object_specs = []
    for obj in objects_list:
        usd_path = _convert_object_entry(obj["name"], obj["mesh_bytes"],
                                         obj["mesh_format"], obj["scale"],
                                         rotation_wxyz=obj["rotation"])
        env_pos, env_rot = robot_pose_to_env(obj["translation"], obj["rotation"])
        env_pos, env_rot = _upright_env_pose(env_pos, env_rot, usd_path, obj["mesh_format"])
        object_specs.append({
            "name":     obj["name"],
            "usd_path": usd_path,
            "scale":    float(obj["scale"]),
            "env_pos":  env_pos,
            "env_rot":  env_rot,
        })

    # 2) Build a SimulationContext + single-env InteractiveScene.
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYS_DT,
        render_interval=PHYS_STEPS_PER_TICK,   # one rendered frame per 60 Hz tick
        physx=_physx_cfg(1),   # single env
        device=_physics_device("settle"),
    )
    sim = SimulationContext(sim_cfg)

    @configclass
    class SettleScene(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 1.0)),
        )
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(2.0, 2.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_TRANSLATION),
        )
        robot: ArticulationCfg = RUBBER_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

    scene_cfg = SettleScene(num_envs=1, env_spacing=ENV_SPACING)
    scene_cfg.robot.init_state.pos = ROBOT_BASE_POS
    scene_cfg.robot.actuators["panda_hand"] = _finger_actuator_cfg()

    # Attach each object as its own scene entity. Slot names are positional so
    # name collisions in the request don't clash with python attribute rules.
    for i, spec in enumerate(object_specs):
        slot = _settle_slot_name(i)
        setattr(scene_cfg, slot, RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + slot,
            spawn=sim_utils.UsdFileCfg(
                usd_path=spec["usd_path"],
                rigid_props=_object_rigid_props(),
                collision_props=_object_collision_props(),
                mass_props=MassPropertiesCfg(mass=OBJECT_MASS_KG),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=spec["env_pos"], rot=spec["env_rot"]),
        ))

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    device = sim.device

    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_cfg.resolve(scene)
    arm_jids = arm_cfg.joint_ids

    finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    finger_cfg.resolve(scene)
    finger_jids = finger_cfg.joint_ids

    home_arm_pose = robot.data.default_joint_pos[:, arm_jids].clone()
    grip_open_eff = torch.full((1, 2), 10.0, device=device, dtype=torch.float32)
    open_grip_pos = torch.tensor([[0.04, 0.04]], device=device, dtype=torch.float32)

    rigid_objects = []
    for i, spec in enumerate(object_specs):
        rigid_objects.append({
            "name":  spec["name"],
            "scale": spec["scale"],
            "obj":   scene[_settle_slot_name(i)],
        })

    # Capture every object's spawn state (the REQUESTED poses, pre-stepping) so
    # settle() can reset to it on every call — this is what makes a resent
    # identical settle request idempotent instead of compounding physics.
    scene.update(0.0)
    init_states = []
    for ro in rigid_objects:
        st = torch.zeros((1, 13), device=device, dtype=torch.float32)
        st[:, 0:3] = ro["obj"].data.root_pos_w
        st[:, 3:7] = ro["obj"].data.root_quat_w
        init_states.append(st)

    print(f"[settle] Scene built with {len(rigid_objects)} object(s). Ready to step.",
          flush=True)

    return types.SimpleNamespace(
        kind="settle",
        sim=sim, scene=scene, robot=robot, device=device,
        arm_jids=arm_jids, finger_jids=finger_jids,
        home_arm_pose=home_arm_pose, grip_open_eff=grip_open_eff,
        open_grip_pos=open_grip_pos,
        rigid_objects=rigid_objects, init_states=init_states,
    )

# ==============================================================================
#  ROLLOUT SCENE BUILD (multi-object + full grasp machinery)
# ==============================================================================
def build_rollout_world(objects_list, target_name, batch_size, lift_height):
    """Build a scene with EVERY mesh spawned as a dynamic RigidObject (like the
    settle scene) PLUS the full grasp machinery (IK controller, gripper targets).

    Exactly one object — `target_name` — is the one the rollout grasps and lifts;
    the rest are spawned as the surrounding clutter so the demo runs against the
    real scene instead of just the isolated target. Mirrors build_world's grasp
    setup, but with N objects instead of one.

    NOTE: callers pass ROLLOUT_BATCH_SIZE (=1) — rollout only ever demos row 0
    broadcast across the batch, so building the evaluation batch here is waste.

    objects_list: list of dicts {name, mesh_bytes, mesh_format, translation,
    rotation, scale} in the ROBOT-BASE frame (same convention as the settle
    request)."""
    # 1) Convert every mesh to USD and compute its env pose. Track which slot is
    #    the grasp target.
    _warn_coincident_objects([(o["name"], o["translation"]) for o in objects_list], "rollout")
    object_specs = []
    target_spec_idx = None
    for i, obj in enumerate(objects_list):
        usd_path = _convert_object_entry(obj["name"], obj["mesh_bytes"],
                                         obj["mesh_format"], obj["scale"],
                                         rotation_wxyz=obj["rotation"])
        env_pos, env_rot = robot_pose_to_env(obj["translation"], obj["rotation"])
        env_pos, env_rot = _upright_env_pose(env_pos, env_rot, usd_path, obj["mesh_format"])
        object_specs.append({
            "name":     obj["name"],
            "usd_path": usd_path,
            "scale":    float(obj["scale"]),
            "env_pos":  env_pos,
            "env_rot":  env_rot,
        })
        if obj["name"] == target_name:
            target_spec_idx = i
    if target_spec_idx is None:
        raise ValueError(f"rollout target {target_name!r} not found in objects list")

    # 2) SimulationContext + InteractiveScene with the Franka and N objects.
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYS_DT,
        render_interval=PHYS_STEPS_PER_TICK,   # one rendered frame per 60 Hz tick
        physx=_physx_cfg(batch_size),
        device=_physics_device("rollout"),
    )
    sim = SimulationContext(sim_cfg)

    @configclass
    class RolloutScene(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 1.0)),
        )
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(2.0, 2.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_TRANSLATION),
        )
        robot: ArticulationCfg = RUBBER_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

    scene_cfg = RolloutScene(num_envs=batch_size, env_spacing=ENV_SPACING)
    scene_cfg.robot.init_state.pos = ROBOT_BASE_POS
    scene_cfg.robot.actuators["panda_hand"] = _finger_actuator_cfg()

    # Attach each object as its own positional scene entity (same slot scheme as
    # the settle scene so name collisions can't clash with attribute rules).
    for i, spec in enumerate(object_specs):
        slot = _settle_slot_name(i)
        setattr(scene_cfg, slot, RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + slot,
            spawn=sim_utils.UsdFileCfg(
                usd_path=spec["usd_path"],
                rigid_props=_object_rigid_props(),
                collision_props=_object_collision_props(),
                mass_props=MassPropertiesCfg(mass=OBJECT_MASS_KG),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=spec["env_pos"], rot=spec["env_rot"]),
        ))

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot:  Articulation = scene["robot"]
    device = sim.device

    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_cfg.resolve(scene)
    arm_jids, hand_idx = arm_cfg.joint_ids, arm_cfg.body_ids[0]

    finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    finger_cfg.resolve(scene)
    finger_jids = finger_cfg.joint_ids
    ee_jacobi_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx

    # Position-only IK: the lift constrains EE x/y/z but leaves orientation (rpy)
    # free. Demanding a fixed EE orientation while raising straight up was driving
    # the wrist into singularities (the "shoot up fast" snaps); a 3-DOF task on the
    # 7-DOF arm has a large null space and stays well-conditioned.
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=batch_size, device=device)

    open_grip_pos  = torch.tensor([[0.04, 0.04]] * batch_size, device=device, dtype=torch.float32)
    grip_close_eff = torch.full((batch_size, 2), -20.0, device=device, dtype=torch.float32)
    grip_open_eff  = torch.full((batch_size, 2),  10.0, device=device, dtype=torch.float32)
    home_arm_pose  = robot.data.default_joint_pos[:, arm_jids].clone()

    rigid_objects = []
    for i, spec in enumerate(object_specs):
        rigid_objects.append({
            "name":  spec["name"],
            "scale": spec["scale"],
            "obj":   scene[_settle_slot_name(i)],
        })

    # ── Settle warmup, ONCE — let every object drop into place ────────────────
    print(f"[rollout] One-time warmup settlement ({WARMUP_STEPS}..{WARMUP_MAX_STEPS} steps, "
          f"until still) with {len(rigid_objects)} object(s)...", flush=True)
    settle_objs = [ro["obj"] for ro in rigid_objects]

    def _warmup_tick():
        robot.set_joint_position_target(home_arm_pose, joint_ids=arm_jids)
        robot.set_joint_effort_target(grip_open_eff, joint_ids=finger_jids)
        phys_tick(sim, scene, render=RENDER)
    settle_until_still(settle_objs, _warmup_tick, WARMUP_STEPS, WARMUP_MAX_STEPS,
                       label="rollout warmup", verbose=True)

    # Capture the settled state of EVERY object so each rollout cycle can reset
    # the whole scene (not just the target) back to the warmed-up configuration.
    init_states = []
    for ro in rigid_objects:
        st = torch.zeros((batch_size, 13), device=device, dtype=torch.float32)
        st[:, 0:3] = ro["obj"].data.root_pos_w
        st[:, 3:7] = ro["obj"].data.root_quat_w
        init_states.append(st)

    target_obj: RigidObject = scene[_settle_slot_name(target_spec_idx)]
    init_target_state = init_states[target_spec_idx]
    settled_pos_world = target_obj.data.root_pos_w.detach().cpu().numpy().copy()
    print(f"[rollout] Warmup complete. Grasp target = '{target_name}'. "
          f"Scene is live and ready.", flush=True)

    return types.SimpleNamespace(
        kind="rollout",
        sim=sim, scene=scene, robot=robot, target=target_obj, device=device,
        arm_jids=arm_jids, hand_idx=hand_idx, finger_jids=finger_jids,
        ee_jacobi_idx=ee_jacobi_idx, ik=ik,
        open_grip_pos=open_grip_pos, grip_close_eff=grip_close_eff,
        grip_open_eff=grip_open_eff, home_arm_pose=home_arm_pose,
        init_target_state=init_target_state, settled_pos_world=settled_pos_world,
        rigid_objects=rigid_objects, init_states=init_states,
        batch_size=batch_size, lift_height=lift_height, target_name=target_name,
    )

# ==============================================================================
#  PER-REQUEST EVALUATION (no rebuild, no reset, no stop)
# ==============================================================================
def _nan_result(global_gi, reason="nan_input", n_extra=0):
    nan = float("nan")
    nan3 = [nan, nan, nan]
    return {
        "index": global_gi, "passed": False, "fail_reason": reason,
        "lift_distance": nan, "roll_offset": nan, "pitch_offset": nan,
        "yaw_offset": nan, "gripper_width_diff": nan, "closed_gripper_width": nan,
        # Pre-close state (right before GRIP_STEPS).
        "pre_close_ee_pos":  nan3, "pre_close_ee_rpy":  nan3,
        "pre_close_obj_pos": nan3, "pre_close_obj_rpy": nan3,
        # Pre-lift state (after GRIP_STEPS, before LIFT_STEPS).
        "pre_lift_ee_pos":  nan3, "pre_lift_ee_rpy":  nan3,
        "pre_lift_obj_pos": nan3, "pre_lift_obj_rpy": nan3,
        # Post-lift state (after LIFT_STEPS).
        "post_lift_ee_pos":  nan3, "post_lift_ee_rpy":  nan3,
        "post_lift_obj_pos": nan3, "post_lift_obj_rpy": nan3,
        # Per-neighbor z rise (clearance-promoted extras), s.extra_objects order.
        "extra_lift_distances": [nan] * n_extra,
    }

def evaluate(s, joints_input, scene_poses=None, extra_scene_poses=None):
    """Evaluate each grasp in `joints_input` (N,7) end-to-end.

    By default every env shares the single built/settled target pose. If
    `scene_poses` is given (a list of robot-base-frame {"translation","rotation"}
    dicts, one per grasp row), each env instead places the target at its OWN
    pose — letting a batch of *different* candidate scenes run in parallel in a
    single rollout, with no per-pose re-exec. The grasp at row i is run against
    scene_poses[i].

    `extra_scene_poses` (optional, requires `scene_poses`) carries per-env poses
    for the clearance-promoted neighbors: a list parallel to the grasps, each a
    list of {"translation","rotation"} in s.extra_objects order. When given, the
    neighbors are placed per-env at their candidate pose (they ride the target's
    randomization as a rigid group); when omitted they reset to their fixed
    warmup-settled pose in every env.

    scene_poses batches get a short PRE-SETTLE (robot parked at home) before the
    snap-drift baseline is captured — raw requested poses can float a few mm or
    sit a hair inside the table, and comparing the post-snap position against
    the un-settled pose was producing false interpenetration_on_snap failures
    (gravity alone moves a free body ~3.4 cm in the 5 snap steps; capped
    depenetration moves it up to ~8 cm). Candidate poses should still come from
    a prior 'settle' request; the pre-settle only absorbs the residual.
    """
    sim, scene, robot, target, device = s.sim, s.scene, s.robot, s.target, s.device
    arm_jids, hand_idx, finger_jids = s.arm_jids, s.hand_idx, s.finger_jids
    n_total = len(joints_input)
    batch_size = s.batch_size
    run_results = []
    home_np = s.home_arm_pose.detach().cpu().numpy()
    n_batches = -(-n_total // batch_size)

    for bi, batch_idx in enumerate(range(0, n_total, batch_size)):
        current_batch_size = min(batch_size, n_total - batch_idx)
        print(f"[sim]   batch {bi + 1}/{n_batches} (grasps {batch_idx}..{batch_idx + current_batch_size - 1})", flush=True)

        # Per-env target placement. With `scene_poses`, each env in this chunk
        # gets its OWN object pose (parallel candidate scenes) instead of the
        # single built/settled pose. World pos = env origin +
        # robot_pose_to_env(pose); env frame carries the rotation directly
        # (env origins are pure translations). The snap-drift baseline is
        # captured AFTER the pre-settle below, not from these raw poses.
        env_origins = s.scene.env_origins
        if scene_poses is not None:
            batch_state = s.init_target_state.clone()
            for i in range(current_batch_size):
                pose = scene_poses[batch_idx + i]
                env_pos, env_rot = robot_pose_to_env(pose["translation"], pose["rotation"])
                env_pos, env_rot = _upright_env_pose(env_pos, env_rot, s.usd_path, s.mesh_format)
                world_pos = env_origins[i] + torch.tensor(env_pos, device=device, dtype=torch.float32)
                batch_state[i, 0:3]  = world_pos
                batch_state[i, 3:7]  = torch.tensor(env_rot, device=device, dtype=torch.float32)
                batch_state[i, 7:13] = 0.0
        else:
            batch_state   = s.init_target_state
            batch_settled = s.settled_pos_world

        # Reset object state for this batch. With `extra_scene_poses` each
        # neighbor is placed per-env at its candidate pose, so it rides the
        # target's randomization as a rigid group; otherwise it goes back to its
        # fixed warmup-settled pose in every env (a previous batch's grasp may
        # have shoved it).
        target.write_root_state_to_sim(batch_state)
        for j, ex in enumerate(s.extra_objects):
            if extra_scene_poses is not None:
                ex_state = ex["init_state"].clone()
                for i in range(current_batch_size):
                    pose = extra_scene_poses[batch_idx + i][j]
                    env_pos, env_rot = robot_pose_to_env(pose["translation"], pose["rotation"])
                    env_pos, env_rot = _upright_env_pose(env_pos, env_rot, ex["usd_path"],
                                                         ex["mesh_format"])
                    world_pos = env_origins[i] + torch.tensor(env_pos, device=device, dtype=torch.float32)
                    ex_state[i, 0:3]  = world_pos
                    ex_state[i, 3:7]  = torch.tensor(env_rot, device=device, dtype=torch.float32)
                    ex_state[i, 7:13] = 0.0
                ex["obj"].write_root_state_to_sim(ex_state)
            else:
                ex["obj"].write_root_state_to_sim(ex["init_state"])

        # ── Pre-settle (scene_poses only): park the robot at HOME so the arm
        #    can't be intersecting the freshly-teleported objects, let the
        #    requested poses come to rest, THEN capture the drift baseline. ────
        if scene_poses is not None:
            zeros_arm = torch.zeros_like(s.home_arm_pose)
            robot.write_joint_state_to_sim(s.home_arm_pose, zeros_arm, joint_ids=arm_jids)
            robot.write_joint_state_to_sim(s.open_grip_pos, torch.zeros_like(s.open_grip_pos),
                                           joint_ids=finger_jids)
            robot.reset()
            presettle_objs = [target] + [ex["obj"] for ex in s.extra_objects]

            def _presettle_tick():
                robot.set_joint_position_target(s.home_arm_pose, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=RENDER)
            settle_until_still(presettle_objs, _presettle_tick,
                               PRESETTLE_STEPS, PRESETTLE_MAX_STEPS, label="pre-settle")
            # Genuinely-at-rest per-env baseline for the interpenetration check.
            batch_settled = target.data.root_pos_w.detach().cpu().numpy().copy()

        batch_joints = np.zeros((batch_size, 7), dtype=np.float32)
        batch_joints[:current_batch_size] = joints_input[batch_idx:batch_idx + current_batch_size]

        finite_mask = np.isfinite(batch_joints[:current_batch_size]).all(axis=1)
        n_bad = int((~finite_mask).sum())
        if n_bad:
            print(f"[sim]     {n_bad} non-finite grasp(s) in this batch → returning NaNs", flush=True)
        for i in range(current_batch_size):
            if not finite_mask[i]:
                batch_joints[i] = home_np[i]
                run_results.append(_nan_result(batch_idx + i,
                                               n_extra=len(s.extra_objects)))

        grasp_q = torch.tensor(batch_joints, device=device, dtype=torch.float32)
        robot.write_joint_state_to_sim(grasp_q, torch.zeros_like(grasp_q), joint_ids=arm_jids)
        robot.write_joint_state_to_sim(s.open_grip_pos, torch.zeros_like(s.open_grip_pos), joint_ids=finger_jids)
        robot.reset()

        # Snap settle
        for _ in range(SNAP_STEPS):
            robot.set_joint_position_target(grasp_q, joint_ids=arm_jids)
            robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
            phys_tick(sim, scene, render=RENDER)

        post_snap_pos = target.data.root_pos_w.detach().cpu().numpy().copy()
        snap_drifts = np.linalg.norm(post_snap_pos - batch_settled, axis=1)

        active_envs = []
        interpenetrated = {}
        for i in range(current_batch_size):
            if not finite_mask[i]:
                continue
            interpenetrated[i] = bool(snap_drifts[i] > SNAP_DRIFT_TOL_M)
            active_envs.append(i)
        if not active_envs:
            continue

        # ── PRE-CLOSE CAPTURE ────────────────────────────────────────────────
        # Pose of the end-effector and the object the instant before we start
        # closing the gripper.
        pre_close_ee_pos   = robot.data.body_pos_w[:,  hand_idx].clone()
        pre_close_ee_quat  = robot.data.body_quat_w[:, hand_idx].clone()
        pre_close_obj_pos  = target.data.root_pos_w.clone()
        pre_close_obj_quat = target.data.root_quat_w.clone()
        # Neighbor heights at the same instant: baseline for the co-lift check.
        pre_close_extra_z = [ex["obj"].data.root_pos_w[:, 2].clone()
                             for ex in s.extra_objects]

        # Close gripper
        for _ in range(GRIP_STEPS):
            robot.set_joint_position_target(grasp_q, joint_ids=arm_jids)
            robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
            phys_tick(sim, scene, render=RENDER)

        nominal_obj_pos  = target.data.root_pos_w.clone()
        nominal_obj_quat = target.data.root_quat_w.clone()
        # Pre-lift EE pose: captured the instant the close finishes, BEFORE the
        # lift starts. This is the correct baseline for the robustness/slip term
        # so that the object re-seating that happens DURING the close isn't
        # counted as slip (that motion is largest for firm, successful grasps).
        pre_lift_ee_pos  = robot.data.body_pos_w[:,  hand_idx].clone()
        pre_lift_ee_quat = robot.data.body_quat_w[:, hand_idx].clone()
        closed_widths    = robot.data.joint_pos[:, finger_jids].sum(dim=1).clone()

        # Lift command — keyed off the INITIAL (pre-close) EE pose, not the
        # post-close pose. The gripper close can shove the EE around if it hits
        # the object; using pre_close_ee_pos keeps the lift target a fixed
        # reference relative to the proposed grasp pose.
        #
        # The z target is RAMPED up over the loop (not commanded to full height
        # up front), so the EE actually climbs slowly instead of snapping to the
        # top and dwelling. Ramp over the first LIFT_RAMP_FRAC of the steps, then
        # hold at full height for the remainder so the grasp can settle before
        # the post-lift capture.
        ramp_steps = max(1, int(LIFT_STEPS * LIFT_RAMP_FRAC))
        s.ik.reset()

        # Lift
        for step in range(LIFT_STEPS):
            frac = min(1.0, (step + 1) / ramp_steps)
            target_pos_w = pre_close_ee_pos.clone()
            target_pos_w[:, 2] += s.lift_height * frac
            target_pos_b, target_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, pre_close_ee_quat,
            )
            # position-only command; ee_quat is required by the API but only used
            # for display in position mode (compute() ignores it), so orientation is free.
            s.ik.set_command(target_pos_b, ee_quat=target_quat_b)

            ee_pos_w_now  = robot.data.body_pos_w[:,  hand_idx]
            ee_quat_w_now = robot.data.body_quat_w[:, hand_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w_now, ee_quat_w_now,
            )
            jacobian = robot.root_physx_view.get_jacobians()[:, s.ee_jacobi_idx, :, arm_jids]
            current_arm_q = robot.data.joint_pos[:, arm_jids]
            arm_q_desired = s.ik.compute(ee_pos_b, ee_quat_b, jacobian, current_arm_q)
            robot.set_joint_position_target(arm_q_desired, joint_ids=arm_jids)
            robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
            phys_tick(sim, scene, render=RENDER)

        post_lift_pos  = target.data.root_pos_w
        post_lift_quat = target.data.root_quat_w
        post_lift_extra_z = [ex["obj"].data.root_pos_w[:, 2]
                             for ex in s.extra_objects]
        # ── POST-LIFT EE CAPTURE ─────────────────────────────────────────────
        post_lift_ee_pos  = robot.data.body_pos_w[:,  hand_idx]
        post_lift_ee_quat = robot.data.body_quat_w[:, hand_idx]
        final_widths   = robot.data.joint_pos[:, finger_jids].sum(dim=1)

        for i in active_envs:
            global_gi = batch_idx + i
            q_nom_np  = nominal_obj_quat[i].cpu().numpy()
            q_lift_np = post_lift_quat[i].cpu().numpy()
            lift_dist_t  = post_lift_pos[i, 2] - nominal_obj_pos[i, 2]
            width_diff_t = final_widths[i] - closed_widths[i]

            # Pull the pre-close / post-lift snapshots for this env.
            pre_ee_pos_np   = pre_close_ee_pos[i].cpu().numpy()
            pre_ee_quat_np  = pre_close_ee_quat[i].cpu().numpy()
            pre_obj_pos_np  = pre_close_obj_pos[i].cpu().numpy()
            pre_obj_quat_np = pre_close_obj_quat[i].cpu().numpy()
            # Pre-lift snapshots (post-close, pre-lift) — slip/robustness baseline.
            pre_lift_ee_pos_np   = pre_lift_ee_pos[i].cpu().numpy()
            pre_lift_ee_quat_np  = pre_lift_ee_quat[i].cpu().numpy()
            pre_lift_obj_pos_np  = nominal_obj_pos[i].cpu().numpy()
            pre_lift_obj_quat_np = nominal_obj_quat[i].cpu().numpy()
            post_ee_pos_np  = post_lift_ee_pos[i].cpu().numpy()
            post_ee_quat_np = post_lift_ee_quat[i].cpu().numpy()
            post_obj_pos_np = post_lift_pos[i].cpu().numpy()
            # Per-neighbor z rise across the lift, s.extra_objects order.
            extra_lifts = [float((post_z[i] - pre_z[i]).item())
                           for pre_z, post_z in zip(pre_close_extra_z,
                                                    post_lift_extra_z)]

            # NaN-guard everything we're about to publish.
            if not (
                torch.isfinite(lift_dist_t) and torch.isfinite(width_diff_t)
                and torch.isfinite(closed_widths[i])
                and np.isfinite(q_nom_np).all() and np.isfinite(q_lift_np).all()
                and np.isfinite(pre_ee_pos_np).all()  and np.isfinite(pre_ee_quat_np).all()
                and np.isfinite(pre_obj_pos_np).all() and np.isfinite(pre_obj_quat_np).all()
                and np.isfinite(pre_lift_ee_pos_np).all()  and np.isfinite(pre_lift_ee_quat_np).all()
                and np.isfinite(pre_lift_obj_pos_np).all() and np.isfinite(pre_lift_obj_quat_np).all()
                and np.isfinite(post_ee_pos_np).all() and np.isfinite(post_ee_quat_np).all()
                and np.isfinite(post_obj_pos_np).all()
                and np.isfinite(extra_lifts).all()
            ):
                run_results.append(_nan_result(global_gi, reason="nan_in_sim",
                                               n_extra=len(s.extra_objects)))
                continue

            lift_dist = float(lift_dist_t.item())
            r_nom  = Rot.from_quat([q_nom_np[1],  q_nom_np[2],  q_nom_np[3],  q_nom_np[0]])
            r_lift = Rot.from_quat([q_lift_np[1], q_lift_np[2], q_lift_np[3], q_lift_np[0]])
            rpy_offsets = (r_lift * r_nom.inv()).as_euler("xyz", degrees=True)
            width_diff  = float(width_diff_t.item())

            # RPY in RADIANS for the snapshots.
            pre_ee_rpy   = _quat_wxyz_to_rpy_rad(pre_ee_quat_np)
            pre_obj_rpy  = _quat_wxyz_to_rpy_rad(pre_obj_quat_np)
            pre_lift_ee_rpy  = _quat_wxyz_to_rpy_rad(pre_lift_ee_quat_np)
            pre_lift_obj_rpy = _quat_wxyz_to_rpy_rad(pre_lift_obj_quat_np)
            post_ee_rpy  = _quat_wxyz_to_rpy_rad(post_ee_quat_np)
            post_obj_rpy = _quat_wxyz_to_rpy_rad(q_lift_np)  # same quat as post_lift_quat[i]

            res = {
                "index": global_gi,
                "passed": not interpenetrated[i],
                # Always present (None on success) so clients never need .get().
                # NOTE "passed" only encodes the snap-interpenetration gate, NOT
                # lift success — judge the lift from lift_distance etc.
                "fail_reason": "interpenetration_on_snap" if interpenetrated[i] else None,
                "lift_distance": round(lift_dist, 4),
                "roll_offset":  round(float(rpy_offsets[0]), 4),
                "pitch_offset": round(float(rpy_offsets[1]), 4),
                "yaw_offset":   round(float(rpy_offsets[2]), 4),
                "gripper_width_diff": round(width_diff, 5),
                "closed_gripper_width": round(float(closed_widths[i].item()), 5),
                # --- Pre-close (right before GRIP_STEPS) -------------------
                "pre_close_ee_pos":  [round(float(v), 4) for v in pre_ee_pos_np.tolist()],
                "pre_close_ee_rpy":  [round(float(v), 4) for v in pre_ee_rpy.tolist()],
                "pre_close_obj_pos": [round(float(v), 4) for v in pre_obj_pos_np.tolist()],
                "pre_close_obj_rpy": [round(float(v), 4) for v in pre_obj_rpy.tolist()],
                # --- Pre-lift (after GRIP_STEPS, before LIFT_STEPS) -------
                # Correct baseline for the slip/robustness term: object has
                # already re-seated into the closed gripper here.
                "pre_lift_ee_pos":  [round(float(v), 4) for v in pre_lift_ee_pos_np.tolist()],
                "pre_lift_ee_rpy":  [round(float(v), 4) for v in pre_lift_ee_rpy.tolist()],
                "pre_lift_obj_pos": [round(float(v), 4) for v in pre_lift_obj_pos_np.tolist()],
                "pre_lift_obj_rpy": [round(float(v), 4) for v in pre_lift_obj_rpy.tolist()],
                # --- Post-lift (after LIFT_STEPS) --------------------------
                "post_lift_ee_pos":  [round(float(v), 4) for v in post_ee_pos_np.tolist()],
                "post_lift_ee_rpy":  [round(float(v), 4) for v in post_ee_rpy.tolist()],
                "post_lift_obj_pos": [round(float(v), 4) for v in post_obj_pos_np.tolist()],
                "post_lift_obj_rpy": [round(float(v), 4) for v in post_obj_rpy.tolist()],
                # --- Neighbors (clearance-promoted extras) -----------------
                # z rise of each extra object across the lift (post-lift minus
                # pre-close), in s.extra_objects order: did this grasp hoist a
                # non-target object too?
                "extra_lift_distances": [round(v, 4) for v in extra_lifts],
            }
            run_results.append(res)

    run_results.sort(key=lambda x: x["index"])
    return run_results

# ==============================================================================
#  VISUALIZE: hold ONE joint configuration and render it (no grip-close, no lift)
# ==============================================================================
def visualize(s, joints_input, hold_steps):
    """Place the robot at a single joint configuration and just render it, so you
    can eyeball how the arm/gripper sits relative to the (settled) object.

    Differences vs evaluate():
      * Only the FIRST joint row of `joints_input` is used; any others are ignored.
      * The gripper stays OPEN — we never close it.
      * We never command a lift.
      * Rendering is forced ON regardless of the global --render flag (the whole
        point here is to look at it). Launch the app with a GUI / livestream for
        anything to actually appear on screen.

    NOTE: if the mesh was voxel-remeshed for SDF cooking, the RENDERED geometry
    is the remeshed (marching-cubes) surface, not the raw capture — expect a
    slightly blobbier look than the perception mesh.
    """
    sim, scene, robot, target = s.sim, s.scene, s.robot, s.target
    arm_jids, finger_jids, device = s.arm_jids, s.finger_jids, s.device

    # Take the single config and broadcast it across the whole batch so every
    # env shows the same pose. Sanitize a non-finite row to the home pose.
    joint_row = np.asarray(joints_input[0], dtype=np.float32)
    if not np.isfinite(joint_row).all():
        print("[viz] non-finite joint row → falling back to home pose.", flush=True)
        joint_row = s.home_arm_pose[0].detach().cpu().numpy()
    batch_joints = np.tile(joint_row, (s.batch_size, 1)).astype(np.float32)
    grasp_q = torch.tensor(batch_joints, device=device, dtype=torch.float32)

    # Reset object to its settled pose, drop the arm into the config, open grip.
    target.write_root_state_to_sim(s.init_target_state)
    for ex in s.extra_objects:
        ex["obj"].write_root_state_to_sim(ex["init_state"])
    robot.write_joint_state_to_sim(grasp_q, torch.zeros_like(grasp_q), joint_ids=arm_jids)
    robot.write_joint_state_to_sim(s.open_grip_pos, torch.zeros_like(s.open_grip_pos),
                                   joint_ids=finger_jids)
    robot.reset()

    print(f"[viz] Holding joint configuration for up to {hold_steps} render steps "
          f"(no grip-close, no lift)...", flush=True)
    steps_done = 0
    for _ in range(hold_steps):
        if not simulation_app.is_running():
            break  # window closed by the user → stop the hold early
        # PD-hold the requested arm pose and keep the fingers open.
        robot.set_joint_position_target(grasp_q, joint_ids=arm_jids)
        robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
        phys_tick(sim, scene, render=True)  # force render regardless of global RENDER
        steps_done += 1

    ee_pos_w  = robot.data.body_pos_w[:, s.hand_idx][0].detach().cpu().numpy().tolist()
    obj_pos_w = target.data.root_pos_w[0].detach().cpu().numpy().tolist()
    print("[viz] Done holding.", flush=True)
    return {
        "index": 0,
        "visualized": True,
        "steps_rendered": steps_done,
        "joints": [round(float(v), 6) for v in joint_row.tolist()],
        "ee_pos_world": [round(float(v), 4) for v in ee_pos_w],
        "object_pos_world": [round(float(v), 4) for v in obj_pos_w],
    }

# ==============================================================================
#  ROLLOUT: loop snap -> close -> lift -> hold -> reset, forever (until killed)
# ==============================================================================
def rollout(s, joints_input, hold_steps=60):
    """Continuously demo a grasp for live viewing (launch the server with --render).

    Loops until the process is killed (Ctrl-C) or the viewer window is closed:
        reset scene/physics -> snap arm to joints -> close gripper -> lift
        -> hold at the top for ~hold_steps -> repeat.

    Mirrors evaluate()'s motion exactly (snap, close, IK lift keyed off the
    pre-close EE pose). Rendering is forced ON every step so you can watch it.

    NOTE: while a rollout runs, the (single, synchronous REP) server can serve
    nothing else — any other client blocks for the rollout's entire lifetime.

    joints_input: (N,7) array-like; only the FIRST row is used (like visualize).
    A non-finite row falls back to the home pose.
    hold_steps:   60 Hz control ticks to hold at the top of the lift (~hold_steps/60 s).

    Never returns under normal use — it runs the loop until interrupted, then
    returns a small dict noting how many cycles completed.
    """
    sim, scene, robot, target = s.sim, s.scene, s.robot, s.target
    arm_jids, hand_idx, finger_jids, device = s.arm_jids, s.hand_idx, s.finger_jids, s.device

    # Take the first joint row and broadcast across the batch. Sanitize a
    # non-finite row to the home pose, same as visualize().
    joint_row = np.asarray(joints_input[0], dtype=np.float32)
    if not np.isfinite(joint_row).all():
        print("[rollout] non-finite joint row → falling back to home pose.", flush=True)
        joint_row = s.home_arm_pose[0].detach().cpu().numpy()
    batch_joints = np.tile(joint_row, (s.batch_size, 1)).astype(np.float32)
    grasp_q = torch.tensor(batch_joints, device=device, dtype=torch.float32)

    print("[rollout] Looping snap -> close -> lift -> hold -> reset. "
          "Ctrl-C (or close the viewer) to stop.", flush=True)
    cycles = 0
    try:
        while simulation_app.is_running():
            # ── Reset scene/physics: EVERY object back to its settled pose,
            #    arm to the grasp config, gripper open. Resetting all objects
            #    (not just the target) keeps the surrounding clutter from
            #    drifting away across cycles. ─────────────────────────────────
            if getattr(s, "init_states", None) is not None:
                for ro, st in zip(s.rigid_objects, s.init_states):
                    ro["obj"].write_root_state_to_sim(st)
            else:
                target.write_root_state_to_sim(s.init_target_state)
            robot.write_joint_state_to_sim(grasp_q, torch.zeros_like(grasp_q),
                                           joint_ids=arm_jids)
            robot.write_joint_state_to_sim(s.open_grip_pos,
                                           torch.zeros_like(s.open_grip_pos),
                                           joint_ids=finger_jids)
            robot.reset()
            scene.update(SIM_DT)

            # ── Snap settle ───────────────────────────────────────────────────
            for _ in range(SNAP_STEPS):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(grasp_q, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            # Capture the pre-close EE pose; the lift target keys off this.
            pre_close_ee_pos  = robot.data.body_pos_w[:, hand_idx].clone()
            pre_close_ee_quat = robot.data.body_quat_w[:, hand_idx].clone()

            # ── Close gripper ─────────────────────────────────────────────────
            for _ in range(GRIP_STEPS):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(grasp_q, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            # ── Lift, keyed off the pre-close EE pose (as in evaluate()) ──────
            # z target is RAMPED over LIFT_RAMP_FRAC of the steps so the EE
            # climbs slowly, then holds at the top for the remainder.
            ramp_steps = max(1, int(LIFT_STEPS * LIFT_RAMP_FRAC))
            s.ik.reset()

            for step in range(LIFT_STEPS):
                if not simulation_app.is_running():
                    break
                frac = min(1.0, (step + 1) / ramp_steps)
                target_pos_w = pre_close_ee_pos.clone()
                target_pos_w[:, 2] += s.lift_height * frac
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, pre_close_ee_quat)
                # position-only command; ee_quat is required by the API but only used
                # for display in position mode (compute() ignores it), so orientation is free.
                s.ik.set_command(target_pos_b, ee_quat=target_quat_b)

                ee_pos_w  = robot.data.body_pos_w[:, hand_idx]
                ee_quat_w = robot.data.body_quat_w[:, hand_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)
                jac = robot.root_physx_view.get_jacobians()[:, s.ee_jacobi_idx, :, arm_jids]
                q_des = s.ik.compute(ee_pos_b, ee_quat_b, jac, robot.data.joint_pos[:, arm_jids])
                robot.set_joint_position_target(q_des, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            # ── Hold at the top for ~hold_steps, keeping the last lift command ─
            held_arm_q = robot.data.joint_pos[:, arm_jids].clone()
            for _ in range(hold_steps):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(held_arm_q, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            cycles += 1
            print(f"[rollout] Completed cycle {cycles}.", flush=True)
    except KeyboardInterrupt:
        print("\n[rollout] Interrupted by user.", flush=True)

    return {"rollout": True, "cycles_completed": cycles}

# ==============================================================================
#  ROLLOUT_TRAJ: like rollout, but DRIVE the arm to the grasp along a planned
#  trajectory instead of teleporting it to the grasp config.
# ==============================================================================
def rollout_traj(s, traj, hold_steps=60, steps_per_waypoint=3, max_cycles=None):
    """Identical to rollout(), except the arm is driven to the grasp pose by
    PLAYING a planned joint trajectory `traj` (from iksolver.solve_traj) instead
    of snapping straight to a single grasp config.

    Each cycle loops until the process is killed (Ctrl-C) or the viewer is closed:
        reset scene/physics -> play `traj` waypoint-by-waypoint (gripper OPEN)
        -> close gripper -> lift -> hold at the top -> repeat.

    The close/lift/hold phase is byte-for-byte the same motion as rollout() and
    keys off the EE pose reached at the END of the trajectory (the final waypoint
    is the grasp config). Rendering is forced ON so you can watch it.

    NOTE: while a rollout runs, the (single, synchronous REP) server can serve
    nothing else — any other client blocks for the rollout's entire lifetime.

    traj:               (T,7) array-like of arm joint positions, one row per tick
                        (e.g. solve_traj's 20 Hz output). Non-finite rows are
                        dropped; if none remain it falls back to the home pose.
    hold_steps:         60 Hz control ticks to hold at the top of the lift (~hold_steps/60 s).
    steps_per_waypoint: 60 Hz ticks spent PD-tracking each waypoint (>=1). Larger =
                        slower, smoother playback (the arm has more time to reach
                        each waypoint before the next is commanded).
    """
    sim, scene, robot, target = s.sim, s.scene, s.robot, s.target
    arm_jids, hand_idx, finger_jids, device = s.arm_jids, s.hand_idx, s.finger_jids, s.device

    # Sanitize the trajectory: drop non-finite rows, keep (M,7). Fall back to the
    # home pose if nothing finite remains (same spirit as rollout's row guard).
    traj = np.asarray(traj, dtype=np.float32).reshape(-1, 7)
    finite = np.isfinite(traj).all(axis=1)
    traj = traj[finite]
    if traj.shape[0] < 1:
        print("[rollout_traj] no finite trajectory rows → falling back to home pose.", flush=True)
        traj = s.home_arm_pose[0].detach().cpu().numpy().reshape(1, 7)
    steps_per_waypoint = max(1, int(steps_per_waypoint))

    # Broadcast the start config and pre-tile every waypoint across the batch once.
    start_q    = torch.tensor(np.tile(traj[0], (s.batch_size, 1)),
                              device=device, dtype=torch.float32)
    waypoint_q = [torch.tensor(np.tile(traj[k], (s.batch_size, 1)),
                               device=device, dtype=torch.float32)
                  for k in range(traj.shape[0])]
    grasp_q = waypoint_q[-1]  # final waypoint = the grasp config (held during close/lift)

    print(f"[rollout_traj] Looping play-traj({traj.shape[0]} wpts ×{steps_per_waypoint} "
          f"steps) -> close -> lift -> hold -> reset. Ctrl-C (or close the viewer) "
          f"to stop.", flush=True)
    cycles = 0
    try:
        # max_cycles=None keeps the original loop-forever demo behavior;
        # a batch caller (main_batched) passes a bound so the run terminates.
        while simulation_app.is_running() and (max_cycles is None or cycles < max_cycles):
            # ── Reset: EVERY object back to its settled pose, arm at the
            #    trajectory START, gripper open. ─────────────────────────────
            if getattr(s, "init_states", None) is not None:
                for ro, st in zip(s.rigid_objects, s.init_states):
                    ro["obj"].write_root_state_to_sim(st)
            else:
                target.write_root_state_to_sim(s.init_target_state)
            robot.write_joint_state_to_sim(start_q, torch.zeros_like(start_q),
                                           joint_ids=arm_jids)
            robot.write_joint_state_to_sim(s.open_grip_pos,
                                           torch.zeros_like(s.open_grip_pos),
                                           joint_ids=finger_jids)
            robot.reset()
            scene.update(SIM_DT)

            # ── Quench the teleport (see RESET_SETTLE_STEPS): bleed with the
            #    arm held at the trajectory start, then hard-zero, so playback
            #    begins from genuine rest instead of a contact-kick rock. ─────
            quench_objs = ([ro["obj"] for ro in s.rigid_objects]
                           if getattr(s, "init_states", None) is not None
                           else [target])
            def _quench_tick():
                if not simulation_app.is_running():
                    return False
                robot.set_joint_position_target(start_q, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)
            settle_until_still(quench_objs, _quench_tick,
                               RESET_SETTLE_STEPS, RESET_SETTLE_MAX_STEPS, label="quench")

            # ── Play the trajectory: PD-track each waypoint with gripper open ─
            running = True
            for q_des in waypoint_q:
                if not running:
                    break
                for _ in range(steps_per_waypoint):
                    if not simulation_app.is_running():
                        running = False
                        break
                    robot.set_joint_position_target(q_des, joint_ids=arm_jids)
                    robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
                    phys_tick(sim, scene, render=True)
            if not running:
                break

            # Capture the pre-close EE pose at the END of the trajectory; the
            # lift target keys off this (exactly as rollout keys off its snap).
            pre_close_ee_pos  = robot.data.body_pos_w[:, hand_idx].clone()
            pre_close_ee_quat = robot.data.body_quat_w[:, hand_idx].clone()

            # ── Close gripper ─────────────────────────────────────────────────
            for _ in range(GRIP_STEPS):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(grasp_q, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            # ── Lift, keyed off the pre-close EE pose (as in rollout()) ───────
            ramp_steps = max(1, int(LIFT_STEPS * LIFT_RAMP_FRAC))
            s.ik.reset()

            for step in range(LIFT_STEPS):
                if not simulation_app.is_running():
                    break
                frac = min(1.0, (step + 1) / ramp_steps)
                target_pos_w = pre_close_ee_pos.clone()
                target_pos_w[:, 2] += s.lift_height * frac
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, pre_close_ee_quat)
                # position-only command; ee_quat is required by the API but only used
                # for display in position mode (compute() ignores it), so orientation is free.
                s.ik.set_command(target_pos_b, ee_quat=target_quat_b)

                ee_pos_w  = robot.data.body_pos_w[:, hand_idx]
                ee_quat_w = robot.data.body_quat_w[:, hand_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)
                jac = robot.root_physx_view.get_jacobians()[:, s.ee_jacobi_idx, :, arm_jids]
                q_des = s.ik.compute(ee_pos_b, ee_quat_b, jac, robot.data.joint_pos[:, arm_jids])
                robot.set_joint_position_target(q_des, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            # ── Hold at the top for ~hold_steps, keeping the last lift command ─
            held_arm_q = robot.data.joint_pos[:, arm_jids].clone()
            for _ in range(hold_steps):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(held_arm_q, joint_ids=arm_jids)
                robot.set_joint_effort_target(s.grip_close_eff, joint_ids=finger_jids)
                phys_tick(sim, scene, render=True)

            cycles += 1
            print(f"[rollout_traj] Completed cycle {cycles}.", flush=True)
    except KeyboardInterrupt:
        print("\n[rollout_traj] Interrupted by user.", flush=True)

    return {"rollout_traj": True, "cycles_completed": cycles}

# ==============================================================================
#  SETTLE: step physics with the robot held at home; return new robot-base poses
# ==============================================================================
def settle(s, settle_steps):
    """Step physics for `settle_steps` while the robot PD-holds the home pose,
    then read each object's final world pose and convert it back to the
    robot-base frame. Returns a list aligned with the build order:
        [{"name", "translation":[x,y,z], "rotation":[w,x,y,z], "scale"}, ...]
    translation/rotation are in the ROBOT-BASE frame (matching the request
    inputs).

    IDEMPOTENT: every call first resets all objects to their REQUESTED
    (build-time) poses and the robot to home, so a resent identical request
    (e.g. after a client timeout + retry) settles from the same starting point
    and returns the same answer, instead of compounding another settle_steps of
    physics on the already-settled scene."""
    if getattr(s, "kind", None) != "settle":
        raise RuntimeError("settle() called against a non-settle scene")

    sim, scene, robot = s.sim, s.scene, s.robot
    arm_jids, finger_jids = s.arm_jids, s.finger_jids

    # ── Reset to the requested (build-time) state ────────────────────────────
    for ro, st in zip(s.rigid_objects, s.init_states):
        ro["obj"].write_root_state_to_sim(st)
    zeros_arm = torch.zeros_like(s.home_arm_pose)
    robot.write_joint_state_to_sim(s.home_arm_pose, zeros_arm, joint_ids=arm_jids)
    robot.write_joint_state_to_sim(s.open_grip_pos, torch.zeros_like(s.open_grip_pos),
                                   joint_ids=finger_jids)
    robot.reset()

    # Make sure the asset data buffers are populated before reading (physics_settle.py
    # does the same with obj.update(0.0) right after sim.reset).
    scene.update(0.0)
    initial = {
        ro["name"]: ro["obj"].data.root_pos_w[0].detach().cpu().numpy().copy()
        for ro in s.rigid_objects
    }

    max_steps = max(settle_steps, WARMUP_MAX_STEPS)
    print(f"[settle] Stepping physics for {settle_steps}..{max_steps} steps "
          f"(~{settle_steps * SIM_DT:.1f}s minimum, until still)...", flush=True)
    settle_objs = [ro["obj"] for ro in s.rigid_objects]

    def _settle_tick():
        # PD-hold robot at home with fingers open so it doesn't slump into the table.
        robot.set_joint_position_target(s.home_arm_pose, joint_ids=arm_jids)
        robot.set_joint_effort_target(s.grip_open_eff, joint_ids=finger_jids)
        phys_tick(sim, scene, render=RENDER)
    settle_until_still(settle_objs, _settle_tick, settle_steps, max_steps,
                       label="settle", verbose=True)

    out = []
    print("[settle] Per-object delta (env-world frame):", flush=True)
    for ro in s.rigid_objects:
        name = ro["name"]
        obj  = ro["obj"]
        pos_w  = obj.data.root_pos_w[0].detach().cpu().numpy()
        quat_w = obj.data.root_quat_w[0].detach().cpu().numpy()  # wxyz
        d = pos_w - initial[name]
        print(f"  {name:<20} d=({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})  |d|={np.linalg.norm(d):.4f}",
              flush=True)
        trans, quat = env_pose_to_robot(tuple(pos_w.tolist()),
                                        tuple(quat_w.tolist()))
        out.append({
            "name":        name,
            "translation": [round(float(v), 6) for v in trans],
            "rotation":    [round(float(v), 6) for v in quat],
            "scale":       ro["scale"],
        })
    return out

# ==============================================================================
#  REQUEST VALIDATION + SCENE IDENTITY
# ==============================================================================
def _mesh_bytes_loadable(mesh_bytes, mesh_format):
    """Actually LOAD the mesh bytes (trimesh) and check for geometry.

    This runs BEFORE a request is allowed to trigger a scene-switch re-exec.
    Without it, a corrupt/empty mesh sails through the cheap dict checks, the
    server re-execs, and the NEW process dies inside the world build — a
    permanent outage with the client stuck retrying against a dead endpoint.
    Costs up to ~a second on a 500k-face capture; trivially cheap next to the
    30-60 s Isaac restart it prevents wasting."""
    try:
        import trimesh
        fmt = str(mesh_format).lstrip(".").lower()
        m = trimesh.load(io.BytesIO(bytes(mesh_bytes)), file_type=fmt, force="mesh")
        if m is None or getattr(m, "faces", None) is None or len(m.faces) == 0:
            return False, "mesh loaded but contains no faces"
        if not np.isfinite(np.asarray(m.vertices)).all():
            return False, "mesh contains non-finite vertices"
        return True, None
    except Exception as e:
        return False, f"mesh bytes not loadable as {mesh_format!r}: {e}"

def _validate_object_entry(o, where):
    """Validate one object dict from a settle request (including mesh loadability)."""
    if not isinstance(o, dict):
        return False, f"{where} is not a dict"
    name = o.get("name")
    if not name or not isinstance(name, str):
        return False, f"{where}: missing/invalid 'name'"
    mb = o.get("mesh_bytes")
    if not isinstance(mb, (bytes, bytearray, memoryview)) or len(mb) == 0:
        return False, f"{where}: 'mesh_bytes' must be non-empty bytes"
    if not o.get("mesh_format"):
        return False, f"{where}: 'mesh_format' required"
    ok, err = _mesh_bytes_loadable(mb, o["mesh_format"])
    if not ok:
        return False, f"{where}: {err}"
    tr = o.get("translation")
    if tr is None or len(tr) != 3:
        return False, f"{where}: 'translation' must be length-3"
    rr = o.get("rotation")
    if rr is None or len(rr) != 4:
        return False, f"{where}: 'rotation' must be length-4 (wxyz)"
    if o.get("scale") is None:
        return False, f"{where}: missing 'scale'"
    return True, None

def _validate_extra_objects(req, target):
    """Optional 'extra_objects' list on the grasp modes (try/visualize/load):
    settle-style entries for the clearance-promoted neighbors spawned beside
    the target. Absent/None is fine (single-object world, the original shape)."""
    xo = req.get("extra_objects")
    if xo is None:
        return True, None
    if not isinstance(xo, list):
        return False, "'extra_objects' must be a list when provided"
    names = []
    for i, o in enumerate(xo):
        ok, err = _validate_object_entry(o, f"extra_objects[{i}]")
        if not ok:
            return False, err
        names.append(o["name"])
    if len(set(names)) != len(names):
        return False, "'extra_objects' names must be unique"
    if target in names:
        return False, f"'extra_objects' must not contain the target {target!r}"
    return True, None

def validate_request(req):
    """Return (ok, error_str). Runs BEFORE any sim work / re-exec. Includes an
    actual mesh-load check so an unbuildable scene is rejected HERE, in the
    still-alive process, instead of killing the freshly re-exec'd one."""
    if not isinstance(req, dict):
        return False, "request is not a dict"

    mode = req.get("mode", "try")
    if mode not in ("try", "visualize", "settle", "rollout", "rollout_traj", "load"):
        return False, (f"'mode' must be 'try', 'visualize', 'settle', 'rollout', "
                       f"'rollout_traj', or 'load', got {mode!r}")

    # ── load mode: pre-build the grasp world (same scene shape/identity as
    #    'try') and keep it hot. No joints — the client fires this right after
    #    settle so the eval world builds while the VLM stages run, and the
    #    later 'try' request lands on a live scene instead of re-exec'ing. ────
    if mode == "load":
        t = req.get("target")
        if not t or not isinstance(t, str):
            return False, "load: missing/invalid 'target' (str)"
        mb = req.get("mesh_bytes")
        if not isinstance(mb, (bytes, bytearray, memoryview)) or len(mb) == 0:
            return False, "load: 'mesh_bytes' must be non-empty bytes"
        if not req.get("mesh_format"):
            return False, "load: 'mesh_format' required (e.g. 'glb')"
        ok, err = _mesh_bytes_loadable(mb, req["mesh_format"])
        if not ok:
            return False, err
        tr = req.get("translation")
        if tr is None or len(tr) != 3:
            return False, "load: 'translation' must be length-3"
        rr = req.get("rotation")
        if rr is None or len(rr) != 4:
            return False, "load: 'rotation' must be length-4 (wxyz)"
        if req.get("scale") is None:
            return False, "load: missing 'scale'"
        return _validate_extra_objects(req, t)

    # ── Settle mode: list of objects, no target/joints. ──────────────────────
    if mode == "settle":
        objs = req.get("objects")
        if not isinstance(objs, list) or len(objs) == 0:
            return False, "'settle' mode requires a non-empty 'objects' list"
        names = []
        for i, o in enumerate(objs):
            ok, err = _validate_object_entry(o, f"objects[{i}]")
            if not ok:
                return False, err
            names.append(o["name"])
        if len(set(names)) != len(names):
            return False, "'objects' names must be unique"
        ss = req.get("settle_steps")
        if ss is not None:
            try:
                if int(ss) <= 0:
                    return False, "'settle_steps' must be a positive integer"
            except Exception:
                return False, "'settle_steps' must be an integer"
        return True, None

    # ── rollout mode: full scene ('objects' list) + which one is the grasp
    #    'target' + (N,7) joints (row 0 used). ──────────────────────────────
    if mode == "rollout":
        t = req.get("target")
        if not t or not isinstance(t, str):
            return False, "rollout: missing/invalid 'target' (str)"
        objs = req.get("objects")
        if not isinstance(objs, list) or len(objs) == 0:
            return False, "rollout: requires a non-empty 'objects' list"
        names = []
        for i, o in enumerate(objs):
            ok, err = _validate_object_entry(o, f"objects[{i}]")
            if not ok:
                return False, err
            names.append(o["name"])
        if len(set(names)) != len(names):
            return False, "rollout: 'objects' names must be unique"
        if t not in names:
            return False, f"rollout: target {t!r} not among objects names {names}"
        try:
            ja = np.asarray(req.get("joints"), dtype=np.float32)
        except Exception as e:
            return False, f"rollout: 'joints' not array-like: {e}"
        if ja.ndim != 2 or ja.shape[1] != 7:
            return False, f"rollout: 'joints' must be (N,7), got {ja.shape}"
        if ja.shape[0] < 1:
            return False, "rollout: needs at least one joint row"
        return True, None

    # ── rollout_traj mode: full scene ('objects') + grasp 'target' + a planned
    #    (T,7) joint 'traj' to play instead of snapping to a config. ──────────
    if mode == "rollout_traj":
        t = req.get("target")
        if not t or not isinstance(t, str):
            return False, "rollout_traj: missing/invalid 'target' (str)"
        objs = req.get("objects")
        if not isinstance(objs, list) or len(objs) == 0:
            return False, "rollout_traj: requires a non-empty 'objects' list"
        names = []
        for i, o in enumerate(objs):
            ok, err = _validate_object_entry(o, f"objects[{i}]")
            if not ok:
                return False, err
            names.append(o["name"])
        if len(set(names)) != len(names):
            return False, "rollout_traj: 'objects' names must be unique"
        if t not in names:
            return False, f"rollout_traj: target {t!r} not among objects names {names}"
        try:
            tj = np.asarray(req.get("traj"), dtype=np.float32)
        except Exception as e:
            return False, f"rollout_traj: 'traj' not array-like: {e}"
        if tj.ndim != 2 or tj.shape[1] != 7:
            return False, f"rollout_traj: 'traj' must be (T,7), got {tj.shape}"
        if tj.shape[0] < 1:
            return False, "rollout_traj: needs at least one trajectory row"
        return True, None

    # ── Grasp modes ('try' / 'visualize'): single target + joints. ───────────
    t = req.get("target")
    if not t or not isinstance(t, str):
        return False, "missing/invalid 'target' (str)"
    mb = req.get("mesh_bytes")
    if not isinstance(mb, (bytes, bytearray, memoryview)) or len(mb) == 0:
        return False, "'mesh_bytes' must be non-empty bytes (mesh is sent in-request)"
    if not req.get("mesh_format"):
        return False, "'mesh_format' required (e.g. 'glb')"
    ok, err = _mesh_bytes_loadable(mb, req["mesh_format"])
    if not ok:
        return False, err
    tr = req.get("translation")
    if tr is None or len(tr) != 3:
        return False, "'translation' must be length-3"
    rr = req.get("rotation")
    if rr is None or len(rr) != 4:
        return False, "'rotation' must be length-4 (wxyz)"
    if req.get("scale") is None:
        return False, "missing 'scale'"
    try:
        ja = np.asarray(req.get("joints"), dtype=np.float32)
    except Exception as e:
        return False, f"'joints' not array-like: {e}"
    if ja.ndim != 2 or ja.shape[1] != 7:
        return False, f"'joints' must be (N,7), got {ja.shape}"
    if ja.shape[0] < 1:
        return False, "'joints' must have at least one row"
    ok, err = _validate_extra_objects(req, t)
    if not ok:
        return False, err
    # Optional per-env target poses (parallel candidate scenes): one per grasp.
    sp = req.get("scene_poses")
    if sp is not None:
        if not isinstance(sp, list) or len(sp) == 0:
            return False, "'scene_poses' must be a non-empty list when provided"
        if len(sp) != ja.shape[0]:
            return False, (f"'scene_poses' length ({len(sp)}) must match "
                           f"joints rows ({ja.shape[0]})")
        for i, p in enumerate(sp):
            if not isinstance(p, dict):
                return False, f"scene_poses[{i}] must be a dict"
            if p.get("translation") is None or len(p["translation"]) != 3:
                return False, f"scene_poses[{i}]['translation'] must be length-3"
            if p.get("rotation") is None or len(p["rotation"]) != 4:
                return False, f"scene_poses[{i}]['rotation'] must be length-4 (wxyz)"
    # Optional per-env neighbor poses (extras riding the target's randomization):
    # one list per grasp, each holding one pose per extra object.
    esp = req.get("extra_scene_poses")
    if esp is not None:
        if sp is None:
            return False, "'extra_scene_poses' requires 'scene_poses'"
        n_extra = len(req.get("extra_objects") or [])
        if not isinstance(esp, list) or len(esp) != ja.shape[0]:
            return False, (f"'extra_scene_poses' must be a list of length "
                           f"{ja.shape[0]} (one per grasp)")
        for i, row in enumerate(esp):
            if not isinstance(row, list) or len(row) != n_extra:
                return False, (f"extra_scene_poses[{i}] must be a list of "
                               f"{n_extra} pose(s) (one per extra object)")
            for j, p in enumerate(row):
                if not isinstance(p, dict):
                    return False, f"extra_scene_poses[{i}][{j}] must be a dict"
                if p.get("translation") is None or len(p["translation"]) != 3:
                    return False, f"extra_scene_poses[{i}][{j}]['translation'] must be length-3"
                if p.get("rotation") is None or len(p["rotation"]) != 4:
                    return False, f"extra_scene_poses[{i}][{j}]['rotation'] must be length-4 (wxyz)"
    return True, None

def _batch_for_mode(mode, cli_batch_size):
    """rollout / rollout_traj worlds are always single-env (they demo one config);
    everything else uses the CLI batch. Used consistently for BOTH building and
    identity so a rollout request never re-execs over a batch-size mismatch."""
    return ROLLOUT_BATCH_SIZE if mode in ("rollout", "rollout_traj") else int(cli_batch_size)

def scene_identity(req, batch_size):
    """The tuple of inputs that define the built scene. Changing any of these
    requires a fresh process (re-exec); 'joints', 'mode' (within a kind) and
    'settle_steps' are NOT part of identity. The mesh is keyed
    by a hash of its bytes — identical geometry reuses the live scene.

    NOTE: 'try', 'visualize' and 'load' share the single-target grasp scene
    shape, so they produce the SAME identity ("kind":"grasp") and switching
    between them against the same target does NOT force a rebuild — that
    sharing is exactly what lets a 'load' request pre-build the world a later
    'try' lands on. 'rollout' and
    'rollout_traj' both build the FULL multi-object scene (every mesh spawned),
    so they share their own identity ("kind":"rollout") keyed on the whole object
    list plus the grasp target — switching between those two on the same scene
    does NOT force a rebuild either.

    Callers must pass batch_size from _batch_for_mode(mode, cli_batch)."""
    def _identity_objects(objs):
        return [{
            "name":        str(o["name"]),
            "mesh":        hashlib.sha256(bytes(o["mesh_bytes"])).hexdigest(),
            "mesh_format": str(o.get("mesh_format", "")).lstrip(".").lower(),
            "translation": [float(x) for x in o["translation"]],
            "rotation":    [float(x) for x in o["rotation"]],
            "scale":       float(o["scale"]),
        } for o in sorted(objs, key=lambda o: o["name"])]

    mode = req.get("mode", "try")
    if mode == "settle":
        return {
            "kind": "settle",
            "objects": _identity_objects(req["objects"]),
        }
    if mode in ("rollout", "rollout_traj"):
        return {
            "kind":   "rollout",
            "target": str(req["target"]),
            "batch":  int(batch_size),
            "objects": _identity_objects(req["objects"]),
        }
    return {
        "kind":        "grasp",
        "target":      str(req["target"]),
        "mesh":        hashlib.sha256(bytes(req["mesh_bytes"])).hexdigest(),
        "mesh_format": str(req.get("mesh_format", "")).lstrip(".").lower(),
        "translation": [float(x) for x in req["translation"]],
        "rotation":    [float(x) for x in req["rotation"]],
        "scale":       float(req["scale"]),
        "batch":       int(batch_size),
        # Absent and [] are the same scene (the plain single-object world).
        "extra_objects": _identity_objects(req.get("extra_objects") or []),
    }

def _object_lists_match(oa, ob, tol):
    """Compare two normalized identity object lists (name-sorted, mesh-hashed)."""
    if len(oa) != len(ob):
        return False
    for ea, eb in zip(oa, ob):
        if (ea["name"] != eb["name"] or ea["mesh"] != eb["mesh"]
                or ea["mesh_format"] != eb["mesh_format"]):
            return False
        if abs(ea["scale"] - eb["scale"]) > tol:
            return False
        if not np.allclose(ea["translation"], eb["translation"], atol=tol, rtol=0):
            return False
        if not np.allclose(ea["rotation"], eb["rotation"], atol=tol, rtol=0):
            return False
    return True

def same_identity(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    if a.get("kind") != b.get("kind"):
        return False
    if a["kind"] in ("settle", "rollout"):
        # rollout also pins the grasp target + batch size on top of the object set.
        if a["kind"] == "rollout":
            if a.get("target") != b.get("target") or a.get("batch") != b.get("batch"):
                return False
        return _object_lists_match(a["objects"], b["objects"], tol)
    # grasp kind
    if (a["target"] != b["target"] or a["mesh"] != b["mesh"]
            or a["mesh_format"] != b["mesh_format"] or a["batch"] != b["batch"]):
        return False
    if not _object_lists_match(a.get("extra_objects", []), b.get("extra_objects", []), tol):
        return False
    return (abs(a["scale"] - b["scale"]) <= tol
            and np.allclose(a["translation"], b["translation"], atol=tol, rtol=0)
            and np.allclose(a["rotation"], b["rotation"], atol=tol, rtol=0))

# ==============================================================================
#  PICKLE WIRE HELPERS
# ==============================================================================
def _send(socket, obj):
    """REP requires exactly one send per recv. Always send a pickled dict."""
    socket.send(pickle.dumps(obj))

# ==============================================================================
#  SCENE-CHANGE: clean re-exec (fresh process image -> fresh CUDA/PhysX)
# ==============================================================================
def _argv_with_bootstrap(boot_path):
    """Rebuild argv, stripping any existing --bootstrap and baking in the new one."""
    argv, out, i = sys.argv[:], [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--bootstrap":
            i += 2; continue
        if a.startswith("--bootstrap="):
            i += 1; continue
        out.append(a); i += 1
    return out + ["--bootstrap", boot_path]

def restart_with_request(socket, context, request):
    """Persist the scene-defining request and re-exec so the new process builds
    it at startup. The client should reconnect and resend the same request."""
    fd, boot_path = tempfile.mkstemp(suffix=".pkl", prefix="grasp_bootstrap_")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(request, f)

    print(f"[server] Scene inputs changed. Re-exec with bootstrap {boot_path}", flush=True)
    try:
        reply = {"status": "restarting", "retry": True, "mode": request.get("mode", "try")}
        if request.get("mode") == "settle":
            reply["object_names"] = [o["name"] for o in request.get("objects", [])]
        else:
            reply["target"] = request.get("target")
        _send(socket, reply)
    except Exception:
        pass
    try:
        # zmq sends are ASYNC: the reply above is only QUEUED at this point.
        # close(linger=0) would DISCARD it if the peer hasn't drained it yet,
        # leaving the client blocked forever on recv() while the new process
        # rebuilds. A positive linger makes close()/term() flush (or wait up to
        # the linger) before we execv away. 5 s is far beyond any ipc latency.
        socket.setsockopt(zmq.LINGER, 5000)
        socket.close()
        context.term()   # blocks until pending messages flush or linger expires
    except Exception:
        pass
    sys.stdout.flush(); sys.stderr.flush()
    os.execv(sys.executable, [sys.executable] + _argv_with_bootstrap(boot_path))

# ==============================================================================
#  STARTUP BOOTSTRAP
# ==============================================================================
def load_bootstrap():
    if not args_cli.bootstrap:
        return None
    if not os.path.isfile(args_cli.bootstrap):
        print(f"[server] --bootstrap path not found: {args_cli.bootstrap}", flush=True)
        return None
    with open(args_cli.bootstrap, "rb") as f:
        return pickle.load(f)

def _remove_bootstrap_file():
    """Delete the consumed bootstrap pickle (it holds full mesh bytes) — on
    success AND on failure, so failed builds don't leak payloads into /tmp."""
    if args_cli.bootstrap:
        try:
            os.remove(args_cli.bootstrap)
        except Exception:
            pass

# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    batch_size  = int(args_cli.num_envs)
    lift_height = float(args_cli.lift_height)

    s = None
    built_identity = None
    # If the startup build FAILS, we do NOT exit: we remember what failed and
    # keep serving so clients get a clear error (or a re-exec toward a
    # DIFFERENT, buildable scene) instead of timing out against a dead process.
    failed_identity = None
    build_error = None

    # Build the scene ONCE at startup if we were handed bootstrap inputs.
    boot = load_bootstrap()
    if boot is not None:
        ok, err = validate_request(boot)
        if not ok:
            # An invalid bootstrap is served cold: the same request arriving
            # over the socket will fail validate_request there too (error
            # reply, no re-exec), so this cannot loop.
            print(f"[server] WARNING: bad bootstrap request ({err}); "
                  f"starting cold with no scene.", flush=True)
            _remove_bootstrap_file()
        else:
            boot_mode = boot.get("mode", "try")
            boot_batch = _batch_for_mode(boot_mode, batch_size)
            try:
                if boot_mode == "settle":
                    print(f"[server] Building SETTLE world for {len(boot['objects'])} object(s)...",
                          flush=True)
                    s = build_settle_world(boot["objects"])
                elif boot_mode in ("rollout", "rollout_traj"):
                    print(f"[server] Building ROLLOUT world ({boot_mode}) for "
                          f"{len(boot['objects'])} object(s), grasp target "
                          f"'{boot['target']}' (batch_size={boot_batch})...", flush=True)
                    s = build_rollout_world(boot["objects"], boot["target"],
                                            boot_batch, lift_height)
                else:
                    # 'try' / 'visualize' / 'load' use the grasp world: the
                    # target plus any clearance-promoted extra objects.
                    n_extra = len(boot.get("extra_objects") or [])
                    print(f"[server] Building world for '{boot['target']}' "
                          f"(batch_size={boot_batch}, mode={boot_mode}, "
                          f"extra_objects={n_extra})...", flush=True)
                    s = build_world(boot["target"], boot["mesh_bytes"], boot["mesh_format"],
                                    boot["translation"], boot["rotation"], boot["scale"],
                                    boot_batch, lift_height,
                                    extra_objects=boot.get("extra_objects"))
                built_identity = scene_identity(boot, boot_batch)
            except Exception as e:
                # Don't die — remember the failure and serve error replies for
                # this identity. A request for anything ELSE still re-execs.
                build_error = f"{type(e).__name__}: {e}"
                try:
                    failed_identity = scene_identity(boot, boot_batch)
                except Exception:
                    failed_identity = None
                s = None
                built_identity = None
                print(f"[server] BUILD FAILED ({build_error}). Staying alive to "
                      f"report the error to clients; a request for a different "
                      f"scene will re-exec normally.", flush=True)
            finally:
                _remove_bootstrap_file()
    else:
        print("[server] Cold start: no scene yet. The first request will trigger a build "
              "(via re-exec).", flush=True)

    from semantic_grasp.config import GRASP, bind
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # ipc:// — co-located, no network. pickle.loads on a hostile payload is RCE,
    # and a Unix socket is reachable only by local users (was 127.0.0.1 TCP).
    bind(socket, GRASP)
    print(f"[ZMQ] Server listening on {GRASP}.", flush=True)

    while True:
        raw = socket.recv()

        # REP requires exactly one send per recv — never `continue` without sending.
        if not raw:
            _send(socket, {"error": "empty payload"})
            continue

        print("\n[ZMQ] Request received.", flush=True)
        try:
            try:
                request = pickle.loads(raw)
            except Exception as unpickle_err:
                _send(socket, {"error": f"could not unpickle request: {unpickle_err}"})
                continue

            ok, err = validate_request(request)
            if not ok:
                _send(socket, {"error": f"invalid request: {err}"})
                continue

            mode = request.get("mode", "try")
            req_identity = scene_identity(request, _batch_for_mode(mode, batch_size))

            # This exact scene already failed to build in this process — reply
            # with the recorded error rather than re-exec'ing into the same
            # failure again (and again, until the client's max_restarts trips).
            if failed_identity is not None and same_identity(req_identity, failed_identity):
                _send(socket, {"error": f"scene build failed: {build_error}. "
                                        f"Fix the inputs and resend (a different "
                                        f"scene will build normally)."})
                continue

            # If the request needs a different scene than the live one (or there
            # is no live scene yet), re-exec into a fresh, fully-built process.
            # We NEVER build live inside this loop — that is what hangs.
            if not same_identity(req_identity, built_identity):
                restart_with_request(socket, context, request)
                return  # not reached; execv replaces the process

            if mode == "settle":
                settle_steps = int(request.get("settle_steps", SETTLE_STEPS))
                print(f"[sim] SETTLE mode — {len(request['objects'])} object(s), "
                      f"{settle_steps} steps.", flush=True)
                settle_results = settle(s, settle_steps)
                _send(socket, {"results": settle_results, "mode": "settle"})
                print(f"[ZMQ] Done. Returned {len(settle_results)} settled poses to client.",
                      flush=True)
            elif mode == "visualize":
                # Show ONE joint configuration (the first row) and just render it.
                joints_input = np.asarray(request["joints"], dtype=np.float32)
                hold_steps = int(request.get("viz_steps", args_cli.viz_steps))
                print(f"[sim] VISUALIZE mode for '{request['target']}' — showing 1 "
                      f"joint configuration (no lift).", flush=True)
                viz_result = visualize(s, joints_input, hold_steps)
                _send(socket, {"results": [viz_result], "mode": "visualize"})
                print("[ZMQ] Done. Visualization complete.", flush=True)
            elif mode == "rollout":
                # Loop snap -> close -> lift -> hold -> reset until killed.
                # Launch the server with --render to watch it live.
                # NOTE: this call monopolizes the (single, synchronous REP)
                # server for its whole lifetime — any other client blocks.
                joints_input = np.asarray(request["joints"], dtype=np.float32)
                hold_steps = int(request.get("hold_steps", 60))
                print(f"[sim] ROLLOUT mode for '{request['target']}' — looping until killed.",
                      flush=True)
                roll_result = rollout(s, joints_input, hold_steps)
                _send(socket, {"results": [roll_result], "mode": "rollout"})
                print("[ZMQ] Done. Rollout stopped.", flush=True)
            elif mode == "load":
                # Pre-build acknowledgement. Reaching dispatch means the scene
                # identity matched the live scene — i.e. the grasp world is
                # built and hot (either this request just triggered the re-exec
                # that built it, or it was already live). Nothing to simulate.
                print(f"[sim] LOAD mode — grasp world for '{request['target']}' "
                      f"is built and hot.", flush=True)
                _send(socket, {"status": "ready", "mode": "load",
                               "target": request["target"]})
            elif mode == "rollout_traj":
                # Like rollout, but PLAY a planned trajectory to the grasp instead
                # of snapping to it. Loops until killed; --render to watch live.
                traj = np.asarray(request["traj"], dtype=np.float32)
                hold_steps = int(request.get("hold_steps", 60))
                steps_per_waypoint = int(request.get("steps_per_waypoint", 3))
                max_cycles = request.get("max_cycles")
                max_cycles = None if max_cycles is None else int(max_cycles)
                print(f"[sim] ROLLOUT_TRAJ mode for '{request['target']}' — playing a "
                      f"{traj.shape[0]}-waypoint trajectory, "
                      f"{'looping until killed' if max_cycles is None else f'{max_cycles} cycle(s)'}.",
                      flush=True)
                roll_result = rollout_traj(s, traj, hold_steps, steps_per_waypoint, max_cycles)
                _send(socket, {"results": [roll_result], "mode": "rollout_traj"})
                print("[ZMQ] Done. Rollout (traj) stopped.", flush=True)
            else:
                # "try" — evaluate every grasp end-to-end. With per-env
                # 'scene_poses' (one pose per grasp row), each grasp runs against
                # its own target pose — parallel candidate scenes in one rollout.
                joints_input = np.asarray(request["joints"], dtype=np.float32)
                scene_poses = request.get("scene_poses")
                extra_scene_poses = request.get("extra_scene_poses")
                tag = (f"{len(joints_input)} grasps across {len(scene_poses)} scenes"
                       if scene_poses is not None else f"{len(joints_input)} grasps")
                if s.extra_objects:
                    moving = " moving w/ target" if extra_scene_poses is not None else " fixed"
                    tag += f" (+{len(s.extra_objects)} extra object(s),{moving})"
                print(f"[sim] Evaluating {tag} for '{request['target']}'...", flush=True)
                run_results = evaluate(s, joints_input, scene_poses, extra_scene_poses)
                _send(socket, {"results": run_results})
                print(f"[ZMQ] Done. Returned {len(run_results)} results to client.", flush=True)

        except Exception as runtime_err:
            # Safety net: always reply so the REP socket stays usable.
            print(f"[CRITICAL ERROR] {runtime_err}", flush=True)
            try:
                _send(socket, {"error": f"simulation error: {runtime_err}"})
            except Exception:
                pass

if __name__ == "__main__":
    main()