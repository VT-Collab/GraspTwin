"""Pipeline configuration: constants, paths, environment variables."""

import math
import os
import shutil
from pathlib import Path

# ── API keys ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta"
VLM_MODEL      = "gemini-robotics-er-2-preview"


# Debug/log output. main.py and main_batched.py wipe it at the start of a run
# (reset_output_dir); everyone else just needs it to exist.
OUTPUT_DIR = Path("pipeline_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def reset_output_dir():
    """Fresh OUTPUT_DIR for a new pipeline run."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


# ── ZMQ endpoints ─────────────────────────────────────────────────
# Single source of truth. Everything runs co-located on one host, so services
# talk over ipc:// (Unix domain sockets) — no ports, no TCP, no SSH tunnels.
# Servers bind() these, clients connect() to the same string. To put one
# service back on TCP, edit its line to e.g. "tcp://127.0.0.1:5556".
_IPC_DIR = "/tmp/semantic_grasp"
os.makedirs(_IPC_DIR, exist_ok=True)

PERCEPTION = f"ipc://{_IPC_DIR}/perception.sock"   # segmentation server   (was tcp 5555)
# GRASP      = f"ipc://{_IPC_DIR}/grasp.sock"        # isaac grasp / physics (was tcp 5556)
GRASP      = "tcp://127.0.0.1:8083"                 # isaac grasp / physics
RENDER     = f"ipc://{_IPC_DIR}/render.sock"       # isaac render / teleop (was tcp 5558)
IK         = "tcp://127.0.0.1:8065"                # curobo IK / trajectory (was ipc ik.sock; before that tcp 5557/5561)
# Camera runs on the lab PC (the one machine off-cluster with the Orbbec), so it
# stays on TCP via a lab-PC reverse tunnel (see RUNNING.md). The two ends use
# DIFFERENT ports on purpose: DCGM's nv-hostengine owns 127.0.0.1:5555 on the
# cluster nodes (the ssh -R can only grab ::1, which IPv4-default ZMQ never
# dials), so the node-side doorway is 5591 and the tunnel remaps
# (-R 5591:localhost:5555). The lab PC keeps binding 5555 locally.
CAMERA          = "tcp://localhost:5591"   # node side: orbbec capture via ssh -R
CAMERA_LAB_BIND = "tcp://localhost:5555"   # lab side: camera_server's bind default

# Wrist-mounted Intel RealSense D435 (servers/camera_intel.py). Separate ports
# from the Orbbec so both can serve at once: the fixed Orbbec sees the whole
# table, the wrist D435 is what scans for the LERF. Same tunnel shape as above,
# shifted by one:  ssh -R 5592:localhost:5556 <node>
CAMERA_INTEL          = "tcp://localhost:5592"   # node side: D435 capture via ssh -R
CAMERA_INTEL_LAB_BIND = "tcp://localhost:5556"   # lab side: camera_intel's bind default

# Trajectory replay on the real arm (servers/robot_server.py). Runs on the LAB PC
# for one reason: the streaming replay holds a 50 Hz command clock, and a round
# trip from a cluster node through the jump host cannot fit in 20 ms, so the arm
# gets its velocities late and irregularly (RUNNING.md "Latency warning"). With
# the server lab-side the control loop is a LAN hop to the NUC and the tunnel
# carries one bulk message per grasp instead of one per control step.
# Same tunnel shape as the cameras, shifted by two: ssh -R 5593:localhost:5557
ROBOT          = "tcp://localhost:5593"   # node side: trajectory replay via ssh -R
ROBOT_LAB_BIND = "tcp://localhost:5557"   # lab side: robot_server's bind default

# Baseline grasp evaluation (servers/baseline_server.py): a baseline method
# submits one EE pose and it is executed with the pipeline's own
# solve_traj + rollout_traj deploy path. JSON wire format on purpose —
# baselines live in foreign conda envs (or other languages) where the pickled
# numpy protocol bites.
BASELINE = f"ipc://{_IPC_DIR}/baseline.sock"
# External GraspMolmo bridge (scripts/submit_baseline_pose.py). Its wire format
# is its own: one npz-compressed blob in (task/rgb/pc/intrinsics/camera_pose),
# one JSON reply out. Runs on whatever GPU box hosts it — tunnel 8015 across.
Grasp_Molmo = "tcp://localhost:8015"
# The shared baseline-method port: every method server (main_thompson_sim.py,
# graspmolmo_sim.py, scripts/test_method_server.py) binds THIS endpoint, one at
# a time — scripts/run_baselines.py is their single generic client and never
# knows which method is behind it. Wire format: npz blob in (scene/task/rgb/pc/
# intrinsics/camera_pose; a JSON {"cmd":"ping"} is also accepted), JSON out
# ({"status","method","pose"}). tcp — not ipc — so the method can live on the GPU cluster
# behind an ssh tunnel (-L 8016:localhost:8016 from the machine running
# run_baselines).
METHOD = "tcp://localhost:8016"

def bind(socket, endpoint):
    """Bind, first removing a stale ipc socket file so a hard-crashed server's
    leftover doesn't block rebind (servers re-exec on scene switch; a clean
    re-exec closes the socket and unlinks, but a crash leaves the file behind).
    ponytail: assumes one server per endpoint on this host — it stomps any
    existing socket file at that path."""
    if endpoint.startswith("ipc://"):
        path = endpoint[len("ipc://"):]
        if os.path.exists(path):
            os.unlink(path)
    socket.bind(endpoint)
    return endpoint

# ─ Pipeline settings ─────────────────────────────────────────────
TASK   = "hand me the spray bottle"
TARGET = "bottle"

# Franka parallel-jaw fingers open to 8 cm. The red (longest) scale bar
# get_grasp_width offers the VLM (rendering.RULER_BARS) is exactly this opening.
GRIPPER_MAX_WIDTH_M = 0.08

Parent_folder = Path(__file__).resolve().parent.parent
TEST_DIR = Parent_folder / "scripts" / "data" / "scene_10"

# Toggle each stage: True = call the live service, False = load from test/
USE_LIVE_CAMERA = True   # False -> load test/rgb.png + test/pointcloud.ply
USE_LIVE_ER  = True   # False -> load test/objects.json
USE_LIVE_LANGSAM = True  # False -> load test/masks/
USE_LIVE_SAM3D  = True  # False -> load test/meshes/ + test/meshes/transforms.json
RENDER_FINAL    = False   # False -> skip Isaac Sim render of selected grasp

# Which capture to use when USE_LIVE_CAMERA is False.
# Set to a prefix like "capture_20260303_131225", or None to use test/ defaults.
CAPTURE_PREFIX = None

IMG_WIDTH  = 1024
IMG_HEIGHT = 1024
CAMERA_FOV_Y = math.pi / 3.0  # 60 vertical FOV

# Manual color exposure for the Orbbec camera, in device units (lower = darker).
# Set to None to keep the sensor's auto-exposure. The first live capture prints
# the valid min..max (and current/default) for your device -- tune from there.
COLOR_EXPOSURE = 50

# ── Real camera (Orbbec Femto Mega) colour intrinsics ─────────────
# Single source of truth for the live colour camera. Every consumer
# (perception back-projection, ArUco/solvePnP, etc.) imports these.
#
# Factory calibration read from the device for the 1920x1080 colour stream via
# pyorbbecsdk: `color_profile.get_intrinsic()`. To refresh after a firmware
# update or when swapping the unit, run that and paste the *_RAW values below.
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080

# Raw device focal lengths / principal point, in the un-rotated sensor frame.
_CAM_FX_RAW, _CAM_FY_RAW = 1128.720, 1128.520
_CAM_CX_RAW, _CAM_CY_RAW = 958.499, 504.588

# Factory distortion in Orbbec's order (k1,k2,k3,k4,k5,k6,p1,p2), un-rotated
# sensor frame. This lens is NOT a pinhole: k1..k3 bend a corner ray by ~20 px
# (~17 mm of lateral error at 1 m), zero at the principal point and growing
# radially. The colour stream is raw MJPG — the device never rectifies it, and
# AlignFilter registers depth into the *distorted* colour image — so anything
# turning a pixel into a ray has to undo this. semantic_grasp/camera_model.py
# is the one place that does; go through it rather than (u - cx) * z / fx.
CAMERA_DISTORTION = (0.080802, -0.106045, 0.043203, 0.0, 0.0, 0.0, -0.000479, 0.000196)

# capture_rgbd() rotates colour+depth 180deg to undo the upside-down sensor
# mounting, so every consumer sees the rotated frame. Focal lengths are
# invariant under a 180deg rotation; the principal point mirrors about centre.
CAMERA_FX = _CAM_FX_RAW
CAMERA_FY = _CAM_FY_RAW
CAMERA_CX = (CAMERA_WIDTH  - 1) - _CAM_CX_RAW
CAMERA_CY = (CAMERA_HEIGHT - 1) - _CAM_CY_RAW

# The same distortion re-expressed for the rotated frame, in OpenCV's order
# (k1,k2,p1,p2,k3,k4,k5,k6) so it can be handed straight to cv2.undistortPoints
# / projectPoints / solvePnP alongside a K built from CAMERA_FX..CAMERA_CY.
#
# Rotating 180deg about the optical axis maps normalized coords (x,y) -> (-x,-y).
# The radial terms are odd in (x,y) and survive untouched; the tangential terms
# are even, so they only stay equivariant if p1 and p2 flip sign. (Verified
# against back-and-forth through the raw frame: agreement to 3e-13 px.)
_k1, _k2, _k3, _k4, _k5, _k6, _p1, _p2 = CAMERA_DISTORTION
CAMERA_DIST_CV = (_k1, _k2, -_p1, -_p2, _k3, _k4, _k5, _k6)

VIEWPOINTS = [
    # label              azimuth(deg)  elevation(deg)
    # --- elevation 0 (8 views at 45 increments) ---
    ("front",               0,          0),
    ("front_right",        45,          0),
    ("right",              90,          0),
    ("back_right",        135,          0),
    ("back",              180,          0),
    ("back_left",         225,          0),
    ("left",              270,          0),
    ("front_left",        315,          0),
    # --- elevation 40 (8 views at 45 increments) ---
    ("front_high",          0,         40),
    ("front_right_high",   45,         40),
    ("right_high",         90,         40),
    ("back_right_high",   135,         40),
    ("back_high",         180,         40),
    ("back_left_high",    225,         40),
    ("left_high",         270,         40),
    ("front_left_high",   315,         40),
    # --- elevation -30 (4 views at 90 increments) ---
    ("front_low",           0,        -30),
    ("right_low",          90,        -30),
    ("back_low",          180,        -30),
    ("left_low",          270,        -30),
    # --- top ---
    ("top",                 0,         85),
]



# ── Unprojection constants ────────────────────────────────────────
COORD_SCALE = 1000
DEPTH_SEARCH_RADIUS = 5
DBSCAN_MIN_SAMPLES = 3

# ── Troubleshooting log ──────────────────────────────────────────
TROUBLESHOOTING_FILE = OUTPUT_DIR / "troubleshooting.txt"
