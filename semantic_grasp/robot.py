"""
Execute Trajectory: replay a cuRobo solve_traj() trajectory on the real Franka.

Create a Robot, hand it the output of IKClient.solve_traj(), and it drives the
arm through the waypoints with joint-space P-control and works the gripper. No
HDF5 file in the loop — the trajectory comes straight from the planner.

    from semantic_grasp.ik import IKClient
    from semantic_grasp.robot import Robot
    import numpy as np

    client = IKClient("tcp://localhost:5557")
    goal = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)   # robot frame
    traj, ok, info = client.solve_traj(goal, meshes, transformations, hz=20.0)

    robot = Robot()                 # binds arm:8090 + gripper:8091
    robot.execute(traj, ok, info)   # go-to-start → open → run → close

Or pass the solve_traj() tuple straight through:

    robot.execute(client.solve_traj(goal, meshes, transformations))
    robot.execute(*client.solve_traj(goal, meshes, transformations))

`solve_traj` returns positions only, so the gripper is opened before the move
and closed at the end (a pick). Toggle with open_gripper=/close_gripper=, or call
robot.open_gripper() / robot.close_gripper() yourself for finer control.

CLI demo (plans against the running server, then executes):
    python -m semantic_grasp.robot --goal 0.45 0.10 0.35 3.14159 0 0
    python -m semantic_grasp.robot --dry-run         # print waypoints, no robot
    python -m semantic_grasp.robot --subsample 5     # thin a dense 20 Hz path
"""

import argparse
import pickle
import socket
import time

import numpy as np


# Joint-space P-control parameters
K_P = 1.5                 # proportional gain (joint velocity = K_P * error)
JOINT_TOL = 0.12          # radians — convergence threshold per waypoint
MAX_STEPS = 100           # max control iterations per waypoint
WAYPOINT_TIMEOUT = 5.0    # seconds — safety timeout per waypoint
QDOT_LIMIT = 1.5          # rad/s — max joint velocity magnitude
START_TOL = 0.3           # radians — warn/confirm if robot is this far from traj[0]
# The loose per-waypoint tolerance keeps the replay smooth but lets the robot
# lag the stream; without a tight final servo it parks ~0.1 rad (several cm of
# EE error, always on the approach side) short of the grasp.
FINAL_TOL = 0.015         # radians — convergence threshold for the goal waypoint
FINAL_TIMEOUT = 10.0      # seconds — the tight final servo needs longer to settle

# Streaming replay (execute_streamed). Franka.send2robot applies this same cap to
# the norm of every command it sends, so it — not QDOT_LIMIT — is the real
# ceiling on how fast the arm may move.
STREAM_LIMIT = 1.0        # rad/s — max commanded joint-velocity norm while streaming
# How often a velocity command goes out, independent of how fast the motion is.
# The arm holds the last command until the next arrives, so a low rate turns any
# path into a staircase; this keeps the steps small no matter the time_scale.
COMMAND_HZ = 50.0         # Hz — streaming command rate
# Clock stretch used when deploying a solved grasp. HIGHER = SLOWER; the path is
# unchanged, it is just walked more slowly. Shared by main_thompson.py's deploy
# step and scripts/run_traj_from_cache.py so tuning a replay in one carries to
# the other. On a typical ~2 s plan: 2.51 -> 5 s, 6.0 -> 12 s, 10.0 -> 20 s.
DEPLOY_TIME_SCALE = 6.0
# Post-grasp lift (Robot.lift): the TCP goes straight up through the Jacobian
# the NUC streams. Shared by the deploy scripts and run_traj_from_cache for the
# same reason as DEPLOY_TIME_SCALE.
LIFT_M        = 0.20      # metres the gripper_tcp rises after the gripper closes
LIFT_SPEED    = 0.05      # m/s — slow to limit slip, the reason the sim lift is slow too
LIFT_HOLD_RPY = 0.3       # 0 = orientation fully free, 1 = held rigidly; see Robot.lift

# Frame bookkeeping for read_tcp_pose(). Franka.joint2pose ends its chain here,
# measured along panda_link7's +z:
_FK_Z_FROM_LINK7 = 0.2
# ...while the IK server targets `gripper_tcp`, at 0.107 (link7 -> flange) plus
# its own panda_hand -> TCP offset. KEEP IN LOCK-STEP with
# ik_server._GRIPPER_TCP_OFFSET — robot.py cannot import it (that module pulls in
# torch + curobo, and this one has to stay numpy-and-sockets so it runs lab-side).
# 2026-09-05: 0.19267 -> 0.20407 to match ik_server (TCP 8.6 mm behind the jaw tips).
_GRIPPER_TCP_OFFSET = 0.20407
_FK_TO_TCP_Z = 0.107 + _GRIPPER_TCP_OFFSET - _FK_Z_FROM_LINK7   # 0.11107 m


def _dls(A, lam):
    """Damped least-squares right inverse A^T (A A^T + lam^2 I)^-1: the
    pseudo-inverse with its gain capped at 1/(2*lam) along directions where A
    loses rank, so a near-singular Jacobian yields a slow motion, not a huge
    joint velocity. Used by Robot.lift."""
    A = np.asarray(A, dtype=float)
    return A.T @ np.linalg.inv(A @ A.T + (lam ** 2) * np.eye(A.shape[0]))


def _resample_path(waypoints, dt_src, dt_dst):
    """Linearly resample a (T, dof) joint path from `dt_src` to `dt_dst` spacing.

    Pure densification/thinning in time: the path through joint space and its
    total duration are unchanged, only the number of samples along it differs.
    The exact final config is always kept as the last row — it is the grasp
    pose, and interpolation must not round it off.
    """
    waypoints = np.asarray(waypoints)
    T = len(waypoints)
    if T < 2 or abs(dt_src - dt_dst) < 1e-9:
        return waypoints

    src_t = np.arange(T, dtype=np.float64) * dt_src
    duration = float(src_t[-1])
    n_dst = int(np.floor(duration / dt_dst)) + 1
    dst_t = np.arange(n_dst, dtype=np.float64) * dt_dst

    out = np.empty((n_dst, waypoints.shape[1]), dtype=waypoints.dtype)
    for j in range(waypoints.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, waypoints[:, j])
    if abs(dst_t[-1] - duration) > 1e-9:      # grid missed the end — append it
        out = np.vstack([out, waypoints[-1]])
    return out


class Robot:
    """High-level wrapper around the Franka comms that replays a solve_traj()
    trajectory. `Franka` (below) is the low-level transport."""

    # 8080/8081 are unusable on BCM-managed cluster nodes (root's cmd.service
    # holds them dual-stack) — the lab-PC tunnel remaps its 8080/8081 to these.
    def __init__(self, arm_port: int = 8090, gripper_port: int = 8091,
                 *, connect: bool = True,
                 k_p: float = K_P, joint_tol: float = JOINT_TOL,
                 max_steps: int = MAX_STEPS, waypoint_timeout: float = WAYPOINT_TIMEOUT,
                 qdot_limit: float = QDOT_LIMIT, start_tol: float = START_TOL,
                 final_tol: float = FINAL_TOL, final_timeout: float = FINAL_TIMEOUT):
        self.robot = Franka()
        self.arm_port = arm_port
        self.gripper_port = gripper_port
        self.conn_robot = None
        self.conn_gripper = None

        self.k_p = k_p
        self.joint_tol = joint_tol
        self.max_steps = max_steps
        self.waypoint_timeout = waypoint_timeout
        self.qdot_limit = qdot_limit
        self.start_tol = start_tol
        self.final_tol = final_tol
        self.final_timeout = final_timeout

        if connect:
            self.connect()

    # ── connection / low-level ─────────────────────────────────────────────────
    def connect(self):
        """Open the arm and gripper sockets (idempotent)."""
        if self.conn_robot is None:
            print(f"Connecting to robot on port {self.arm_port}...")
            self.conn_robot = self.robot.connect(self.arm_port)
        if self.conn_gripper is None:
            print(f"Connecting to gripper on port {self.gripper_port}...")
            self.conn_gripper = self.robot.connect(self.gripper_port)
        return self

    def _require_conn(self):
        if self.conn_robot is None:
            self.connect()

    def read_state(self):
        self._require_conn()
        return self.robot.readState(self.conn_robot)

    def stop(self):
        """Command zero joint velocity."""
        state = self.read_state()
        self.robot.send2robot(self.conn_robot, 0.0 * state["q"])

    def read_tcp_pose(self, state=None):
        """Measured gripper_tcp pose [x, y, z, roll, pitch, yaw] in the ROBOT frame
        — the same frame IK solves against, so it is directly comparable to a
        deploy_pose.

        state["x"] is NOT that frame: Franka.joint2pose stops _FK_Z_FROM_LINK7 m
        along panda_link7's +z, while gripper_tcp sits at 0.107 (link7->flange)
        + the IK server's gripper offset. Both frames share that +z axis (the
        -45 deg hand rotation is about z), so the gap is a pure translation
        along the tool axis — see _FK_TO_TCP_Z.
        """
        if state is None:
            state = self.read_state()
        xyz_fk = np.asarray(state["x"][:3], dtype=float)
        rpy    = np.asarray(state["x"][3:6], dtype=float)
        # Tool +z in robot coords, rebuilt from the measured roll/pitch/yaw the
        # same way joint2pose derived them (Rz(yaw) @ Ry(pitch) @ Rx(roll)).
        r, p, y = rpy
        z_tool = np.array([
            np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r),
            np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r),
            np.cos(p) * np.cos(r),
        ])
        return np.concatenate([xyz_fk + _FK_TO_TCP_Z * z_tool, rpy])

    def open_gripper(self, wait: float = 1.0):
        self._require_conn()
        print("Opening gripper...")
        self.robot.send2gripper(self.conn_gripper, "o")
        if wait:
            time.sleep(wait)

    def close_gripper(self, wait: float = 1.0):
        self._require_conn()
        print("Closing gripper...")
        self.robot.send2gripper(self.conn_gripper, "c")
        if wait:
            time.sleep(wait)

    # ── post-grasp lift: Cartesian velocity through the streamed Jacobian ─────
    def lift(self, height_m=LIFT_M, *, speed=LIFT_SPEED, command_hz=COMMAND_HZ,
             limit=STREAM_LIMIT):
        """Raise the gripper_tcp straight up by `height_m` (robot-frame +z), the
        gripper as it is, by streaming joint velocities from the Jacobian the
        NUC sends with every state (the same J calibrate_c2r drives the arm by):

            qdot = dls(W J) @ W [0, 0, speed, 0, 0, 0]     W = diag(1,1,1, w,w,w)

        w = LIFT_HOLD_RPY down-weights the angular rows, so orientation is held
        softly and yields when holding it would fight the climb — the lift stays
        feasible from the awkward configurations a refined grasp can end in.
        (w = 0 is fully free, but the minimum-norm motion then pitches the hand
        ~30 deg over 20 cm and swings the TCP a few cm sideways; w = 1 holds it
        rigidly.) Damped least squares keeps joint speeds bounded near
        singularities. Stops when read_tcp_pose's z has risen `height_m`, when
        it gains under 1 cm in 1.5 s (stalled on a joint limit or singularity),
        or at a generous timeout. Always brakes. Returns (rose_m, reached).
        """
        self._require_conn()
        height_m, speed = float(height_m), float(speed)
        dt, timeout, stall_s = 1.0 / float(command_hz), 2.0 * height_m / speed + 3.0, 1.5
        W = np.array([1.0, 1.0, 1.0] + [LIFT_HOLD_RPY] * 3)
        xdot = W * np.array([0.0, 0.0, speed, 0.0, 0.0, 0.0])
        z0 = float(self.read_tcp_pose()[2])
        print(f"\nLifting {height_m * 100:.0f} cm at {speed * 100:.0f} cm/s...")
        t0 = time.perf_counter()
        t_mark, rose_mark, rose, reached, i = t0, 0.0, 0.0, False, 0
        try:
            while True:
                now = time.perf_counter()
                state = self.read_state()
                rose = float(self.read_tcp_pose(state)[2]) - z0
                if rose >= height_m:
                    reached = True
                    break
                if now - t_mark > stall_s:
                    if rose - rose_mark < 0.01:
                        print(f"  WARN: lift stalled at {rose * 100:.1f} cm; stopping")
                        break
                    t_mark, rose_mark = now, rose
                if now - t0 > timeout:
                    print(f"  WARN: lift timed out at {rose * 100:.1f} cm; stopping")
                    break
                qdot = _dls(state["J"] * W[:, None], 0.05) @ xdot
                self.robot.send2robot(self.conn_robot, qdot, limit=limit)
                i += 1
                slack = (t0 + i * dt) - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
        finally:
            self.stop()
        print(f"  rose {rose * 100:.1f} cm in {time.perf_counter() - t0:.1f} s"
              f"{'' if reached else ' (short of target)'}")
        return rose, reached

    # ── joint-space P-control to one target ────────────────────────────────────
    def move_to_joints(self, goal_q, *, joint_tol=None, timeout=None, max_steps=None,
                       brake=True):
        """Drive to a target joint configuration using joint-space P-control.
        joint_tol / timeout / max_steps default to the per-waypoint settings.

        brake : command zero velocity once the target is reached. Leave it True
                for a standalone move. Pass False when streaming a path — the
                waypoints of a solve_traj() plan sit well inside joint_tol
                (~0.04 rad apart vs a 0.12 rad tolerance), so most of them
                return on the first iteration, and braking at each one turns a
                continuous motion into a drive-stop-drive pulse.
        Returns (reached: bool, steps: int, dist: float)."""
        self._require_conn()
        joint_tol = self.joint_tol if joint_tol is None else joint_tol
        timeout = self.waypoint_timeout if timeout is None else timeout
        max_steps = self.max_steps if max_steps is None else max_steps
        goal_q = np.asarray(goal_q, dtype=float)
        start_time = time.time()
        dist = float("inf")
        for step in range(max_steps):
            if time.time() - start_time > timeout:
                print(f"    Timeout after {timeout}s")
                break

            state = self.read_state()
            error = goal_q - state["q"]
            dist = float(np.linalg.norm(error))

            if dist < joint_tol:
                if brake:
                    self.robot.send2robot(self.conn_robot, 0.0 * state["q"])
                return True, step, dist

            qdot = self.k_p * error
            scale = np.linalg.norm(qdot)
            if scale > self.qdot_limit:
                qdot *= self.qdot_limit / scale

            self.robot.send2robot(self.conn_robot, qdot)

        self.stop()
        return False, max_steps, dist

    # ── shared trajectory preamble ─────────────────────────────────────────────
    @staticmethod
    def _unpack_traj(traj, success, info):
        """Normalize the (traj | (traj, success[, info])) argument forms.

        Returns (waypoints (T,7) float32, success bool, dt float or None), or
        (None, ...) if there is nothing runnable.
        """
        if isinstance(traj, (tuple, list)) and len(traj) in (2, 3) and \
                not isinstance(traj[0], (int, float)):
            if len(traj) == 3:
                traj, success, info = traj
            else:
                traj, success = traj

        if not success:
            print("Refusing to execute: planner reported success=False.")
            return None, False, None

        waypoints = np.asarray(traj, dtype=np.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] < 7:
            raise ValueError(f"traj must be (T, >=7), got {waypoints.shape}")
        waypoints = waypoints[:, :7]                  # trim finger cols if present
        if len(waypoints) == 0:
            print("Refusing to execute: empty trajectory.")
            return None, False, None

        dt = float(info["dt"]) if (info and "dt" in info) else None
        return waypoints, True, dt

    def _preflight(self, waypoints, *, confirm_start, go_to_start):
        """Connect, report where the arm is vs the planned start, optionally
        drive to it. Returns False if the operator aborted."""
        self.connect()

        state = self.read_state()
        print(f"Current joints: {[round(v, 4) for v in state['q']]}")
        start_dist = float(np.linalg.norm(state["q"] - waypoints[0]))
        if confirm_start and start_dist > self.start_tol:
            print(f"\nWARNING: robot is {start_dist:.3f} rad from the planned start.")
            print(f"  Expected: {[round(v, 4) for v in waypoints[0]]}")
            print(f"  Current:  {[round(v, 4) for v in state['q']]}")
            if input("Continue anyway? [y/N] ").lower() != "y":
                print("Aborted.")
                return False

        if go_to_start:
            print("\nMoving to start config...")
            ok, steps, dist = self.move_to_joints(waypoints[0])
            print(f"  {'Reached' if ok else 'Warning: not reached'} "
                  f"(steps={steps}, dist={dist:.4f})")
        return True

    # ── streaming replay: feedforward velocity on the plan's own clock ─────────
    def execute_streamed(self, traj, success=True, info=None, *,
                         dt=None, time_scale=1.0, command_hz=COMMAND_HZ,
                         limit=STREAM_LIMIT,
                         go_to_start=True, confirm_start=True,
                         open_gripper=True, close_gripper=True,
                         dry_run=False) -> bool:
        """Replay a solve_traj() trajectory by streaming velocity commands.

        The alternative to execute(), which servos to each waypoint in turn.
        That loop derives its speed from the *position error* to the next
        waypoint, so with a dense plan (waypoints ~0.04 rad apart) it commands
        k_p * 0.04 ~ 0.06 rad/s where the plan wants ~1.9 — it cannot track the
        path at the speed it was planned for, and the motion comes out as a
        stutter of brief commands separated by stalls.

        Here each command is instead

            qdot = (q[i+1] - q[i]) / dt      feedforward: the planned velocity
                 + k_p * (q[i+1] - q_meas)   feedback: corrects drift only

        issued on the plan's own clock, so the trajectory's timing is honoured
        rather than discarded. The feedforward term does the work; the feedback
        term stays small.

        dt          : step of the plan, in seconds. Defaults to info["dt"].
        time_scale  : >1 stretches the plan (slower, same path). The peak
                      feedforward must fit under `limit` or the arm lags and the
                      final servo has to make up the difference — a warning below
                      reports the time_scale that would fit.
        limit       : max commanded joint-velocity norm (rad/s). Franka.send2robot
                      applies the same cap internally, so raising this alone does
                      nothing — the transport clamps it back down.

        Returns True if the full trajectory was streamed.
        """
        waypoints, success, info_dt = self._unpack_traj(traj, success, info)
        if waypoints is None:
            return False

        dt = float(dt if dt is not None else (info_dt if info_dt is not None else 0.0))
        if dt <= 0:
            print("Refusing to stream: no dt (pass dt=... or an info dict with one).")
            return False
        dt_exec = dt * float(time_scale)

        n = len(waypoints)
        if n < 2:
            print("Refusing to stream: need at least 2 waypoints.")
            return False

        # Hold the COMMAND RATE fixed, independent of how slow the motion is.
        # time_scale stretches dt, which would otherwise stretch the interval
        # between commands too: at time_scale=6 a 50 Hz plan issues velocities
        # only every 120 ms, and the arm holds each one until the next arrives —
        # a staircase that reads as jerky however smooth the underlying path is.
        # Resampling onto a 1/command_hz grid keeps the same duration and the
        # same path, just walked in smaller, more frequent steps.
        dt_cmd = 1.0 / float(command_hz)
        if abs(dt_cmd - dt_exec) > 1e-9:
            waypoints = _resample_path(waypoints, dt_exec, dt_cmd)
            print(f"  resampled {n} -> {len(waypoints)} waypoints "
                  f"for a {command_hz:.0f} Hz command rate "
                  f"(was {1.0 / dt_exec:.1f} Hz)")
            n, dt_exec = len(waypoints), dt_cmd

        # What the plan asks for, vs what the transport will actually pass.
        qdot_ff = np.diff(waypoints, axis=0) / dt_exec
        v_peak = float(np.linalg.norm(qdot_ff, axis=1).max())
        print(f"Streaming {n} waypoints @ dt={dt_exec:.4f}s "
              f"({(n - 1) * dt_exec:.2f}s of planned motion)")
        print(f"  planned |qdot|: median {np.median(np.linalg.norm(qdot_ff, axis=1)):.2f}  "
              f"peak {v_peak:.2f} rad/s   (limit {limit:.2f})")
        if v_peak > limit:
            print(f"  NOTE: peak exceeds the limit — the arm will move at ~{limit / v_peak:.0%} "
                  f"of planned speed and lag the stream; the final servo makes up "
                  f"the difference. time_scale={v_peak / limit * float(time_scale):.2f} "
                  f"would fit the plan under the limit.")

        if dry_run:
            print(f"\n--- Dry run: {n - 1} velocity commands ---")
            for i, v in enumerate(qdot_ff):
                print(f"  [{i:3d}] qdot_ff={[round(float(x), 4) for x in v]}  "
                      f"|{np.linalg.norm(v):.3f}|")
            return True

        if not self._preflight(waypoints, confirm_start=confirm_start,
                               go_to_start=go_to_start):
            return False

        if open_gripper:
            self.open_gripper()

        print(f"\nStreaming trajectory ({n} waypoints)...")
        overruns = 0
        t0 = time.perf_counter()
        try:
            for i in range(n - 1):
                q_next = waypoints[i + 1]
                state = self.read_state()
                qdot = qdot_ff[i] + self.k_p * (q_next - state["q"])
                self.robot.send2robot(self.conn_robot, qdot, limit=limit)

                # Hold the plan's clock. Falling behind means read_state or the
                # socket is slower than dt — the trajectory then plays slower
                # than planned, which is safe but worth knowing about.
                slack = (t0 + (i + 1) * dt_exec) - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    overruns += 1
        finally:
            self.stop()

        elapsed = time.perf_counter() - t0
        print(f"  streamed {n - 1} commands in {elapsed:.2f}s "
              f"(planned {(n - 1) * dt_exec:.2f}s)"
              + (f", {overruns} step(s) overran dt" if overruns else ""))

        # The stream leaves the arm wherever it got to; the clamp above and any
        # tracking lag both land on the approach side of the grasp.
        print(f"\nFinal servo to goal (tol={self.final_tol} rad)...")
        ok, steps, dist = self.move_to_joints(
            waypoints[-1], joint_tol=self.final_tol,
            timeout=self.final_timeout, max_steps=100 * self.max_steps)
        print(f"  {'Converged' if ok else 'WARN: not converged'} "
              f"(steps={steps}, dist={dist:.4f})")
        self.stop()

        if close_gripper:
            print()
            self.close_gripper()

        state = self.read_state()
        print(f"\nFinal joints: {[round(v, 4) for v in state['q']]}")
        print(f"Final EE pos: {[round(v, 4) for v in state['x'][:3]]}")
        print(f"Goal  joints: {[round(v, 4) for v in waypoints[-1]]}")
        print("Done.")
        return True

    # ── main entry: replay a solve_traj() trajectory ───────────────────────────
    def execute(self, traj, success=True, info=None, *,
                subsample: int = 1, skip: int = 0,
                go_to_start: bool = True, confirm_start: bool = True,
                open_gripper: bool = True, close_gripper: bool = True,
                dry_run: bool = False) -> bool:
        """Replay the output of IKClient.solve_traj() on the robot.

        traj    : (T, 7) joint-position trajectory, or the whole solve_traj()
                  tuple (traj, success, info) — both forms are accepted.
        success : planner success flag (refuses to run a failed/empty plan).
        info    : solve_traj() info dict; only used for the dt/duration printout.

        subsample / skip : thin a dense path / drop the first N waypoints.
        go_to_start      : P-control to traj[0] before streaming the path.
        confirm_start    : prompt if the robot starts far (> start_tol) from traj[0].
        open_gripper     : open before the move; close_gripper: close at the end.

        Returns True if the full trajectory was sent.
        """
        # Accept the solve_traj() tuple straight through.
        waypoints, success, dt = self._unpack_traj(traj, success, info)
        if waypoints is None:
            return False
        n_orig = len(waypoints)

        # Subsample (always keep the final / goal waypoint).
        if subsample > 1:
            idx = list(range(0, n_orig, subsample))
            if idx[-1] != n_orig - 1:
                idx.append(n_orig - 1)
            waypoints = waypoints[idx]
            print(f"Subsampled: {len(waypoints)}/{n_orig} waypoints (every {subsample}th)")
        else:
            print(f"Using full trajectory: {n_orig} waypoints")

        # Skip leading waypoints (e.g. trim a home-approach segment).
        if skip > 0:
            if skip >= len(waypoints):
                print(f"ERROR: --skip {skip} >= waypoints {len(waypoints)}")
                return False
            waypoints = waypoints[skip:]
            print(f"Skipped first {skip} waypoints, {len(waypoints)} remaining")

        if dt is not None:
            print(f"dt={dt}  (~{(len(waypoints) - 1) * dt:.2f}s of planned motion)")
        print(f"Start config: {[round(v, 4) for v in waypoints[0]]}")
        print(f"Goal  config: {[round(v, 4) for v in waypoints[-1]]}")

        if dry_run:
            print(f"\n--- Dry run: {len(waypoints)} waypoints ---")
            for i, wp in enumerate(waypoints):
                print(f"  [{i:3d}] joints={[round(v, 4) for v in wp]}")
            return True

        if not self._preflight(waypoints, confirm_start=confirm_start,
                               go_to_start=go_to_start):
            return False

        if open_gripper:
            self.open_gripper()

        print(f"\nExecuting trajectory ({len(waypoints)} waypoints)...")
        for i, wp in enumerate(waypoints):
            ok, steps, dist = self.move_to_joints(wp, brake=False)
            status = "OK" if ok else "WARN"
            print(f"  [{i+1:3d}/{len(waypoints)}] {status}  steps={steps:3d}  dist={dist:.4f}")

        # Servo the goal waypoint down to final_tol: the loose per-waypoint
        # tolerance leaves the EE parked on the approach side of the grasp.
        print(f"\nFinal servo to goal (tol={self.final_tol} rad)...")
        ok, steps, dist = self.move_to_joints(
            waypoints[-1], joint_tol=self.final_tol,
            timeout=self.final_timeout, max_steps=100 * self.max_steps)
        print(f"  {'Converged' if ok else 'WARN: not converged'} "
              f"(steps={steps}, dist={dist:.4f})")

        self.stop()

        if close_gripper:
            print()
            self.close_gripper()

        state = self.read_state()
        print(f"\nFinal joints: {[round(v, 4) for v in state['q']]}")
        print(f"Final EE pos: {[round(v, 4) for v in state['x'][:3]]}")
        print(f"Goal  joints: {[round(v, 4) for v in waypoints[-1]]}")
        print("Done.")
        return True


class RobotClient:
    """Client for servers/robot_server.py — replay a trajectory on the lab-side arm.

    Same shape as GraspClient / IKClient: pickled dicts over ZMQ REQ. Use this
    instead of Robot() when the planning process is NOT on the lab PC — it ships
    the whole trajectory in one message and lets the 50 Hz control loop run next
    to the robot, where its command clock can actually be met.

        from semantic_grasp.robot import RobotClient
        client = RobotClient()                     # tcp://localhost:5593 (ssh -R)
        res = client.execute(traj, traj_ok, traj_info)
        print(res["q"], res["tcp_pose"])

    `execute` returns the server's reply dict: {"status", "executed", "q",
    "tcp_pose", "start_dist"} — or {"status": "error", "message": ...}. It does
    NOT raise on a refused move, so check "status".
    """

    def __init__(self, addr: str | None = None, timeout_s: float = 600.0):
        import zmq
        from .config import ROBOT
        self._zmq = zmq
        self.addr = addr or ROBOT
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        # A replay runs for tens of seconds; the default (infinite) would hang
        # forever on a dead server, and a short one would abort a live move.
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def _request(self, req: dict) -> dict:
        self.sock.send(pickle.dumps(req))
        return pickle.loads(self.sock.recv())

    def ping(self) -> dict:
        """Is the server up and the arm connected?"""
        return self._request({"cmd": "ping"})

    def state(self) -> dict:
        """Measured {"q", "tcp_pose"} without moving anything."""
        return self._request({"cmd": "state"})

    def execute(self, traj, success=True, info=None, *, dt=None,
                time_scale=DEPLOY_TIME_SCALE, command_hz=COMMAND_HZ,
                go_to_start=True, open_gripper=True, close_gripper=True,
                force=False, lift_m=0.0) -> dict:
        """Replay `traj` on the lab-side arm. Accepts the same (traj, success,
        info) forms as Robot.execute_streamed; `dt` defaults to info["dt"].

        force=True overrides the server's refusal when the arm starts far from
        the trajectory's first waypoint. lift_m > 0 has the server raise the
        TCP that far straight up after the gripper closes (Robot.lift,
        orientation free); the reply's q/tcp_pose are measured BEFORE that
        lift, and "lifted_m"/"lift_ok" report how it went.
        """
        waypoints, success, info_dt = Robot._unpack_traj(traj, success, info)
        if waypoints is None:
            return {"status": "error", "message": "no runnable trajectory",
                    "executed": False}
        dt = dt if dt is not None else info_dt
        if not (isinstance(dt, (int, float)) and dt > 0):
            return {"status": "error", "message": "no dt (pass dt= or info with one)",
                    "executed": False}
        return self._request({
            "cmd": "execute", "traj": waypoints, "dt": float(dt),
            "time_scale": float(time_scale), "command_hz": float(command_hz),
            "go_to_start": bool(go_to_start), "open_gripper": bool(open_gripper),
            "close_gripper": bool(close_gripper), "force": bool(force),
            "lift_m": float(lift_m),
        })


# ── CLI demo: plan with solve_traj against the running server, then execute ────
# ══════════════════════════════════════════════════════════════════════════════
#  Low-level Franka socket protocol (arm + gripper TCP servers on the robot PC)
# ══════════════════════════════════════════════════════════════════════════════
class Franka(object):

    def __init__(self):
        self.home = np.array([-0.232867, -0.524729, 0.137301, -2.3478, 0.0455863, 1.85082, 0.64003])

    def connect(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', port))
        except OSError as exc:
            # Lab-side, the usual culprit is the ssh tunnel: `-g -L 8080:...`
            # binds 8080/8081 on this machine's interfaces, which is exactly
            # what robot_server.py needs. Only one of them can own the port,
            # and whichever does is the one the NUC ends up talking to.
            raise OSError(
                f"could not bind port {port}: {exc}\n"
                f"  Something already holds it — check with:  "
                f"ss -lntp | grep :{port}\n"
                f"  If that is an ssh process, your tunnel's "
                f"'-L {port}:localhost:...' forward is squatting it. Drop the "
                f"-L 8080/-L 8081 pair when running robot_server.py lab-side "
                f"(the NUC then connects here directly); keep it only when "
                f"Robot() runs on the GPU cluster."
            ) from exc
        s.listen()
        conn, addr = s.accept()
        return conn

    def send2gripper(self, conn, command):
        send_msg = "s," + command + ","
        conn.send(send_msg.encode())

    def send2robot(self, conn, qdot, limit=1.0):
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot *= limit/scale
        send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + ","
        conn.send(send_msg.encode())

    def listen2robot(self, conn):
        state_length = 7 + 6 + 42
        message = str(conn.recv(20480))[2:-2]
        state_str = list(message.split(","))
        for idx in range(len(state_str)):
            if state_str[idx] == "s":
                state_str = state_str[idx+1:idx+1+state_length]
                break
        try:
            state_vector = [float(item) for item in state_str]
        except ValueError:
            return None
        if len(state_vector) is not state_length:
            return None
        state_vector = np.asarray(state_vector)
        states = {}
        states["q"] = state_vector[0:7]
        states["O_F"] = state_vector[7:13]
        states["J"] = state_vector[13:].reshape((7,6)).T
        xyz_lin, R = self.joint2pose(state_vector[0:7])
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        states["x"] = np.array(xyz)
        states["angle"] = np.array(xyz_ang)
        return states

    def readState(self, conn):
        while True:
            states = self.listen2robot(conn)
            if states is not None:
                break
        return states

    def xdot2qdot(self, xdot, states):
        J_inv = np.linalg.pinv(states["J"])
        return J_inv @ np.asarray(xdot)


    def joint2pose(self, q):
        def RotX(q):
            return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
        def RotZ(q):
            return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        def TransX(q, x, y, z):
            return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
        def TransZ(q, x, y, z):
            return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
        # Stops _FK_Z_FROM_LINK7 along panda_link7's +z — NOT the gripper_tcp
        # frame. read_tcp_pose() adds the remaining _FK_TO_TCP_Z to get there,
        # so this value and _FK_Z_FROM_LINK7 must stay equal. Sanity check: at
        # the home config the result must be near [0.307, 0.0, 0.497]; a wrong
        # value here shows up as an absurd z, not as a subtle offset.
        H_panda_hand = TransZ(-np.pi/4, 0, 0, _FK_Z_FROM_LINK7)
        T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        R = T[:,:3][:3]
        xyz = T[:,3][:3]
        return xyz, R

    def go2position(self, conn, goal=False):
        if type(goal) == bool:
            goal = self.home
        total_time = 20.0
        start_time = time.time()
        states = self.readState(conn)
        dist = np.linalg.norm(states["q"] - goal)
        elapsed_time = time.time() - start_time
        while dist > 0.05 and elapsed_time < total_time:
            qdot = np.clip(goal - states["q"], -0.1, 0.1)
            self.send2robot(conn, qdot)
            states = self.readState(conn)
            dist = np.linalg.norm(states["q"] - goal)
            elapsed_time = time.time() - start_time
        states = self.readState(conn)
        qdot = 0.0 * states["q"]
        self.send2robot(conn, qdot)


def main():
    parser = argparse.ArgumentParser(
        description="Plan with solve_traj and execute it on the real Franka")
    parser.add_argument("--addr", default=None,
                        help="IK/trajectory server address (default: ipc endpoint)")
    parser.add_argument("--goal", type=float, nargs=6,
                        metavar=("X", "Y", "Z", "R", "P", "YAW"),
                        default=[0.45, 0.10, 0.35, np.pi, 0.0, 0.0],
                        help="EE goal pose [x y z roll pitch yaw] in ROBOT frame")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="trajectory sampling rate")
    parser.add_argument("--subsample", type=int, default=1,
                        help="use every Nth waypoint (1 = all)")
    parser.add_argument("--skip", type=int, default=0,
                        help="skip the first N waypoints")
    parser.add_argument("--no-gripper", action="store_true",
                        help="don't open/close the gripper")
    parser.add_argument("--dry-run", action="store_true",
                        help="print waypoints without connecting to the robot")
    args = parser.parse_args()

    from .ik import IKClient

    # Table-only scene by default (no perception). Swap in your {name: trimesh}
    # meshes dict + name-keyed scene dict to plan around real objects.
    meshes = {}
    transformations = {"objects": {}}

    client = IKClient(args.addr) if args.addr else IKClient()
    print(f"Planning trajectory to goal {args.goal} ...")
    traj, ok, info = client.solve_traj(
        np.array(args.goal, dtype=np.float32), meshes, transformations, hz=args.hz)
    client.close()

    if not ok:
        print("Planning failed — goal unreachable or in collision.")
        return
    print(f"Planned {info['n_waypoints']} waypoints @ {info['hz']:.0f}Hz "
          f"({info['duration_s']}s)")

    robot = Robot(connect=not args.dry_run)
    robot.execute(
        traj, ok, info,
        subsample=args.subsample, skip=args.skip,
        open_gripper=not args.no_gripper, close_gripper=not args.no_gripper,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()