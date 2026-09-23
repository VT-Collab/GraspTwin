#!/usr/bin/env python3
"""Client for the cuRobo ZMQ IK/trajectory server (ik_server.py) — pickle I/O.

Arrays cross the wire as plain lists in BOTH directions (a pickled ndarray
embeds a numpy-version-specific module path — numpy._core on >=2.0 — and fails
to unpickle across the 1.x/2.x split, e.g. inside Isaac apps whose bundled
numpy 1.x shadows the env's). This client rebuilds ndarrays on receipt, so
callers still get the array types documented below.

Two entry points:

  • solve(...)       batch collision-free IK. Hand in an (N, 6) matrix of
                     [x, y, z, roll, pitch, yaw] poses (robot frame) plus the
                     perception outputs; get back (N, 7) joint solutions.

  • solve_traj(...)  full collision-free *trajectory* to a single EE goal pose,
                     sampled at a fixed rate (default 50 Hz) for direct replay on
                     a physical robot. Returns an (T, 7) joint-position array.

Object geometry travels INSIDE the request — meshes are never read from disk by
the server, so they don't need to be saved.

    from semantic_grasp.ik import IKClient
    import numpy as np

    client = IKClient()

    # ── batch IK ──────────────────────────────────────────────────────────────
    poses = np.zeros((100, 6), dtype=np.float32)
    poses[:, :3] = ...        # xyz in robot frame
    poses[:, 3:] = ...        # roll, pitch, yaw (radians)

    # meshes:          {name: trimesh Scene/Trimesh}
    # transformations: transforms.json dict (entries carry camera-frame
    #                  translation / rotation_wxyz / scale)
    joints, success, info = client.solve(poses, meshes, transformations)
    good = joints[success]    # only the feasible configs

    # ── full trajectory to one EE goal, sampled at 20 Hz ───────────────────────
    goal = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)   # robot frame
    traj, ok, info = client.solve_traj(goal, meshes, transformations, hz=20.0)
    # traj : (T, 7) float32 joint positions, one row per 1/hz tick
    # ok   : bool (did planning succeed)
    # info : {"dt", "hz", "n_waypoints", "duration_s", "times", "elapsed_s"}
    #
    # Replay: stream traj rows at `hz`, e.g.
    #   for q in traj:
    #       robot.command_joint_position(q)
    #       time.sleep(info["dt"])
"""

import pickle

import numpy as np
import zmq

from .config import IK

# GLB export of a 100k+-face perception mesh costs ~seconds and the BO loop
# resends the SAME meshes every round — export each mesh once and reuse the
# bytes. Also keeps the payload byte-identical across calls, so the server's
# scene-digest solver cache always hits. The bytes are cached ON the mesh
# (trimesh's metadata dict) — same key as semantic_grasp.isaac, so the two
# clients share one export — and live and die with the object: a long-running
# server that builds fresh meshes per request can never be handed a previous
# scene's bytes (a global dict keyed by id() did exactly that once ids were
# reused), and nothing accumulates. Meshes are never mutated after reconstruct_3d.
_GLB_KEY = "_glb_bytes"


def _glb_bytes(mesh) -> bytes:
    b = mesh.metadata.get(_GLB_KEY)
    if b is None:
        b = bytes(mesh.export(file_type="glb"))
        mesh.metadata[_GLB_KEY] = b
    return b


class IKClient:
    def __init__(self, addr: str = IK, timeout_s: float = 300.0):
        self.addr = addr
        self.timeout_ms = int(timeout_s * 1000)
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)       # don't block on close
        self.sock.connect(self.addr)

    # ── build a scene object whose geometry rides along in the request ─────────
    @staticmethod
    def mesh_object(name, translation, rotation, scale, *,
                    mesh_bytes: bytes | None = None, mesh_format: str | None = None,
                    vertices=None, faces=None) -> dict:
        """Construct one collision-scene object. Geometry is embedded directly:
        give either (mesh_bytes + mesh_format) or (vertices + faces).
        translation/rotation are CAMERA frame (rotation is wxyz); they get pushed
        through the server's C2R extrinsics."""
        obj = {
            "name":        str(name),
            "translation": [float(v) for v in translation],
            "rotation":    [float(v) for v in rotation],
            "scale":       float(scale),
        }
        if mesh_bytes is not None:
            if mesh_format is None:
                raise ValueError("mesh_format is required alongside mesh_bytes")
            obj["mesh_bytes"]  = bytes(mesh_bytes)
            obj["mesh_format"] = str(mesh_format)
        elif vertices is not None and faces is not None:
            obj["vertices"] = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
            obj["faces"]    = np.ascontiguousarray(faces,    dtype=np.int64).reshape(-1, 3)
        else:
            raise ValueError("provide mesh_bytes(+mesh_format) or vertices+faces")
        return obj

    @staticmethod
    def _objects_from_scene(meshes, transformations) -> list[dict]:
        """Turn the perception outputs into the collision-scene object list the
        server wants (the wire format keys objects by a "name" field). `meshes`
        is {name: trimesh Scene/Trimesh}; `transformations` is the pipeline
        scene dict ({"objects": {name: {translation, rotation_wxyz, scale}}}).
        Each mesh is exported to GLB bytes and rides along in the request
        (nothing is read from disk by the server)."""
        objects = []
        for name, entry in transformations["objects"].items():
            mesh = meshes.get(name)
            if mesh is None:
                continue  # no reconstructed geometry for this label — skip it
            objects.append({
                "name":        str(name),
                "mesh_bytes":  _glb_bytes(mesh),
                "mesh_format": "glb",
                "translation": [float(v) for v in entry["translation"]],
                "rotation":    [float(v) for v in entry["rotation_wxyz"]],
                "scale":       float(entry["scale"]),
            })
        return objects

    def _request(self, req: dict):
        """Send one pickled request, return the unpickled reply dict. Resets the
        REQ socket and raises TimeoutError if the server doesn't answer in time."""
        self.sock.send(pickle.dumps(req))
        try:
            raw = self.sock.recv()
        except zmq.Again:
            self._connect()                       # REQ socket is now dead — reset it
            raise TimeoutError(
                f"no reply within {self.timeout_ms/1000:.0f}s "
                f"(solver/planner build can be slow on the first call)"
            )
        resp = pickle.loads(raw)
        if resp.get("status") != "ok":
            raise RuntimeError(f"server error: {resp.get('message', resp)}")
        return resp

    def solve(self, poses, meshes, transformations, batch_size: int | None = None,
              num_seeds: int | None = None, table: dict | None = None,
              gripper_sphere_scale: float | None = None):
        """Solve collision-free IK for an (N, 6) matrix of [x,y,z,roll,pitch,yaw].

        meshes          : {object_name: trimesh Scene/Trimesh} — the geometry.
        transformations : scene dict; "objects" maps each name to its robot-base
                          translation / rotation_wxyz / scale.
        Every object present in both becomes a collision obstacle (the server adds
        the table itself). Pass empty {} / {"objects": {}} for a table-only scene.
        batch_size      : max poses per GPU launch. None → server default; the
                          server sizes the solver to min(batch_size, N) anyway,
                          so only pass this to pin the solver-cache key across
                          calls whose N varies.
        gripper_sphere_scale : factor applied to the gripper's collision-sphere
                          radii (panda_hand / left+right finger). >1.0 inflates them
                          for more conservative clearance. None → server default.

        Returns (joints (N,7) float32, success (N,) bool, info dict).
        Infeasible rows of `joints` are NaN.
        """
        poses = np.ascontiguousarray(poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 6:
            raise ValueError(f"poses must be (N, 6), got {poses.shape}")

        objects = self._objects_from_scene(meshes, transformations) if transformations else []

        req = {
            # List, not a raw ndarray: a pickled numpy array carries a
            # numpy-version-specific module path (numpy._core on >=2.0) that
            # fails to unpickle in a server env on a different numpy. The server
            # does np.asarray(...) on receipt, so a list is equivalent.
            "poses":      poses.tolist(),
            "objects":    objects,
        }
        if batch_size is not None:
            req["batch_size"] = int(batch_size)
        if num_seeds is not None:
            req["num_seeds"] = int(num_seeds)
        if table is not None:
            req["table"] = table
        if gripper_sphere_scale is not None:
            req["gripper_sphere_scale"] = float(gripper_sphere_scale)

        resp = self._request(req)

        # Replies carry arrays as lists (same numpy-version story as requests);
        # rebuild them under OUR numpy.
        joints  = np.asarray(resp["joints"], dtype=np.float32).reshape(-1, 7)
        success = np.asarray(resp["success"], dtype=bool)
        info = {k: resp[k] for k in ("n_feasible", "n_total", "elapsed_s") if k in resp}
        return joints, success, info

    def solve_traj(self, pose, meshes, transformations, *,
                   start_state=None, hz: float = 50.0,
                   max_attempts: int | None = None, table: dict | None = None,
                   gripper_sphere_scale: float | None = None,
                   ignore_collisions: bool = False):
        """Plan a full collision-free joint trajectory from `start_state` to a
        single end-effector goal `pose`, sampled at `hz` for direct robot replay.

        pose            : length-6 [x, y, z, roll, pitch, yaw] in ROBOT frame —
                          where you want the end effector to end up.
        meshes          : {object_name: trimesh Scene/Trimesh} — collision geometry.
        transformations : scene dict ({"objects": {name: {translation,
                          rotation_wxyz, scale}}}, robot-base frame). Pass
                          {} / {"objects": {}} (and {} for meshes) for a
                          table-only scene.
        start_state     : length-7 start joint config; None → planner home/default.
        hz              : sampling rate of the returned trajectory (default 50 Hz,
                          which is the planner's native interpolation rate — ask
                          for more and the extra rows are pure linear
                          interpolation, carrying no new plan detail).
        gripper_sphere_scale : factor applied to the gripper's collision-sphere
                          radii (>1.0 = more conservative). None → server default.
        ignore_collisions : True → the server plans with NO world obstacles (the
                          objects AND the table are dropped from the collision
                          world). Self-collision and joint limits still apply.

        Returns (traj (T,7) float32, success bool, info dict).
          traj : joint positions spaced 1/hz apart — play these straight on the
                 robot (one row per tick). (0,7) and success=False on plan failure.
          info : {"dt", "hz", "n_waypoints", "duration_s", "times" (T,), "elapsed_s"}.
        """
        pose = np.ascontiguousarray(pose, dtype=np.float32).reshape(-1)
        if pose.shape[0] != 6:
            raise ValueError(f"pose must be length-6 [x,y,z,r,p,y], got {pose.shape}")

        objects = self._objects_from_scene(meshes, transformations) if transformations else []

        req = {
            "cmd":     "solve_traj",
            "pose":    pose.tolist(),     # list, not ndarray — see note in solve()
            "objects": objects,
            "hz":      float(hz),
        }
        if start_state is not None:
            start_state = np.ascontiguousarray(start_state, dtype=np.float32).reshape(-1)
            if start_state.shape[0] != 7:
                raise ValueError(f"start_state must be length-7, got {start_state.shape}")
            req["start_state"] = start_state.tolist()
        if max_attempts is not None:
            req["max_attempts"] = int(max_attempts)
        if table is not None:
            req["table"] = table
        if gripper_sphere_scale is not None:
            req["gripper_sphere_scale"] = float(gripper_sphere_scale)
        if ignore_collisions:
            req["ignore_collisions"] = True

        resp = self._request(req)

        # reshape(-1, 7) keeps the documented (0, 7) shape on plan failure.
        traj    = np.asarray(resp["positions"], dtype=np.float32).reshape(-1, 7)
        success = bool(resp["success"])
        info = {k: resp[k] for k in
                ("dt", "hz", "n_waypoints", "duration_s", "times", "elapsed_s")
                if k in resp}
        if "times" in info:
            info["times"] = np.asarray(info["times"], dtype=np.float32)
        return traj, success, info

    def penetration(self, joints, meshes, transformations, *,
                    activation_distance: float = 0.0, table: dict | None = None,
                    gripper_sphere_scale: float | None = None):
        """Total world-collision penetration of the robot at each joint config.

        For each config the server runs FK to get the robot's collision spheres
        (the same sphere model IK / the planner collision-check against, incl. the
        inflated gripper spheres) and sums how far they poke into the scene
        obstacles (table + objects). Robot-vs-SCENE only — NOT self-collision.

        joints          : (N, 7) arm configs (e.g. the IK solutions from solve()).
                          A single length-7 config is accepted and reshaped to (1,7).
        meshes / transformations : same scene inputs as solve()/solve_traj().
        activation_distance : 0.0 → pure penetration depth (0 when collision-free).
                          >0 also ramps the cost up within that margin OUTSIDE the
                          obstacle, giving a soft clearance signal.
        gripper_sphere_scale : factor applied to the gripper's collision-sphere
                          radii (>1.0 = fatter spheres → larger penetration near
                          obstacles). None → server default.

        Returns (penetration (N,) float32, in_collision (N,) bool, info dict).
          penetration : 0.0 = clear, larger = deeper; NaN for non-finite input rows.
          in_collision: penetration > 0 (and finite).
          info        : {"n_total", "elapsed_s"}.
        """
        joints = np.ascontiguousarray(joints, dtype=np.float32)
        if joints.ndim == 1:
            joints = joints.reshape(1, -1)
        if joints.ndim != 2 or joints.shape[1] != 7:
            raise ValueError(f"joints must be (N,7), got {joints.shape}")

        objects = self._objects_from_scene(meshes, transformations) if transformations else []

        req = {
            "cmd":                 "penetration",
            "joints":              joints.tolist(),   # list, not ndarray — see note in solve()
            "objects":             objects,
            "activation_distance": float(activation_distance),
        }
        if table is not None:
            req["table"] = table
        if gripper_sphere_scale is not None:
            req["gripper_sphere_scale"] = float(gripper_sphere_scale)

        resp = self._request(req)

        penetration  = np.asarray(resp["penetration"], dtype=np.float32)   # (N,)
        in_collision = np.asarray(resp["in_collision"], dtype=bool)        # (N,)
        info = {k: resp[k] for k in ("n_total", "elapsed_s") if k in resp}
        return penetration, in_collision, info

    def visualize(self, meshes, transformations, *, points=None,
                  show_spheres: bool = False, joint_config=None,
                  port: int | None = None, host: str | None = None,
                  table: dict | None = None,
                  gripper_sphere_scale: float | None = None):
        """Launch (or refresh) a Viser web view of the cuRobo collision scene.

        meshes / transformations : same scene inputs as solve()/solve_traj(); the
                          server builds the identical collision scene. Pass
                          {} / {"objects": {}} for a table-only scene.
        show_spheres    : draw the robot's collision-sphere model — the exact sphere
                          decomposition IK collision-checks against (incl. the
                          gripper spheres on panda_hand / leftfinger / rightfinger).
        points          : optional (N,3) ROBOT-frame points scattered as red
                          icospheres — e.g. IK targets `poses[:, :3]` or grasp
                          candidates.
        joint_config    : optional length-7 arm config to pose the robot at (so the
                          spheres sit at a real configuration rather than the default).
        gripper_sphere_scale : factor applied to the drawn gripper collision-sphere
                          radii (matches what solve() collision-checks against).
                          None → server default. Changing it rebuilds the viewer.
        port / host     : where Viser serves. None → the server's default (8082,
                          or its --port flag). Open http://<host>:<port> in a browser.

        Returns the server ack: {"status":"ok","url":..,"show_spheres":..,
        "n_objects":..,"n_points":..}. Note: toggling show_spheres or changing the
        port rebuilds the viewer (the flag is set at construction).
        """
        objects = self._objects_from_scene(meshes, transformations) if transformations else []

        req = {
            "cmd":          "visualize",
            "objects":      objects,
            "show_spheres": bool(show_spheres),
        }
        if port is not None:
            req["port"] = int(port)
        if host is not None:
            req["host"] = str(host)
        if points is not None:
            pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
            req["points"] = pts.tolist()
        if joint_config is not None:
            q = np.ascontiguousarray(joint_config, dtype=np.float32).reshape(-1)
            if q.shape[0] != 7:
                raise ValueError(f"joint_config must be length-7, got {q.shape}")
            req["joint_config"] = q.tolist()
        if table is not None:
            req["table"] = table
        if gripper_sphere_scale is not None:
            req["gripper_sphere_scale"] = float(gripper_sphere_scale)

        return self._request(req)

    def viz_traj(self, pose, meshes, transformations, *, points=None,
                 show_spheres: bool = True, start_state=None, hz: float = 20.0,
                 max_attempts: int | None = None, port: int | None = None,
                 host: str | None = None, table: dict | None = None,
                 gripper_sphere_scale: float | None = None,
                 ignore_collisions: bool = False):
        """Plan a collision-free trajectory to `pose` and loop-play it in the
        Viser viewer at http://<host>:<port>.

        pose            : length-6 [x,y,z,roll,pitch,yaw] EE goal, ROBOT frame.
        meshes / transformations : same scene inputs as solve_traj()/visualize().
        points          : optional (N,3) ROBOT-frame markers (e.g. grasp candidates).
        show_spheres    : draw the robot's collision-sphere model.
        start_state     : length-7 start config; None → planner default.
        hz              : sampling rate for the played trajectory (default 20).
        gripper_sphere_scale : factor applied to the gripper collision-sphere radii
                          (planning + drawn spheres). None → server default.
        ignore_collisions : True → plan with NO world obstacles (objects + table
                          dropped from the planner's collision world; self-
                          collision and joint limits still apply). The viewer
                          still draws the real objects, so you can watch the
                          trajectory sweep through them.
        port / host     : where Viser serves. None → the server's default (8082,
                          or its --port flag).

        Returns the server ack: {"status":"ok","url":..,"n_waypoints":..,"hz":..,
        "dt":..,"n_objects":..,"n_points":..,"elapsed_s":..}.
        """
        pose = np.ascontiguousarray(pose, dtype=np.float32).reshape(-1)
        if pose.shape[0] != 6:
            raise ValueError(f"pose must be length-6 [x,y,z,r,p,y], got {pose.shape}")

        objects = self._objects_from_scene(meshes, transformations) if transformations else []

        req = {
            "cmd":          "viz_traj",
            "pose":         pose.tolist(),
            "objects":      objects,
            "hz":           float(hz),
            "show_spheres": bool(show_spheres),
        }
        if port is not None:
            req["port"] = int(port)
        if host is not None:
            req["host"] = str(host)
        if points is not None:
            pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
            req["points"] = pts.tolist()
        if start_state is not None:
            start_state = np.ascontiguousarray(start_state, dtype=np.float32).reshape(-1)
            if start_state.shape[0] != 7:
                raise ValueError(f"start_state must be length-7, got {start_state.shape}")
            req["start_state"] = start_state.tolist()
        if max_attempts is not None:
            req["max_attempts"] = int(max_attempts)
        if table is not None:
            req["table"] = table
        if gripper_sphere_scale is not None:
            req["gripper_sphere_scale"] = float(gripper_sphere_scale)
        if ignore_collisions:
            req["ignore_collisions"] = True

        return self._request(req)

    def ping(self) -> bool:
        self.sock.send(pickle.dumps({"cmd": "ping"}))
        try:
            resp = pickle.loads(self.sock.recv())
        except zmq.Again:
            self._connect()
            return False
        return resp.get("status") == "ok"

    def close(self):
        if self.sock is not None:
            self.sock.close(0)


def _unit_cube(half=0.03):
    """A tiny axis-aligned cube as (vertices, faces) — self-contained demo geometry,
    so the demo exercises the in-request path without needing any mesh file."""
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float32) * half
    f = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int64)
    return v, f


if __name__ == "__main__":
    # Minimal demo: one in-memory box obstacle, passed the same way the pipeline
    # does — a {name: mesh} dict + a name-keyed scene dict.
    import trimesh

    client = IKClient()
    print("ping:", client.ping())

    meshes = {"cube": trimesh.creation.box(extents=(0.06, 0.06, 0.06))}
    transformations = {"objects": {
        "cube": {"translation": [0.45, 0.0, 0.10],
                 "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "scale": 1.0},
    }}

    # ── batch IK: 200 random poses ─────────────────────────────────────────────
    poses = np.zeros((200, 6), dtype=np.float32)
    poses[:, :3] = np.random.uniform([0.30, -0.30, 0.10],
                                     [0.60,  0.30, 0.50], (200, 3))
    poses[:, 3:] = np.random.uniform(-np.pi, np.pi, (200, 3))

    joints, success, info = client.solve(poses, meshes, transformations)
    print(info)
    print("feasible joints (first 3):\n", joints[success][:3])

    # ── full trajectory: plan to a single EE goal, sampled at 20 Hz for replay ──
    goal = np.array([0.45, 0.10, 0.35, np.pi, 0.0, 0.0], dtype=np.float32)
    traj, ok, tinfo = client.solve_traj(goal, meshes, transformations, hz=20.0)
    if ok:
        print(f"trajectory: {tinfo['n_waypoints']} waypoints @ {tinfo['hz']:.0f}Hz "
              f"({tinfo['duration_s']}s), dt={tinfo['dt']}")
        print("first 3 waypoints:\n", traj[:3])
        # Replay would be: for q in traj: robot.command(q); time.sleep(tinfo["dt"])
    else:
        print("trajectory planning failed (goal unreachable / in collision)")

    # ── visualize the scene + robot collision spheres in a browser ──────────────
    viz = client.visualize(
        meshes, transformations,
        points=poses[success][:, :3],   # scatter the feasible IK targets
        show_spheres=True,              # draw the robot's collision-sphere model
        joint_config=joints[success][0] if success.any() else None,
        port=8082,
    )
    print(f"viewer: {viz['url']}  (spheres={viz['show_spheres']}, "
          f"objects={viz['n_objects']}, points={viz['n_points']})")

    client.close()