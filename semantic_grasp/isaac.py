#!/usr/bin/env python3
"""Client for the Isaac Lab grasp-evaluation ZMQ server (pickle I/O, in-request mesh).

Hands the server an (N, 7) matrix of Franka joint angles plus the scene the grasps
are evaluated against — the target's name, its mesh (sent as bytes; it is never read
from disk), and its robot-base-frame pose/scale — and returns per-grasp lift/stability
metrics.

    from grasp_client import GraspClient
    import numpy as np

    client = GraspClient("tcp://localhost:5556")

    joint_candidates = np.zeros((100, 7), dtype=np.float32)
    joint_candidates[:, :] = ...

    # meshes / transformations are the reconstruct_3d outputs, passed straight
    # through; the target's GLB bytes + robot-base-frame pose/scale are pulled out here.
    results, info = client.evaluate(
        joint_candidates,
        target="green bowl",
        meshes=meshes,                   # {name: trimesh Scene/Trimesh}
        transformations=transformations, # scene dict (objects keyed by name)
    )
    # results : list of dicts with "lift_distance", "passed", orientation offsets, etc.
    # info    : {"n_total": 100, "n_passed": 12, "elapsed_s": ..., "restarts": 0}

Every joint set is run through the full pipeline: each set is snapped, the gripper is
closed, the object is lifted, and per-grasp metrics come back.

Live rollout
------------
`rollout()` starts a looping visual demo on the server: it loops
snap -> close -> lift -> hold -> reset for the first joint row FOREVER, until
you kill the server (Ctrl-C) or close the viewer. Launch the server with
--render to watch it locally. Same inputs as evaluate():

    client.rollout(
        joint_candidates,            # (N,7); row 0 is demoed
        target="green bowl",
        meshes=meshes,
        transformations=transformations,
        hold_steps=60,               # ~1s hold at the top of each lift
    )
    # blocks until the server's loop ends (i.e. until you kill it)

Scene switching
---------------
The server builds one scene at a time. When a request's scene inputs (target / mesh
bytes / pose / scale) differ from what's currently loaded, it re-execs into a fresh
process for a clean CUDA/PhysX init and replies {"status": "restarting", "retry": true}.
This client absorbs that automatically: it waits for the new process and resends. So
switching scenes makes that one call take the ~30-60s Isaac startup; repeated calls
against the same scene are fast.
"""

import time
import pickle
import numpy as np
import zmq

from .config import GRASP

# GLB export of a 100k+-face perception mesh costs ~seconds and the BO loop
# resends the SAME meshes every round — export each mesh once and reuse the
# bytes. Byte-identical payloads also keep the server's scene identity stable
# (no spurious re-execs). The bytes are cached ON the mesh (trimesh's metadata
# dict), so the cache lives and dies with the object: a long-running server
# that builds fresh meshes per request can never be handed a previous scene's
# bytes (a global dict keyed by id() did exactly that once ids were reused),
# and nothing accumulates. Meshes are never mutated after reconstruct_3d.
_GLB_KEY = "_glb_bytes"


def _glb_bytes(mesh) -> bytes:
    b = mesh.metadata.get(_GLB_KEY)
    if b is None:
        b = bytes(mesh.export(file_type="glb"))
        mesh.metadata[_GLB_KEY] = b
    return b


class GraspClient:
    def __init__(
        self,
        addr: str = GRASP,
        timeout_s: float = 600.0,
        restart_grace_s: float = 5.0,
        max_restarts: int = 3,
    ):
        """
        Args:
            addr: ZMQ address of the server (server default port is 5556).
            timeout_s: Seconds to wait for a single reply before timing out.
                       Also covers the new server's startup after a scene switch,
                       so keep it comfortably larger than Isaac's launch time.
            restart_grace_s: Short pause after a restart reply to let the old
                       process release the port before the new one binds.
            max_restarts: Safety cap on consecutive scene-switch restarts for a
                       single evaluate() call.
        """
        self.addr = addr
        self.timeout_ms = int(timeout_s * 1000)
        self.restart_grace_s = restart_grace_s
        self.max_restarts = max_restarts
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()

    def _connect(self):
        """Create/reconnect a clean REQ socket, resetting state if necessary."""
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)  # don't block context termination on close
        self.sock.connect(self.addr)

    @staticmethod
    def _scene_for_target(target, meshes, transformations) -> dict:
        """Pull the target's GLB bytes + robot-base-frame pose/scale out of the
        reconstruct_3d outputs. `meshes` is {name: trimesh Scene/Trimesh};
        `transformations` is the pipeline scene dict whose "objects" map each
        name to its robot-base-frame translation / rotation_wxyz / scale. The
        grasp server evaluates one object at a time, so only the target is
        extracted. The mesh is exported to GLB bytes and rides along in the
        request."""
        entry = transformations["objects"].get(target)
        if entry is None:
            raise ValueError(f"target {target!r} not in transformations['objects']")
        mesh = meshes.get(target)
        if mesh is None:
            raise ValueError(f"no reconstructed mesh for target {target!r}")
        return {
            "mesh_bytes":  _glb_bytes(mesh),
            "mesh_format": "glb",
            "translation": [float(v) for v in entry["translation"]],
            "rotation":    [float(v) for v in entry["rotation_wxyz"]],
            "scale":       float(entry["scale"]),
        }

    @staticmethod
    def _extra_objects_payload(extra_objects, target, meshes, transformations):
        """Settle-style wire entries for the extra objects spawned in the eval
        world beside the grasped target (the clearance-promoted neighbors from
        reconstruct_3d). Names are deduped and the target itself is skipped;
        poses come from `transformations` (a single scene dict)."""
        entries = []
        for name in dict.fromkeys(extra_objects):
            if name == target:
                continue
            entry = transformations["objects"].get(name)
            if entry is None:
                raise ValueError(f"extra object {name!r} not in transformations['objects']")
            mesh = meshes.get(name)
            if mesh is None:
                raise ValueError(f"no reconstructed mesh for extra object {name!r}")
            entries.append({
                "name":        name,
                "mesh_bytes":  _glb_bytes(mesh),
                "mesh_format": "glb",
                "translation": [float(v) for v in entry["translation"]],
                "rotation":    [float(v) for v in entry["rotation_wxyz"]],
                "scale":       float(entry["scale"]),
            })
        return entries

    def evaluate(self, joint_positions, target: str, meshes, transformations,
                 extra_objects=None):
            """Send an (N, 7) array of joint vectors + the scene to evaluate against.

            Geometry/pose travels straight from perception: hand in the `reconstruct_3d`
            outputs (`meshes`, `transformations`) and the target's GLB bytes and
            robot-base-frame pose/scale are extracted here (mesh is never read from disk).

            Args:
                joint_positions: (N, 7) array of float32 Franka joint positions. A
                            single (7,) vector is also accepted and treated as one grasp.
                target:      Object name (label); selects which object in the scene to grasp.
                meshes:      {object_name: trimesh Scene/Trimesh} from reconstruct_3d.
                transformations: a scene dict (its "objects" map each name to the
                            robot-base-frame translation / rotation_wxyz / scale), OR a LIST
                            of such dicts (candidate scenes). With a list, every candidate
                            runs in parallel as its own env (one target pose each) in a
                            single rollout; pass one grasp row (broadcast to all) or one
                            row per candidate. All candidates must share the target mesh.
                extra_objects: optional list of object NAMES (from `meshes` /
                            `transformations`) to spawn in the eval world beside the
                            target — the clearance-promoted neighbors from
                            reconstruct_3d. When `transformations` is a candidate
                            list, each neighbor is placed per-env at its pose in that
                            candidate (sent as `extra_scene_poses`), so it rides the
                            target's domain randomization as a rigid group; with a
                            single scene dict it just sits at its settled pose.
                            Changing this set changes the scene identity (one
                            ~30-60s rebuild); the per-env poses do not.

            Returns:
                results: list of dicts. Each dict has:
                    - 'index' (int)
                    - 'passed' (bool)
                    - 'lift_distance' (float, meters; object Z delta nominal -> post-lift)
                    - 'roll_offset', 'pitch_offset', 'yaw_offset'
                        (floats, degrees; object orientation drift across the lift)
                    - 'gripper_width_diff' (float, meters)
                    - 'pre_close_ee_pos',  'pre_close_ee_rpy'   ([x,y,z] / [r,p,y] rad)
                    - 'pre_close_obj_pos', 'pre_close_obj_rpy'
                    - 'pre_lift_ee_pos',   'pre_lift_ee_rpy'    (post-close, pre-lift;
                        slip/robustness baseline — object already seated in gripper)
                    - 'pre_lift_obj_pos',  'pre_lift_obj_rpy'
                    - 'post_lift_ee_pos',  'post_lift_ee_rpy'
                    - 'post_lift_obj_pos', 'post_lift_obj_rpy'
                    - 'extra_lift_distances' (list of floats, meters; z rise of
                        each extra_objects neighbor across the lift, in the
                        order they were passed — empty without extra_objects)
                    All XYZ are in Isaac's world frame; all RPY are radians,
                    extrinsic xyz (SciPy's lowercase 'xyz' convention).
                    Failed rows ('passed': False) carry the same keys filled with
                    NaNs and a 'fail_reason' string
                    ('nan_input', 'nan_in_sim', or 'interpenetration_on_snap').
                info: dict with
                    {'n_total', 'n_passed', 'n_nan', 'n_interpen',
                    'elapsed_s', 'restarts'}.
            """
            # ── Input sanity ─────────────────────────────────────────────────────
            if not isinstance(joint_positions, np.ndarray):
                joint_positions = np.array(joint_positions, dtype=np.float32)
            joint_positions = np.ascontiguousarray(joint_positions, dtype=np.float32)
            # Allow a single (7,) joint vector — treated as a single grasp.
            if joint_positions.ndim == 1 and joint_positions.shape[0] == 7:
                joint_positions = joint_positions.reshape(1, 7)
            if joint_positions.ndim != 2 or joint_positions.shape[1] != 7:
                raise ValueError(f"joint_positions must be (N, 7), got {joint_positions.shape}")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("target must be a non-empty object name string")

            # `transformations` may be a single scene dict (evaluate N grasps
            # against one pose) OR a list of candidate scene dicts to run in
            # parallel — one env per candidate, each with its own target pose.
            # In the list case the geometry (mesh/scale + the reference pose that
            # keys the built scene) comes from the first candidate, since every
            # candidate shares the same target mesh; a per-env `scene_poses` list
            # carries the per-candidate target pose, and a single grasp row is
            # broadcast across all candidates.
            scene_poses = None
            extra_scene_poses = None
            ref_scene = transformations      # single dict: build geometry + poses
            # Extra (clearance-promoted) neighbor names, in the SAME order
            # _extra_objects_payload builds them, so per-candidate poses line up.
            extra_names = ([n for n in dict.fromkeys(extra_objects) if n != target]
                           if extra_objects else [])
            if isinstance(transformations, (list, tuple)):
                candidates = list(transformations)
                if not candidates:
                    raise ValueError("evaluate: `transformations` list is empty")
                # candidates[0] is the untouched base: it keys the built scene and
                # supplies the extras' build-time geometry/pose.
                ref_scene = candidates[0]
                scene = self._scene_for_target(target, meshes, candidates[0])
                scene_poses = []
                extra_scene_poses = [] if extra_names else None
                for ci, cand in enumerate(candidates):
                    entry = cand["objects"].get(target)
                    if entry is None:
                        raise ValueError(
                            f"evaluate: target {target!r} not in candidate scene {ci}"
                        )
                    scene_poses.append({
                        "translation": [float(v) for v in entry["translation"]],
                        "rotation":    [float(v) for v in entry["rotation_wxyz"]],
                    })
                    # Neighbors ride the target's randomization (rigid group): one
                    # pose per extra per candidate, ordered to match extra_objects.
                    if extra_names:
                        per_scene = []
                        for n in extra_names:
                            e = cand["objects"].get(n)
                            if e is None:
                                raise ValueError(
                                    f"evaluate: extra object {n!r} not in candidate scene {ci}"
                                )
                            per_scene.append({
                                "translation": [float(v) for v in e["translation"]],
                                "rotation":    [float(v) for v in e["rotation_wxyz"]],
                            })
                        extra_scene_poses.append(per_scene)
                # Pair grasps with scenes 1:1; a single grasp is broadcast to all.
                if joint_positions.shape[0] == 1:
                    joint_positions = np.repeat(joint_positions, len(scene_poses), axis=0)
                elif joint_positions.shape[0] != len(scene_poses):
                    raise ValueError(
                        f"evaluate: joints rows ({joint_positions.shape[0]}) must be 1 "
                        f"or match the number of candidate scenes ({len(scene_poses)})"
                    )
            else:
                # Extract the target's GLB bytes + robot-base-frame pose/scale.
                scene = self._scene_for_target(target, meshes, transformations)

            # Pre-flight: catch an empty mesh before paying for a full ZMQ round trip.
            if not scene.get("mesh_bytes"):
                raise ValueError(
                    f"_scene_for_target('{target}') returned empty mesh_bytes; "
                    f"check that '{target}' exists in `meshes` and has geometry."
                )
            if not scene.get("mesh_format"):
                raise ValueError(
                    f"_scene_for_target('{target}') returned no mesh_format."
                )

            payload = {
                "target":      target,
                "mesh_bytes":  scene["mesh_bytes"],
                "mesh_format": scene["mesh_format"],
                "translation": scene["translation"],
                "rotation":    scene["rotation"],
                "scale":       scene["scale"],
                # Send as a nested list, NOT a raw ndarray: a pickled numpy array
                # carries a numpy-version-specific module path (numpy._core on >=2.0),
                # which fails to unpickle if the server's env is on a different numpy.
                # The server does np.asarray(...) on receipt, so a list is equivalent.
                "joints":      joint_positions.tolist(),
            }
            if scene_poses is not None:
                # Per-env target poses -> parallel candidate scenes, one rollout.
                payload["scene_poses"] = scene_poses
            if extra_scene_poses is not None:
                # Per-env neighbor poses so the extras move WITH the target
                # (rigid group randomization) instead of sitting fixed.
                payload["extra_scene_poses"] = extra_scene_poses
            if extra_objects:
                payload["extra_objects"] = self._extra_objects_payload(
                    extra_objects, target, meshes, ref_scene)
            msg = pickle.dumps(payload)

            start_time = time.time()
            restarts = 0

            # ── Send / receive, absorbing scene-switch restarts ──────────────────
            while True:
                self.sock.send(msg)
                try:
                    raw = self.sock.recv()
                except zmq.Again:
                    self._connect()  # REQ state machine is locked out; recycle it
                    raise TimeoutError(
                        f"No reply from the Isaac Lab server within {self.timeout_ms/1000:.0f}s. "
                        f"Check the server is up at {self.addr} and hasn't hit an Omni crash."
                    )

                response_data = pickle.loads(raw)

                # Server is re-exec'ing for a new scene — wait for it and resend.
                if response_data.get("retry"):
                    new_t = response_data.get("target", target)
                    if restarts >= self.max_restarts:
                        raise RuntimeError(
                            f"Server kept restarting (>{self.max_restarts}) while loading "
                            f"target '{new_t}'. Check the inputs and server logs."
                        )
                    restarts += 1
                    print(f"[client] Server re-exec'ing for target '{new_t}' "
                        f"(restart {restarts}/{self.max_restarts}); waiting for it to come back...")
                    time.sleep(self.restart_grace_s)
                    self._connect()  # force a clean reconnect to the new process
                    continue         # resend the same payload

                break  # got a real reply

            elapsed = time.time() - start_time

            if "error" in response_data:
                raise RuntimeError(f"Server execution error: {response_data['error']}")

            results = response_data.get("results", [])

            n_total    = len(results)
            n_passed   = sum(1 for r in results if r.get("passed", False))
            n_nan      = sum(1 for r in results
                            if r.get("fail_reason") in ("nan_input", "nan_in_sim"))
            n_interpen = sum(1 for r in results
                            if r.get("fail_reason") == "interpenetration_on_snap")
            info = {
                "n_total":    n_total,
                "n_passed":   n_passed,
                "n_nan":      n_nan,
                "n_interpen": n_interpen,
                "elapsed_s":  round(elapsed, 3),
                "restarts":   restarts,
            }
            return results, info

    def load_meshes(self, target: str, meshes, transformations, extra_objects=None):
        """Pre-build the grasp/eval world for `target` and keep it hot — WITHOUT
        blocking on the build.

        Sends the same scene the next evaluate() will key on ('load' shares the
        'try' scene identity server-side), so the server re-execs and builds NOW
        — overlapping the pipeline's VLM stages — and the later evaluate() finds
        the scene live instead of paying the rebuild on its own critical path.
        Fire this right after settle(); the settled transforms are what
        evaluate's candidate scenes are generated from, so the identities match.

        `transformations` may be the scene dict or a candidates list (the first
        candidate keys the scene, matching evaluate's convention).

        `extra_objects` is the same list of neighbor names the later evaluate()
        will pass — it is part of the scene identity, so pass it HERE too or the
        evaluate lands on a mismatched world and pays the rebuild anyway.

        Returns {"status": "restarting"} when the build was just kicked off (the
        usual case) or {"status": "ready"} when the scene was already live.
        Raises on server validation errors."""
        if isinstance(transformations, (list, tuple)):
            if not transformations:
                raise ValueError("load_meshes: `transformations` list is empty")
            transformations = transformations[0]
        scene = self._scene_for_target(target, meshes, transformations)
        payload = {
            "mode":        "load",
            "target":      target,
            "mesh_bytes":  scene["mesh_bytes"],
            "mesh_format": scene["mesh_format"],
            "translation": scene["translation"],
            "rotation":    scene["rotation"],
            "scale":       scene["scale"],
        }
        if extra_objects:
            payload["extra_objects"] = self._extra_objects_payload(
                extra_objects, target, meshes, transformations)
        self.sock.send(pickle.dumps(payload))
        try:
            raw = self.sock.recv()
        except zmq.Again:
            self._connect()  # REQ state machine is locked out; recycle it
            raise TimeoutError(
                f"No reply from the Isaac Lab server within {self.timeout_ms/1000:.0f}s "
                f"for load_meshes. Check the server is up at {self.addr}."
            )
        reply = pickle.loads(raw)
        if "error" in reply:
            raise RuntimeError(f"load_meshes: server error: {reply['error']}")
        if reply.get("retry"):
            # Scene switch kicked off: the server replied "restarting" and is
            # re-exec'ing; the new process builds the world at startup. Do NOT
            # wait for it — reconnect so the socket is clean for the next call
            # and let the build overlap the caller's other work. A later
            # request for the same scene lands on the built world (or queues
            # in ZMQ until the new process binds).
            self._connect()
            print(f"[client] load_meshes: server building grasp world for "
                  f"'{target}' in the background")
            return {"status": "restarting", "target": target}
        print(f"[client] load_meshes: grasp world for '{target}' already live")
        return {"status": "ready", "target": target}

    def settle(self, meshes, transformations, settle_steps=None):
        """Run physics settling on the full multi-object scene and return updated
        transforms in the same robot-base-frame format as `transformations`.

        For each object in `transformations["objects"]`, the matching mesh is
        exported to GLB bytes and sent to the server. The server builds a single
        scene with the Franka at home + all objects as dynamic rigid bodies,
        steps physics for `settle_steps` (server default if omitted), then
        returns each object's new (translation, rotation_wxyz) in the
        robot-base frame.

        Args:
            meshes:           {object_name: trimesh Scene/Trimesh}.
            transformations:  scene dict ({"objects": {name: {translation,
                              rotation_wxyz, scale}}}, robot-base frame).
            settle_steps:     Optional override for the number of physics steps.

        Returns:
            new_transformations: dict in the same shape as `transformations`,
                                 with each object's translation/rotation_wxyz
                                 replaced by the settled pose (scale unchanged).
                                 Original object ordering is preserved.
            info: {"elapsed_s", "restarts", "n_settled"}.
        """
        if "objects" not in transformations or not transformations["objects"]:
            raise ValueError("transformations['objects'] is empty")

        objects_payload = []
        for name, entry in transformations["objects"].items():
            mesh = meshes.get(name)
            if mesh is None:
                raise ValueError(f"no reconstructed mesh for object {name!r}")
            objects_payload.append({
                "name":        name,
                "mesh_bytes":  _glb_bytes(mesh),
                "mesh_format": "glb",
                "translation": [float(v) for v in entry["translation"]],
                "rotation":    [float(v) for v in entry["rotation_wxyz"]],
                "scale":       float(entry["scale"]),
            })

        payload = {"mode": "settle", "objects": objects_payload}
        if settle_steps is not None:
            payload["settle_steps"] = int(settle_steps)
        msg = pickle.dumps(payload)

        start_time = time.time()
        restarts = 0

        while True:
            self.sock.send(msg)
            try:
                raw = self.sock.recv()
            except zmq.Again:
                self._connect()
                raise TimeoutError(
                    f"No reply from the Isaac Lab server within {self.timeout_ms/1000:.0f}s. "
                    f"Check the server is up at {self.addr} and hasn't hit an Omni crash."
                )

            response_data = pickle.loads(raw)
            if response_data.get("retry"):
                if restarts >= self.max_restarts:
                    raise RuntimeError(
                        f"Server kept restarting (>{self.max_restarts}) while loading "
                        f"settle scene. Check the inputs and server logs."
                    )
                restarts += 1
                print(f"[client] Server re-exec'ing for settle scene "
                      f"(restart {restarts}/{self.max_restarts}); waiting for it to come back...")
                time.sleep(self.restart_grace_s)
                self._connect()
                continue
            break

        elapsed = time.time() - start_time

        if "error" in response_data:
            raise RuntimeError(f"Server execution error: {response_data['error']}")

        results = response_data.get("results", [])
        by_name = {r["name"]: r for r in results}

        # Build the new transformations dict, preserving the original ordering
        # and any extra fields on each entry (glb path, etc.).
        new_objects = {}
        for name, entry in transformations["objects"].items():
            new_entry = dict(entry)
            if name in by_name:
                r = by_name[name]
                new_entry["translation"]   = list(r["translation"])
                new_entry["rotation_wxyz"] = list(r["rotation"])
            new_objects[name] = new_entry
        new_transformations = dict(transformations)
        new_transformations["objects"] = new_objects

        info = {
            "elapsed_s": round(elapsed, 3),
            "restarts":  restarts,
            "n_settled": len(results),
        }
        return new_transformations, info

    def rollout(self, joint_positions, target, meshes, transformations,
                hold_steps=60):
        """Start a looping visual rollout on the server and block until it stops.

        The server loops snap -> close -> lift -> hold -> reset for the FIRST row
        of `joint_positions`, FOREVER, until you kill the server process
        (Ctrl-C) or close the viewer window. Launch the server with --render so
        you can watch it locally.

        The rollout now spawns the FULL scene — every object in `transformations`
        is sent and built — and `target` selects which one the robot actually
        grasps and lifts. The rest are present as surrounding clutter.

        Same inputs as evaluate(): an (N,7) joints array (only the first row is
        used) plus the perception outputs.

        NOTE: this call blocks for the entire life of the rollout — the server
        does not reply until its loop ends. The receive timeout is disabled for
        this call so it won't spuriously time out; stop the demo by killing the
        server. When the server's loop ends it returns a small summary.

        Args:
            joint_positions: (N, 7) float32 Franka joint positions, or a single
                             (7,) vector. Only the first row is used.
            target:          object name to grasp.
            meshes:          {object_name: trimesh Scene/Trimesh} from reconstruct_3d.
            transformations: scene dict ({"objects": {name: entry}},
                             robot-base-frame poses/scale).
            hold_steps:      sim steps to hold at the top of each lift (~steps/60 s).

        Returns:
            result: {"rollout": True, "cycles_completed": int} (after the server stops).
            info:   {"elapsed_s", "restarts"}.
        """
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty object name string")

        # Same joint-shape handling as evaluate().
        if not isinstance(joint_positions, np.ndarray):
            joint_positions = np.array(joint_positions, dtype=np.float32)
        joint_positions = np.ascontiguousarray(joint_positions, dtype=np.float32)
        if joint_positions.ndim == 1 and joint_positions.shape[0] == 7:
            joint_positions = joint_positions.reshape(1, 7)
        if joint_positions.ndim != 2 or joint_positions.shape[1] != 7:
            raise ValueError(f"joint_positions must be (N, 7), got {joint_positions.shape}")

        # Build the FULL multi-object scene payload (same shape settle() sends):
        # every object's GLB bytes + robot-base-frame pose/scale. `target` then tells
        # the server which of these to grasp/lift.
        if "objects" not in transformations or not transformations["objects"]:
            raise ValueError("transformations['objects'] is empty")
        if target not in transformations["objects"]:
            raise ValueError(
                f"rollout target {target!r} not in transformations['objects'] "
                f"{list(transformations['objects'])}"
            )

        objects_payload = []
        for name, entry in transformations["objects"].items():
            mesh = meshes.get(name)
            if mesh is None:
                raise ValueError(f"no reconstructed mesh for object {name!r}")
            objects_payload.append({
                "name":        name,
                "mesh_bytes":  _glb_bytes(mesh),
                "mesh_format": "glb",
                "translation": [float(v) for v in entry["translation"]],
                "rotation":    [float(v) for v in entry["rotation_wxyz"]],
                "scale":       float(entry["scale"]),
            })

        payload = {
            "mode":        "rollout",
            "target":      target,
            "objects":     objects_payload,
            # Nested list, not ndarray (numpy-version pickle portability) — server
            # does np.asarray() on receipt. Same as evaluate().
            "joints":      joint_positions.tolist(),
            "hold_steps":  int(hold_steps),
        }
        msg = pickle.dumps(payload)

        start_time = time.time()
        restarts = 0
        while True:
            self.sock.send(msg)

            # The rollout loops indefinitely server-side, so disable the receive
            # timeout for the wait. A scene-switch restart reply (if any) still
            # comes back quickly, before the loop starts.
            self.sock.setsockopt(zmq.RCVTIMEO, -1)  # block forever
            try:
                raw = self.sock.recv()
            finally:
                # Restore the normal timeout for subsequent calls.
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

            response_data = pickle.loads(raw)
            if response_data.get("retry"):
                # Scene switch — wait for the fresh process and resend.
                if restarts >= self.max_restarts:
                    raise RuntimeError(
                        f"Server kept restarting (>{self.max_restarts}) while loading "
                        f"rollout scene for '{target}'. Check inputs and server logs."
                    )
                restarts += 1
                print(f"[client] Server re-exec'ing for rollout scene "
                      f"(restart {restarts}/{self.max_restarts}); waiting for it to come back...")
                time.sleep(self.restart_grace_s)
                self._connect()
                continue
            break

        elapsed = time.time() - start_time
        if "error" in response_data:
            raise RuntimeError(f"Server execution error: {response_data['error']}")

        results = response_data.get("results", [])
        result = results[0] if results else {}
        info = {"elapsed_s": round(elapsed, 3), "restarts": restarts}
        return result, info

    def rollout_traj(self, traj, target, meshes, transformations,
                     hold_steps=60, steps_per_waypoint=3, max_cycles=None):
        """Like rollout(), but DRIVE the arm to the grasp along a planned
        trajectory instead of snapping it straight to the grasp config.

        Same full-scene build and looping behavior as rollout() — every object in
        `transformations` is spawned and `target` selects which one is grasped —
        but instead of teleporting the arm to a joint config, the server PLAYS
        `traj` waypoint-by-waypoint (gripper open) to reach the grasp pose, then
        closes / lifts / holds exactly as rollout() does. Loops FOREVER until you
        kill the server (Ctrl-C) or close the viewer. Launch the server with
        --render to watch it locally.

        `traj` is the (T,7) joint trajectory from IKClient.solve_traj(), e.g.:
            traj, ok, info = iksolver.solve_traj(goal_pose, meshes, transformations)
            client.rollout_traj(traj, target, meshes, transformations)

        Args:
            traj:            (T, 7) float32 array of arm joint waypoints (one row
                             per tick). The final row should be the grasp config.
            target:          object name to grasp.
            meshes:          {object_name: trimesh Scene/Trimesh} from reconstruct_3d.
            transformations: scene dict ({"objects": {name: entry}},
                             robot-base-frame poses/scale).
            hold_steps:      sim steps to hold at the top of each lift (~steps/60 s).
            steps_per_waypoint: sim steps spent tracking each waypoint (>=1); larger
                             = slower, smoother playback.

        Returns:
            result: {"rollout_traj": True, "cycles_completed": int} (after stop).
            info:   {"elapsed_s", "restarts"}.
        """
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty object name string")

        # Trajectory shape handling: accept (T,7) or a single (7,) row.
        if not isinstance(traj, np.ndarray):
            traj = np.array(traj, dtype=np.float32)
        traj = np.ascontiguousarray(traj, dtype=np.float32)
        if traj.ndim == 1 and traj.shape[0] == 7:
            traj = traj.reshape(1, 7)
        if traj.ndim != 2 or traj.shape[1] != 7:
            raise ValueError(f"traj must be (T, 7), got {traj.shape}")
        if traj.shape[0] < 1:
            raise ValueError("traj must have at least one waypoint")

        # Build the FULL multi-object scene payload (same shape settle/rollout send).
        if "objects" not in transformations or not transformations["objects"]:
            raise ValueError("transformations['objects'] is empty")
        if target not in transformations["objects"]:
            raise ValueError(
                f"rollout_traj target {target!r} not in transformations['objects'] "
                f"{list(transformations['objects'])}"
            )

        objects_payload = []
        for name, entry in transformations["objects"].items():
            mesh = meshes.get(name)
            if mesh is None:
                raise ValueError(f"no reconstructed mesh for object {name!r}")
            objects_payload.append({
                "name":        name,
                "mesh_bytes":  _glb_bytes(mesh),
                "mesh_format": "glb",
                "translation": [float(v) for v in entry["translation"]],
                "rotation":    [float(v) for v in entry["rotation_wxyz"]],
                "scale":       float(entry["scale"]),
            })

        payload = {
            "mode":               "rollout_traj",
            "target":             target,
            "objects":            objects_payload,
            # Nested list, not ndarray (numpy-version pickle portability) — server
            # does np.asarray() on receipt. Same as evaluate()/rollout().
            "traj":               traj.tolist(),
            "hold_steps":         int(hold_steps),
            "steps_per_waypoint": int(steps_per_waypoint),
        }
        if max_cycles is not None:
            # Bound the demo loop (batch runs must terminate); None = loop forever.
            payload["max_cycles"] = int(max_cycles)
        msg = pickle.dumps(payload)

        start_time = time.time()
        restarts = 0
        while True:
            self.sock.send(msg)

            # The rollout loops indefinitely server-side, so disable the receive
            # timeout for the wait. A scene-switch restart reply (if any) still
            # comes back quickly, before the loop starts.
            self.sock.setsockopt(zmq.RCVTIMEO, -1)  # block forever
            try:
                raw = self.sock.recv()
            finally:
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

            response_data = pickle.loads(raw)
            if response_data.get("retry"):
                if restarts >= self.max_restarts:
                    raise RuntimeError(
                        f"Server kept restarting (>{self.max_restarts}) while loading "
                        f"rollout_traj scene for '{target}'. Check inputs and server logs."
                    )
                restarts += 1
                print(f"[client] Server re-exec'ing for rollout_traj scene "
                      f"(restart {restarts}/{self.max_restarts}); waiting for it to come back...")
                time.sleep(self.restart_grace_s)
                self._connect()
                continue
            break

        elapsed = time.time() - start_time
        if "error" in response_data:
            raise RuntimeError(f"Server execution error: {response_data['error']}")

        results = response_data.get("results", [])
        result = results[0] if results else {}
        info = {"elapsed_s": round(elapsed, 3), "restarts": restarts}
        return result, info

    def close(self):
        """Gracefully disconnect resources."""
        if self.sock is not None:
            self.sock.close(0)


if __name__ == "__main__":
    # Quick functional demo. Needs trimesh only to fabricate a throwaway mesh so
    # the demo is self-contained; real callers pass their own in-memory bytes.
    server_addr = GRASP
    target_obj = "demo box"

    try:
        import trimesh
        # Same shape the pipeline uses: a {name: mesh} dict + name-keyed scene dict.
        meshes = {target_obj: trimesh.creation.box(extents=(0.05, 0.05, 0.05))}
        transformations = {"objects": {
            target_obj: {"translation": [0.1, 0.0, 0.5],
                         "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "scale": 1.0},
        }}
    except Exception as e:
        print(f"(demo needs trimesh to fabricate a mesh: {e})")
        raise SystemExit(0)

    print(f"Connecting to grasp evaluation server at {server_addr}...")
    client = GraspClient(server_addr)

    sample_joints = np.zeros((5, 7), dtype=np.float32)
    for idx in range(5):
        sample_joints[idx] = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7] + np.random.uniform(-0.05, 0.05, 7)

    try:
        # Full evaluation of all candidates.
        results, info = client.evaluate(
            sample_joints, target=target_obj,
            meshes=meshes, transformations=transformations,
        )
        print("\n--- Evaluation Run Summary ---")
        print(f"Stats: {info}")
        print("\nDetailed Output Rows:")
        for res in results:
            print(f"  Env {res['index']}: Passed={res.get('passed')} | "
                  f"Lift={res.get('lift_distance')}m | Reason={res.get('fail_reason', 'N/A')}")

        # Live looping rollout of one grasp (first row). This BLOCKS until you
        # kill the server, so it's left commented in the demo. Launch the server
        # with --render to watch it.
        # roll, rinfo = client.rollout(
        #     sample_joints, target=target_obj,
        #     meshes=meshes, transformations=transformations,
        #     hold_steps=60,
        # )
        # print(f"\n--- Rollout ended --- {roll} ({rinfo})")
    except Exception as e:
        print(f"Execution failure: {e}")
    finally:
        client.close()
        print("Client disconnected.")