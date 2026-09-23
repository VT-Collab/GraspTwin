#!/usr/bin/env python3
"""Semantic grasping with batched Thompson-sampling pose optimization.

Pipeline for one grasp:

  1. capture an RGB-D frame and detect / segment / reconstruct the scene
     (semantic_grasp.perception)
  2. ask the VLM which object and which part to grasp, and from which side
  3. seed a 6-DoF grasp pose from that answer
  4. refine it: Ax proposes each round's batch of pose deltas by Thompson
     sampling (Sobol init, then a GP with BoTorch's PathwiseThompsonSampling
     acquisition); every batch is scored with one batched IK solve and one
     physics rollout (semantic_grasp.ik / semantic_grasp.isaac)
  5. plan to the best pose and replay it on the arm (semantic_grasp.robot)

Entry point: run_once(). See README.md for the services this talks to.

    python main.py
"""
import copy
import json
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
# Line-buffer stdout so prints show up immediately even when the output is
# piped/redirected (otherwise they sit in an 8KB buffer and never appear while
# the run is blocked on a long op).
sys.stdout.reconfigure(line_buffering=True)

print("[startup] importing dependencies (cv2/open3d/trimesh)...")
import cv2
import matplotlib
matplotlib.use("Agg")   # headless HPC: no display, render figures straight to file
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.adapter.registry import Generators
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling


# botorch ships no input constructor for PathwiseThompsonSampling, and Ax's
# modular generator can only instantiate acquisitions that have one registered.
@acqf_input_constructor(PathwiseThompsonSampling)
def _construct_inputs_pathwise_thompson(model, objective=None, posterior_transform=None,
                                        X_pending=None):
    # X_pending is accepted only because Ax always passes it mid-batch; TS gets
    # batch diversity from a fresh posterior path per proposal, so it's unused.
    # PathwiseThompsonSampling applies its default IdentityMCObjective *after*
    # the posterior transform, squeezing the q dim as well as the (already
    # scalarized) metric dim and crashing optimize_acqf. Fold the scalarizing
    # transform into an equivalent MC objective instead.
    if objective is None and posterior_transform is not None:
        objective = LinearMCObjective(weights=posterior_transform.weights)
        posterior_transform = None
    return {"model": model, "objective": objective,
            "posterior_transform": posterior_transform}

from pathlib import Path

from semantic_grasp.config import TASK, TARGET, reset_output_dir, OUTPUT_DIR
from semantic_grasp.perception import (
    capture_rgbd, get_pertinent_objects, segment_objects, reconstruct_3d,
    VLM_Predict_XYZ_Semantic, VLM_Predict_RPY_Semantic, VLM_Predict_XYZ,
    VLM_Predict_RPY, find_target_name,
)
from semantic_grasp.isaac import GraspClient
from semantic_grasp.ik import IKClient
from semantic_grasp.robot import RobotClient, DEPLOY_TIME_SCALE, LIFT_M
from semantic_grasp.camera import OrbbecClient
print("[startup] imports done; constructing grasp + IK clients...")

MAIN_DIR = Path(__file__).resolve().parent   # repo root -- not OUTPUT_DIR

grasp = GraspClient()
iksolver = IKClient()
print("[startup] clients ready")

# Batched Thompson sampling: round 1 scores N_INIT Sobol deltas in one big
# batch; every later round scores BATCH_SIZE GP-proposed deltas.
BATCH_SIZE = 140
N_ROUNDS = 3
N_INIT = 140                     # round-1 Sobol batch; the GP takes over after it

# 6-DoF search bounds for the pose delta [dx, dy, dz, tilt_x, tilt_y, twist].
# Rotation deltas compose in the GRIPPER'S LOCAL frame (see apply_delta):
# tilt_x/tilt_y rock the approach axis away from the VLM's choice, twist spins
# the jaw line about it. A cluster whose search never lifts the object is
# abandoned and the refine loop moves to the next cluster (see run_once). Only
# once EVERY cluster has failed does cluster 1 get one retry with
# WIDE_BOUND_RPY on the tilts (twist stays +-pi/2 — already every jaw line).
BOUND_XYZ = 0.020                 # translation bound (m); +-2 cm, 4 cm span
BOUND_RPY = math.pi / 3          # tilt bound (rad); twist spans [-pi/2, pi/2]
WIDE_BOUND_RPY = math.pi         # tilt bound (rad) for the last-resort retry

# When a cluster's round-1 Sobol batch never lifts, re-roll it with a fresh
# batch up to this many times before moving to the next cluster (see the
# refine loop). Same value as main_thompson_sim.SOBOL_RETRIES.
SOBOL_RETRIES = 3

NAN_COST = 0.1                   # IK-infeasible cost (worse than any real grasp)
COST_CUTOFF = -0.85              # >=85% lift success -> good enough, stop searching
LIFT_THRESHOLD = 0.1             # metres an object must rise to count as lifted
# Multi-object scenes only (clearance-promoted neighbors in the eval world):
# a rollout counts as a success only when the target lifts CLEANLY — hoisting
# any neighbor past LIFT_THRESHOLD alongside it is a failure, same as lifting
# nothing or lifting only a neighbor.

# ── THE COST FUNCTION ────────────────────────────────────────────────────────
# Every grasp is scored with two terms:
#
#     cost = -success_rate  +  WIDTH_COST_WEIGHT * width_error
#            └── primary ──┘   └──── secondary, tunable ─────┘
#
#   success_rate  in [0, 1]: fraction of candidate scenes the grasp lifted the
#                            target cleanly in. This is the objective that
#                            matters, so it owns the whole [-1, 0] range.
#   width_error   in METRES: |VLM-predicted gripper width - the width the jaws
#                            actually closed to|. A grasp that lifts but closes
#                            far tighter than predicted caught a thin lip or a
#                            corner instead of the part the VLM chose; far wider
#                            means it straddled something bulkier. Both still
#                            lift, so success alone cannot tell them apart —
#                            this term does.
#
# The two terms read DIFFERENT scenes. Success is averaged over the whole domain
# randomization — that's the point of it, a grasp should survive the object
# being somewhere slightly different. Width is read off the GROUND-TRUTH scene
# alone: it is the scene that matches the built/settled world exactly, so its
# closed aperture is the one comparable to the VLM's prediction.
#
# TUNING: WIDTH_COST_WEIGHT is the only knob, and it is currently 0.0 — the
# width term is OFF and the cost is pure -success_rate, exactly as it was before
# the term existed. The plumbing stays live (the widths are still measured and
# printed each round), so raising this is the only edit needed to switch it on.
#
# Because the error is raw metres, the weight also carries the unit conversion —
# what a given weight would charge, against a success term worth a full 1.0:
#
#     weight   1 cm miss   3 cm miss   full 8 cm miss
#      0.0       0.00        0.00          0.00        <- current: term disabled
#      1.0       0.01        0.03          0.08
#      5.0       0.05        0.15          0.40
#     10.0       0.10        0.30          0.80
#
# Keep any nonzero weight under ~12 so even the worst possible width miss can't
# make a lifting grasp score above a non-lifting one (-1 + weight * 0.08 must
# stay < 0). It also interacts with COST_CUTOFF: at 5.0 a grasp that lifts every
# scene but misses width by 3 cm scores -0.85, right at the cutoff, so bigger
# misses would keep the search running.
WIDTH_COST_WEIGHT = 0.0
# Which candidate scene the width is read from. generate_pose_candidates puts
# the untouched scene — no xy shift, no yaw — at index 0, and that is the one
# matching the world grasp.load_meshes actually built and settled.
GROUND_TRUTH_SCENE = 0


def width_error(closed_width, predicted_width):
    """|width the VLM predicted - width the jaws actually closed to|, in metres.

    Returns 0.0 — no opinion — when there is no width to compare against (the
    VLM produced none, or the rollout reported none), so the term quietly
    disappears instead of taxing every grasp equally.
    """
    if predicted_width is None or closed_width is None:
        return 0.0
    if not math.isfinite(closed_width):
        return 0.0
    return abs(predicted_width - closed_width)

# Clearance (m) added to every side of the TARGET's OBB for the crowding check
# in reconstruct_3d: any non-target whose OBB intersects the inflated target OBB
# is promoted to a full sam3d reconstruction and spawned in the Isaac eval world
# beside the target. Domain randomization still moves ONLY the target.
TARGET_OBB_CLEARANCE = 0.01


def generate_pose_candidates(transformation, target, *, radius=0.05, num_points=8,
                             yaw_degrees=(0, 5, -5, 10, -10)):
    """Perturbed copies of the scene dict, randomizing the whole EVAL GROUP as a
    rigid unit so the objects move together.

    The eval group is every object spawned in the Isaac world —
    `transformation["eval_objects"]` (the target plus any clearance-promoted
    neighbors), or just the target when that key is absent. Each candidate
    applies ONE rigid transform to the whole group, preserving their relative
    arrangement: an xy shift (the un-shifted centre, or one of `num_points`
    points on a circle of `radius`) combined with a yaw from `yaw_degrees`,
    taken ABOUT THE TARGET'S CENTRE. Scale is never changed — the Isaac eval
    world is built from ONE cooked mesh, so a per-scene scale could not be
    honored anyway. The target sits at the pivot; the neighbors orbit it rigidly.
    Index 0 is the untouched scene (no shift, no yaw) so it matches the
    built/settled world; length is (num_points + 1) * len(yaw_degrees) = 45.
    """
    def yaw_quat(angle):  # wxyz quaternion for a rotation about +Z
        return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]

    def quat_mul(q1, q2):  # Hamilton product, wxyz
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ]

    # The group that goes to Isaac: target + clearance-promoted neighbors.
    # Everything else (far box stand-ins) is IK-only context, left untouched.
    objects = transformation["objects"]
    group = [n for n in transformation.get("eval_objects", []) if n in objects]
    if target not in group:
        group = [target] + group

    # Rigid-transform pivot: the target's centre. Freeze each member's base
    # pose before we start writing copies.
    px, py = objects[target]["translation"][0], objects[target]["translation"][1]
    bases = {n: (list(objects[n]["translation"]), list(objects[n]["rotation_wxyz"]))
             for n in group}

    # xy offsets: the centre first (so index 0 is the ground-truth scene), then
    # the ring.
    offsets = [(0.0, 0.0)] + [(radius * math.cos(2 * math.pi * k / num_points),
                               radius * math.sin(2 * math.pi * k / num_points))
                              for k in range(num_points)]

    candidates = []
    for yaw_deg in yaw_degrees:
        yaw = math.radians(yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        for dx, dy in offsets:
            scene = copy.deepcopy(transformation)
            for n in group:
                base_t, base_q = bases[n]
                # rotate this object's centre about the target pivot, then shift
                rx, ry = base_t[0] - px, base_t[1] - py
                obj = scene["objects"][n]
                obj["translation"][0] = px + (cy * rx - sy * ry) + dx
                obj["translation"][1] = py + (sy * rx + cy * ry) + dy
                obj["rotation_wxyz"] = quat_mul(yaw_quat(yaw), base_q)
            candidates.append(scene)
    return candidates


def apply_delta(base_pose, deltas):
    """Compose search deltas onto base_pose; the ONLY delta -> pose mapping.

    Translation deltas add in the robot frame. Rotation deltas compose in the
    gripper's LOCAL frame: R = R_base @ Rx(tilt_x) @ Ry(tilt_y) @ Rz(twist),
    so the innermost Rz is always a jaw twist about the (tilted) approach axis
    — local +z is the approach axis (gripper_R_from_approach). +-pi/2 of twist
    covers every distinct jaw line for ANY approach (two-finger 180-degree
    symmetry); the old additive-euler delta's yaw was a world-vertical spin
    that only equalled the jaw twist for top-down grasps. For the top-face
    base [pi, 0, 0] the two parameterizations span the identical pose set
    (conjugation through Rx(pi) flips signs only), so top-down searches are
    unchanged. Returns (B, 6) poses [xyz, rpy] for (B, 6) deltas.
    """
    deltas = np.asarray(deltas, dtype=float).reshape(-1, 6)
    R = Rot.from_euler("xyz", base_pose[3:]) * Rot.from_euler("XYZ", deltas[:, 3:])
    return np.column_stack([base_pose[:3] + deltas[:, :3], R.as_euler("xyz")])


def evaluate_batch(deltas, base_pose, meshes, transformations, candidates, target,
                   extra_objects=(), predicted_width=None):
    """Score an Ax batch of pose deltas with one IK solve + one sim rollout.

    `extra_objects` are the clearance-promoted neighbor names spawned in the
    eval world beside the target (the candidates only randomize the target, so
    the neighbors sit at their settled poses in every scene).
    `predicted_width` is VLM_Predict_Gripper_Width's answer in metres, or None.

    Returns costs (B,): -success_rate + WIDTH_COST_WEIGHT * width_error for a
    reachable pose, or NAN_COST (>0) when IK is infeasible. Single-object scenes
    score each rollout 1 if the target rose >= LIFT_THRESHOLD else 0. With
    neighbors in the eval world a rollout scores 1 only if the target lifted
    CLEANLY; it scores 0 if any neighbor lifted too (co-lift), if the target
    stayed down, or if only a neighbor got picked up. See THE COST FUNCTION
    above for the width term.
    """
    deltas = np.asarray(deltas, dtype=float).reshape(-1, 6)
    B = deltas.shape[0]
    poses = apply_delta(base_pose, deltas)

    # The full scene — target included — is the collision world for IK.
    # Pin batch_size to BATCH_SIZE (the server pads short batches): it keys the
    # IK solver cache, so a varying B would rebuild the solver every round.
    joints, _, _ = iksolver.solve(poses, meshes, transformations,
                                  batch_size=BATCH_SIZE, gripper_sphere_scale=1)
    joints = np.asarray(joints, dtype=np.float32).reshape(B, 7)

    feasible = np.flatnonzero(np.all(np.isfinite(joints), axis=1))
    print(f"    IK: {feasible.size}/{B} feasible")
    costs = np.full(B, NAN_COST, dtype=float)
    if not feasible.size:
        return costs

    # Score every feasible grasp against every candidate scene in one rollout;
    # result row g*C + c is grasp g in scene c.
    C = len(candidates)
    flat_joints = np.repeat(joints[feasible], C, axis=0)
    flat_scenes = list(candidates) * feasible.size
    print(f"    sim: rolling out {feasible.size} grasp(s) x {C} scene(s) "
          f"= {flat_joints.shape[0]} evals...")
    results, _ = grasp.evaluate(flat_joints, target, meshes, flat_scenes,
                                extra_objects=extra_objects)

    # Graded per-rollout success (see docstring). NaN heights compare False,
    # so a nan_in_sim rollout scores 0 either way; .get() keeps this working
    # against an isaac_server that predates extra_lift_distances.
    success = np.zeros((feasible.size, C), dtype=float)
    widths  = np.zeros(feasible.size, dtype=float)     # one per grasp, not per scene
    for r in results:
        g, c = divmod(r["index"], C)
        target_lifted = (r["post_lift_obj_pos"][2] - r["pre_close_obj_pos"][2]) >= LIFT_THRESHOLD
        if not target_lifted:
            continue                    # neighbor-only lift or no lift -> 0
        co_lifted = any(dz >= LIFT_THRESHOLD
                        for dz in r.get("extra_lift_distances") or [])
        if co_lifted:
            continue
        success[g, c] = 1.0
        if c == GROUND_TRUTH_SCENE:
            widths[g] = width_error(r.get("closed_gripper_width"), predicted_width)

    # ── the two cost terms (see THE COST FUNCTION at the top of the file) ──
    # Primary: fraction of ALL candidate scenes this grasp lifted the target in.
    success_rate = success.mean(axis=1)
    # Secondary: the width miss in the GROUND-TRUTH scene only (see the constant).
    # Stays 0 for a grasp that didn't lift there — the jaws closed on nothing, so
    # their aperture says nothing about the grasp, and success already scored it.
    costs[feasible] = -success_rate + WIDTH_COST_WEIGHT * widths

    # Diagnostic only — nothing downstream reads this; Ax gets the per-trial
    # costs. It exists to size the two terms against each other while tuning
    # WIDTH_COST_WEIGHT. The width figure is averaged over just the grasps that
    # lifted in the ground-truth scene, because those are the only ones the term
    # can charge; averaging over the whole batch would dilute it with the
    # structural zeros and make the term look smaller than it is.
    scored = widths[success[:, GROUND_TRUTH_SCENE] > 0]
    miss = scored.mean() if scored.size else 0.0
    print(f"    cost: mean success={success_rate.mean():.3f} over {feasible.size} grasp(s); "
          f"width miss={miss * 100:.2f} cm over the {scored.size} that lifted in the "
          f"ground-truth scene (x{WIDTH_COST_WEIGHT} -> {WIDTH_COST_WEIGHT * miss:.3f})")
    return costs


def optimize_point(base_pose, meshes, transformations, candidates, target,
                   extra_objects=(), predicted_width=None,
                   bound_rpy=BOUND_RPY, give_up_if_round_1_never_lifts=False):
    """Batched Ax Thompson-sampling search for the best pose delta around one VLM point.

    Round 1 is one N_INIT-trial Sobol batch that seeds the GP; after that every
    proposal maximizes a fresh GP posterior sample (PathwiseThompsonSampling),
    so a round is BATCH_SIZE independent posterior-sample argmaxes. Ax only proposes points
    inside [lower, upper], so no manual bounds guard is needed.
    `predicted_width` feeds evaluate_batch's width cost term.
    `bound_rpy` is the +- gripper-local tilt bound on d3/d4 (see apply_delta);
    the d5 jaw twist is always +-pi/2. `give_up_if_round_1_never_lifts`
    returns after round 1 when its best cost is still >= 0.
    Returns (best_cost, best_pose, func_vals).
    """
    # [dx, dy, dz, tilt_x, tilt_y, twist] — mapped to a pose by apply_delta.
    lower = np.array([-BOUND_XYZ] * 3 + [-bound_rpy, -bound_rpy, -math.pi / 2])
    upper = -lower
    # max_parallelism=None: let every round return a full batch.
    strategy = GenerationStrategy(
        name="sobol+thompson",
        steps=[
            GenerationStep(generator=Generators.SOBOL, num_trials=N_INIT,
                           max_parallelism=None),
            GenerationStep(generator=Generators.BOTORCH_MODULAR, num_trials=-1,
                           max_parallelism=None,
                           model_kwargs={"botorch_acqf_class": PathwiseThompsonSampling}),
        ],
    )
    ax_client = AxClient(generation_strategy=strategy, verbose_logging=False)
    ax_client.create_experiment(
        name="grasp_delta",
        parameters=[
            {"name": f"d{i}", "type": "range",
             "bounds": [float(lower[i]), float(upper[i])], "value_type": "float"}
            for i in range(6)
        ],
        objectives={"loss": ObjectiveProperties(minimize=True)},
    )

    best_cost, best_delta, func_vals = float("inf"), np.zeros(6), []

    for round_i in range(N_ROUNDS):
        batch = N_INIT if round_i == 0 else BATCH_SIZE
        print(f"  [round {round_i + 1}/{N_ROUNDS}] requesting {batch} trials...")
        trials, _ = ax_client.get_next_trials(max_trials=batch)
        if not trials:
            break
        idxs = list(trials.keys())
        deltas = np.array([[trials[i][f"d{j}"] for j in range(6)] for i in idxs], dtype=float)

        costs = evaluate_batch(deltas, base_pose, meshes, transformations, candidates,
                               target, extra_objects, predicted_width)
        for k, i in enumerate(idxs):
            cost = float(costs[k])
            ax_client.complete_trial(trial_index=i, raw_data=cost)
            func_vals.append(cost)
            if cost < best_cost:
                best_cost, best_delta = cost, deltas[k].copy()

        print(f"  [round {round_i + 1}/{N_ROUNDS}] costs={np.round(costs, 3)} "
              f"best={best_cost:.3f}")
        if round_i == 0 and give_up_if_round_1_never_lifts and best_cost >= 0.0:
            print(f"  round 1 never lifted the object (best {best_cost:.3f}); "
                  f"giving up on this point")
            break
        if best_cost <= COST_CUTOFF:       # good enough -> stop burning budget
            print(f"  best {best_cost:.3f} <= cutoff {COST_CUTOFF}; "
                  f"good enough, skipping the remaining rounds")
            break

    # Same composition the batch was scored under — a plain base + delta here
    # would deploy a DIFFERENT pose than the one that earned best_cost.
    return best_cost, apply_delta(base_pose, best_delta)[0], func_vals


def plot_histories(histories):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, costs in histories:
        iters = np.arange(1, len(costs) + 1)
        line, = ax.plot(iters, costs, marker='o', markersize=3, alpha=0.6, label=label)
        ax.plot(iters, np.minimum.accumulate(costs), '--', color=line.get_color(),
                label=f"{label} (best so far)")
    ax.set(xlabel="iteration",
           ylabel=f"cost (-success rate + {WIDTH_COST_WEIGHT}*width miss)",
           title="Ax batched Thompson-sampling cost vs iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "cost_history.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved cost-vs-iteration graph to {out_path}")


def graph_candidates(candidates, target):
    """Scatter the target object's xy across every domain-randomized candidate.

    Sanity-checks generate_pose_candidates: each candidate scene should place
    `target` on the expected pattern of xy offsets (a ring of `num_points` at
    `radius`, plus the un-offset centre). One point per candidate; positions
    repeat across the yaw sweep, so overlap shows as darker blobs.
    Candidate 0 is the untouched ground-truth scene and is marked distinctly.
    """
    xs = np.array([c["objects"][target]["translation"][0] for c in candidates])
    ys = np.array([c["objects"][target]["translation"][1] for c in candidates])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=70, alpha=0.4, color="#3b6ea5", edgecolor="white",
               linewidth=0.5, label=f"candidates (n={len(candidates)})")
    # candidate 0 == base xy, no yaw, true scale: the real (un-randomized) scene.
    ax.scatter(xs[0], ys[0], s=180, marker="*", color="#d1495b", zorder=3,
               edgecolor="white", linewidth=0.6, label="ground truth (base)")

    # recessive reference ring at the empirical randomization radius.
    r = float(np.hypot(xs - xs[0], ys - ys[0]).max())
    if r > 0:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(xs[0] + r * np.cos(theta), ys[0] + r * np.sin(theta), "--",
                color="#9aa0a6", linewidth=1, alpha=0.7, label=f"ring r={r:.3f} m")

    ax.set(xlabel="target x (m)", ylabel="target y (m)",
           title=f"Domain-randomized xy placements for {target!r}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "candidate_xy.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved candidate xy scatter to {out_path} "
          f"(x in [{xs.min():.3f}, {xs.max():.3f}], "
          f"y in [{ys.min():.3f}, {ys.max():.3f}])")
    return out_path


def run_once(task):
    """One full pipeline cycle: perception -> VLM proposal -> sim refinement ->
    deploy, for the typed task. Everything heavy (imports, GraspClient/IKClient,
    the servers' own solver caches) lives at module level, so repeated calls
    stay warm.

    This is main_thompson_sim.run_once with only its two ends swapped — keep
    the middle identical to it (see the module docstring):
        obs     one live camera frame (capture_rgbd) instead of the request
                blob's rgb + organized cloud
        deploy  the chosen gripper_tcp pose is planned to over
                baseline_server's five-rung ladder and replayed on the lab arm,
                instead of being returned as JSON for baseline_server to deploy
    """
    # --- perception (live camera frame) ----------------------------------------
    print("Starting")
    time_start = time.perf_counter()
    reset_output_dir()
    print("[perception] capturing RGB-D frame...")
    img, depth = capture_rgbd()
    # depth is a uint16 mm map (live camera) or an open3d PointCloud (offline
    # test path); reconstruct_3d back-projects it through camera_model. In the
    # sim server the request's organized cloud stands in for it.
    depth_desc = (f"{depth.shape} map" if hasattr(depth, "shape")
                  else f"pointcloud {len(depth.points)} pts")
    print(f"[perception] captured img={img.shape} depth={depth_desc}")

    print("[perception] detecting pertinent objects...")
    objects = get_pertinent_objects(img)
    target = find_target_name(task, objects)
    print(f"[perception] objects={objects}  target={target!r}")

    print(f"[perception] segmenting {len(objects)} object(s)...")
    masks = segment_objects(img, objects)
    print(f"[perception] got {len(masks)} mask(s)")

    print("[perception] reconstructing 3D meshes...")
    # capture_rgbd hands back BGR (what detection/segmentation take);
    # reconstruct_3d wants RGB.
    meshes, transformations = reconstruct_3d(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), depth,
                                             masks, target, full_scene=False,
                                             target_clearance=TARGET_OBB_CLEARANCE)
    print(f"[perception] reconstructed meshes for {list(meshes.keys())}")

    # Clearance-promoted neighbors: fully reconstructed by sam3d so the IK
    # collision world (iksolver.solve gets the whole scene) sees real geometry
    # beside the target. IK-only context — the Isaac eval world spawns the
    # target alone, so no extra_objects go to load_meshes/evaluate below.
    ik_only_neighbors = [n for n in transformations.get("eval_objects", [])
                         if n != target]
    if ik_only_neighbors:
        print(f"[perception] IK collision world also carries {ik_only_neighbors} "
              f"(inside the {TARGET_OBB_CLEARANCE * 1000:.0f} mm clearance of "
              f"{target!r}); Isaac eval world spawns the target alone")

    print("[perception] settling scene in sim...")
    transformations, _ = grasp.settle(meshes, transformations)
    print("[perception] settle done")

    # Kick off the eval-world build NOW (non-blocking): 'load' shares the
    # 'try' scene identity, so the server re-execs and builds the grasp world
    # while the VLM stages below run — evaluate() then lands on a live scene.
    grasp.load_meshes(target, meshes, transformations)

    # --- VLM grasp proposal --------------------------------------------------
    # Semantic first — WHERE to grasp and HOW to approach, in words — then each
    # phrase conditions its numeric stage. Same dependency DAG as main_thompson_sim:
    #   xyz_semantic ─┬─> rpy_semantic ──> rpy_vlm
    #                 └─> xyz_vlm
    print(f"[vlm] in:  task={task!r}  target={target!r}")
    xyz_semantic = VLM_Predict_XYZ_Semantic(task, target, meshes, transformations)
    print(f"[vlm]      XYZ_Semantic -> grasp part {xyz_semantic!r}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_rpy_sem = pool.submit(VLM_Predict_RPY_Semantic, task, target, meshes,
                                transformations, xyz_semantic)
        f_xyz = pool.submit(VLM_Predict_XYZ, task, target, meshes, transformations,
                            xyz_semantic, save_dir=OUTPUT_DIR / "grasp_xyz")

        # rpy_vlm needs the approach phrase — launch it as soon as it's ready.
        rpy_semantic = f_rpy_sem.result()
        f_rpy = pool.submit(VLM_Predict_RPY, task, target, meshes, transformations,
                            grasp_part=xyz_semantic, approach_hint=rpy_semantic,
                            save_dir=OUTPUT_DIR / "grasp_rpy")

        xyz_vlm, _ = f_xyz.result()
        rpy_vlm = f_rpy.result()

    # unproject_predictions returns one row per HDBSCAN cluster — the mean of
    # each place a subset of views agreed on — sorted nearest-to-overall-mean
    # first. The refine loop below tries them in that order.
    xyz_vlm = np.asarray(xyz_vlm, dtype=float).reshape(-1, 3)
    base_rpy = np.asarray(rpy_vlm, dtype=float).reshape(3)

    # One block, one line per stage: what the stage is called -> what it returned.
    more = ("" if len(xyz_vlm) < 2 else
            f"  (+{len(xyz_vlm) - 1} more cluster(s) to fall back on)")
    print(f"[vlm] out: RPY_Semantic  -> approach   {rpy_semantic!r}\n"
          f"[vlm]      XYZ           -> xyz        {np.round(xyz_vlm[0], 4).tolist()} m{more}\n"
          f"[vlm]      RPY           -> rpy        {np.round(base_rpy, 4).tolist()} rad\n"
          f"[vlm] photos sent to each stage: {OUTPUT_DIR}/grasp_{{xyz,rpy}}/")

    # --- refine each grasp point against the physics sim ----------------------
    candidates = generate_pose_candidates(transformations, target)
    graph_candidates(candidates, target)   # verify the xy domain randomization
    # Try the HDBSCAN cluster centroids IN ORDER and stop at the first that
    # works. They arrive sorted nearest-to-overall-mean first, so [0] is the
    # cluster closest to the whole-scene consensus; each next one only gets a
    # turn if every earlier one failed to lift the object in any rollout. A cost
    # below zero means at least one delta lifted it, so that is the bar for
    # "found a grasp" — the same test the deploy guard uses. Anything >= 0 falls
    # through, including NAN_COST (0.1) when every pose was IK-infeasible.
    print(f"[refine] {len(candidates)} candidate scene(s); up to "
          f"{len(xyz_vlm)} cluster(s) over {N_INIT} + "
          f"{N_ROUNDS - 1} x {BATCH_SIZE} deltas")
    histories, best_overall = [], None

    def optimize_with_sobol_retries(base_pose, kind, bound_rpy=BOUND_RPY):
        # give_up_if_round_1_never_lifts: a dead Sobol round means constant
        # loss, and fitting the GP to that is degenerate and slow (many-minute
        # rounds of flat-likelihood optimization for near-random proposals).
        # Instead re-roll round 1 up to SOBOL_RETRIES times — each
        # optimize_point call builds a fresh AxClient, so every retry scores a
        # differently-scrambled Sobol batch, not the same 140 deltas. Still
        # dead after all retries -> the caller falls through to the next
        # cluster / the wide-band fallback / the raw-VLM-pose deploy guard.
        best_cost, best_pose, func_vals = optimize_point(
            base_pose, meshes, transformations, candidates, target,
            bound_rpy=bound_rpy, give_up_if_round_1_never_lifts=True)
        for retry_i in range(SOBOL_RETRIES):
            if best_cost < 0.0:
                break
            print(f"[refine] {kind}: round 1 never lifted (cost {best_cost:.3f}); "
                  f"fresh Sobol batch retry {retry_i + 1}/{SOBOL_RETRIES}")
            retry_cost, retry_pose, retry_vals = optimize_point(
                base_pose, meshes, transformations, candidates, target,
                bound_rpy=bound_rpy, give_up_if_round_1_never_lifts=True)
            func_vals = func_vals + retry_vals
            if retry_cost < best_cost:
                best_cost, best_pose = retry_cost, retry_pose
        return best_cost, best_pose, func_vals

    for pt_i, point in enumerate(xyz_vlm):
        kind = f"cluster {pt_i + 1}"
        print(f"[refine] {kind}/{len(xyz_vlm)}: xyz={point}")
        base_pose = np.concatenate((np.asarray(point, dtype=float).reshape(3), base_rpy))
        time1 = time.perf_counter()
        best_cost, best_pose, func_vals = optimize_with_sobol_retries(base_pose, kind)
        time2 = time.perf_counter()
        print(f"[timing] optimization: {time2 - time1:.2f} s "
              f"({len(func_vals)} evals, budget {N_INIT} + "
              f"{N_ROUNDS - 1} x {BATCH_SIZE})")
        print(f"[refine] point={point}  cost={best_cost}\n"
              f"  best pose xyz={best_pose[:3]} rpy={best_pose[3:]}")
        if best_overall is None or best_cost < best_overall[0]:
            best_overall = (best_cost, best_pose)
        histories.append((f"{kind} {np.round(point, 3)}", np.asarray(func_vals)))
        if best_cost < 0.0:
            print(f"[refine] {kind} point lifted the object (cost {best_cost:.3f}); "
                  f"deploying it, not trying the remaining "
                  f"{len(xyz_vlm) - pt_i - 1} candidate(s)")
            break
        print(f"[refine] {kind} point never lifted the object (cost {best_cost:.3f})"
              + (f"; falling back to the next candidate"
                 if pt_i + 1 < len(xyz_vlm) else "; no candidates left"))

    # Every cluster failed -> one last retry on cluster 1 with the approach
    # tilts widened to +-WIDE_BOUND_RPY (the jaw twist is already complete at
    # +-pi/2), with the same fresh-Sobol re-rolls; the deploy guard
    # below then falls back to the raw VLM pose.
    if best_overall[0] >= 0.0:
        point = xyz_vlm[0]
        print(f"[refine] wide band: retrying cluster 1 xyz={point} with tilt "
              f"+-{math.degrees(WIDE_BOUND_RPY):.0f} deg "
              f"(was +-{math.degrees(BOUND_RPY):.0f})")
        base_pose = np.concatenate((np.asarray(point, dtype=float).reshape(3), base_rpy))
        time1 = time.perf_counter()
        best_cost, best_pose, func_vals = optimize_with_sobol_retries(
            base_pose, "wide band", bound_rpy=WIDE_BOUND_RPY)
        time2 = time.perf_counter()
        print(f"[timing] wide-band optimization: {time2 - time1:.2f} s ({len(func_vals)} evals)")
        print(f"[refine] wide band: point={point}  cost={best_cost}\n"
              f"  best pose xyz={best_pose[:3]} rpy={best_pose[3:]}")
        if best_cost < best_overall[0]:
            best_overall = (best_cost, best_pose)
        histories.append((f"wide cluster 1 {np.round(point, 3)}", np.asarray(func_vals)))
    time_end = time.perf_counter()
    infer_s = time_end - time_start        # task entered -> refined xyzrpy chosen
    plot_histories(histories)

    # --- choose the pose: the sim server replies with this; here it gets deployed
    best_cost, deploy_pose = best_overall
    print(f"[deploy] best grasp over all points: cost={best_cost}")
    # cost >= 0 means nothing lifted in sim; deploy the raw VLM proposal anyway
    # rather than deploying nothing.
    if best_cost >= 0.0:
        deploy_pose = np.concatenate((xyz_vlm[0], base_rpy))
        print("[deploy] no grasp candidate lifted the object in sim; "
              "falling back to the VLM-proposed pose")

    print(f"[deploy] deploying xyz={deploy_pose[:3]} rpy={deploy_pose[3:]} (cost {best_cost})")
    print(f"[timing] inference {infer_s:.2f} s "
          f"(task entered -> refined xyzrpy chosen)")

    # --- plan: baseline_server's five-rung planner ladder ------------------------
    # The same ladder handle_submit walks for every method's sim pose (and
    # graspmolmo_real.py walks on the arm): gripper_sphere_scale=1 throughout,
    # each rung dropping more of the settled world from the collision scene.
    # The table stays an obstacle on every rung except the last.
    settled = transformations["objects"]
    others = [n for n in settled if n != target]
    ladder = [
        ("all objects + table",
         meshes, transformations, False),
        ("all objects but the target + table",
         {n: meshes[n] for n in others}, {"objects": {n: settled[n] for n in others}}, False),
        ("target only + table",
         {target: meshes[target]}, {"objects": {target: settled[target]}}, False),
        ("table only",
         {}, {}, False),
        ("ignore_collisions",
         {}, {}, True),
    ]
    sphere_scale = 1
    for rung, (label, rung_meshes, rung_transforms, ignore) in enumerate(ladder, start=1):
        print(f"[deploy] rung {rung} ({label})...")
        traj, traj_ok, traj_info = iksolver.solve_traj(
            deploy_pose, rung_meshes, rung_transforms,
            gripper_sphere_scale=sphere_scale, ignore_collisions=ignore)
        if traj_ok:
            print(f"[deploy] solved on rung {rung} ({label}) len={len(traj)} info={traj_info}")
            break
        print(f"[deploy] rung {rung} ({label}) did not solve")
    else:
        raise ValueError("no trajectory found on any rung")
    if rung >= 3:
        print(f"[deploy] WARNING: rung {rung} planned without "
              f"{'the table or any object' if rung == 5 else 'the non-target objects'} "
              f"in the collision world — the arm may sweep through them")

    # --- cache the solved trajectory to disk (same layout as main.py, consumed
    # by scripts/run_traj_from_cache.py to execute and viz_traj_from_cache.py to
    # replay in Viser). Keep only the newest traj_* entry, but leave any other
    # files in the dir alone.
    cache_dir = Path("trajectory_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("traj_*"):
        old.unlink()
    safe_target = "".join(c if c.isalnum() else "_" for c in str(target))
    base = f"traj_{safe_target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    traj_arr = np.asarray(traj, dtype=np.float32)
    np.save(cache_dir / f"{base}.npy", traj_arr)
    # meshes + transforms are required to replay in sim; joints alone are
    # meaningless without the scene they were solved against.
    with open(cache_dir / f"{base}_scene.pkl", "wb") as f:
        pickle.dump({"meshes": meshes, "transformations": transformations}, f)
    with open(cache_dir / f"{base}.json", "w") as f:
        json.dump({
            "target": target,
            "task": task,
            "deploy_pose": np.asarray(deploy_pose, dtype=float).tolist(),
            "best_cost": float(best_cost),
            "infer_s": round(infer_s, 3),
            "gripper_sphere_scale": sphere_scale,
            "ik_rung": rung,
            "traj_ok": bool(traj_ok),
            "traj_info": str(traj_info),
            "dt": (float(traj_info["dt"])
                   if isinstance(traj_info, dict) and "dt" in traj_info else None),
            "traj_shape": list(traj_arr.shape),
            "traj_npy": f"{base}.npy",
            "scene_pkl": f"{base}_scene.pkl",
        }, f, indent=2)
    print(f"[deploy] cached trajectory -> trajectory_cache/{base}.npy {traj_arr.shape}")

    # --- replay on the lab arm ---------------------------------------------------
    # Through robot_server.py on the LAB PC rather than driving the arm from
    # here: execute_streamed holds a 50 Hz command clock, and a round trip from
    # this node through the jump host does not fit in 20 ms — the arm then gets
    # its velocities late and irregularly, which is what jitter is. The client
    # ships the whole trajectory in one ~5 KB message and the control loop runs
    # next to the robot (RUNNING.md "Latency warning").
    #
    # DEPLOY_TIME_SCALE stretches the plan's clock (HIGHER = SLOWER); it lives in
    # robot.py so scripts/run_traj_from_cache.py replays this exact motion, which
    # is where a deploy gets tuned before it runs from here.
    robot = RobotClient()
    # Nothing moves until enter is pressed. The rollout is then recorded by
    # camera_server onto ITS OWN disk (RECORD_DIR at the top of that file, on
    # the lab PC) with the task string burned into the frames — no video
    # crosses the tunnel. The clip takes the trajectory_cache stem, so
    # <base>.mp4 there pairs with <base>.npy here.
    input("[deploy] ready to deploy and record? press enter: ")
    with OrbbecClient() as cam:
        rec = cam.record_start(task, name=base)
        print(f"[deploy] recording -> {rec['path']} (on the camera host)")
        try:
            # force=True: the arm is normally parked at home while the plan
            # starts from cuRobo's default config, so the server's
            # start-distance gate fires on essentially every deploy.
            # go_to_start still walks it there with the same slow P-control
            # move — this only skips the confirmation the server cannot ask.
            # lift_m: once the gripper closes the server raises the TCP LIFT_M
            # straight up with the orientation free (Robot.lift), still on
            # camera, before replying.
            res = robot.execute(traj, traj_ok, traj_info, time_scale=DEPLOY_TIME_SCALE,
                                force=True, lift_m=LIFT_M)
        finally:
            # Stop the clip whatever the replay did (refusal, timeout, Ctrl-C).
            try:
                rec = cam.record_stop()
                print(f"[deploy] recording stopped: {rec['frames']} frames, "
                      f"{rec['duration_s']:.1f} s -> {rec['path']}"
                      + (f" (recorder error: {rec['error']})" if rec.get("error") else ""))
            except Exception as e:
                print(f"[deploy] record_stop failed: {e}")
    if res.get("status") != "ok":
        # A refusal is not a crash: the server declines to move an arm that is
        # somewhere unexpected. Report it and leave the grasp cached.
        print(f"[deploy] robot_server refused the replay: {res.get('message')}")
        return
    print(f"[deploy] replay done (executed={res.get('executed')})")
    if res.get("lift_m"):
        print(f"[deploy] lift: rose {res.get('lifted_m', 0.0) * 100:.1f} cm of "
              f"{res['lift_m'] * 100:.0f} cm requested"
              + ("" if res.get("lift_ok") else " (stalled or timed out)"))

    # --- where did it actually end up? -----------------------------------------
    # Both measurements below were taken BEFORE the lift, right as the replay
    # finished, so they compare against the plan rather than the raised pose.
    # Two independent checks on the same move:
    #   joints  — measured vs the last planned waypoint. Unambiguous, and the
    #             number to trust: it is the quantity the controller servos on.
    #   tcp     — measured gripper_tcp vs deploy_pose, the pose IK was given.
    #             read_tcp_pose() shifts the arm's own FK onto the gripper_tcp
    #             frame so the two are comparable at all (see its docstring).
    #             This one also carries every calibration error in the chain, so
    #             treat a few cm here as "check the C2R", not "the replay failed".
    # Both come back in the server's reply — it measured them lab-side the
    # instant the move finished, so there is no second round trip here.
    print("\n[check] where the arm finished")
    q_meas   = np.asarray(res["q"], dtype=float)
    q_goal   = np.asarray(traj, dtype=float)[-1, :7]
    tcp_meas = np.asarray(res["tcp_pose"], dtype=float)
    tcp_goal = np.asarray(deploy_pose, dtype=float)

    print(f"[check] joints measured : {np.round(q_meas, 4).tolist()}")
    print(f"[check] joints planned  : {np.round(q_goal, 4).tolist()}")
    print(f"[check] joint error     : {np.round(q_meas - q_goal, 4).tolist()}")
    print(f"[check]   |error| = {np.linalg.norm(q_meas - q_goal):.4f} rad "
          f"(worst joint {np.abs(q_meas - q_goal).max():.4f})")
    print(f"[check] tcp xyz measured: {np.round(tcp_meas[:3], 4).tolist()} m")
    print(f"[check] tcp xyz target  : {np.round(tcp_goal[:3], 4).tolist()} m")
    print(f"[check]   position error = {np.linalg.norm(tcp_meas[:3] - tcp_goal[:3]) * 100:.2f} cm")
    print(f"[check] tcp rpy measured: {np.round(tcp_meas[3:], 4).tolist()} rad")
    print(f"[check] tcp rpy target  : {np.round(tcp_goal[3:], 4).tolist()} rad")


# --- interactive loop ---------------------------------------------------------
# The expensive startup (imports, clients, the servers' solver/scene caches)
# happens once; every cycle after that only pays for the pipeline itself.
# Empty input keeps the previous value, so plain enter-enter reruns the last
# task/target. Ctrl-C during a run aborts just that cycle; Ctrl-C (or Ctrl-D)
# at the prompt exits.
def main():  # config values seed the first prompt
    while True:
        try:
            task = input("Task:")
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] done")
            return
        try:
            run_once(task)
        except KeyboardInterrupt:
            print("\n[loop] cycle aborted; clients stay warm")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[loop] cycle failed ({e}); clients stay warm — press enter to go again")


if __name__ == "__main__":
    main()
