"""Geometry helpers: transforms, unprojection, clustering."""

import logging
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors

from ..config import (
    COORD_SCALE, DEPTH_SEARCH_RADIUS, DBSCAN_MIN_SAMPLES,
)

log = logging.getLogger("pipeline")

_DEGENERATE_DOT = 0.95   # |approach . +x| above this -> use +z as the up reference
# unproject_predictions appends the overall mean as a last-resort candidate,
# unless a cluster centroid is already this close to it (m) — two grasp points
# a millimetre apart are the same grasp, and trying both wastes a refine pass.
_MEAN_DEDUPE_M = 0.001


def gripper_R_from_approach(z_g):
    """Full gripper rotation (columns [x_g, y_g, z_g], robot frame) whose local +z
    (the approach axis, the way the gripper points) is `z_g`.

    The in-plane axes are fixed by Gram-Schmidt against robot +x (falling back to
    robot +z when the approach is nearly along +x, the only degenerate case). This
    is deliberately chosen so that for the TOP face — approach axis straight down,
    z_g = [0,0,-1] — it reproduces R = Rx(pi), i.e. rpy [pi,0,0]."""
    z_g = np.asarray(z_g, dtype=float)
    z_g = z_g / (np.linalg.norm(z_g) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(z_g @ ref)) > _DEGENERATE_DOT:
        ref = np.array([0.0, 0.0, 1.0])
    x_g = ref - (ref @ z_g) * z_g
    x_g = x_g / (np.linalg.norm(x_g) + 1e-12)
    y_g = np.cross(z_g, x_g)
    return np.column_stack([x_g, y_g, z_g])


def orientation_candidates(outward_normal):
    """Two rpy (xyz, radians) candidates for an OBB face: a base orientation with
    the gripper approach axis ANTIPARALLEL to the outward normal (gripper points
    inward toward the object), plus a 90deg twist about that approach axis.

    The twist sign is chosen so the TOP face (outward normal +z) yields the two
    poses [pi,0,0] and [pi,0,pi/2] given as the correctness check. (Note: a +90deg
    turn about the OUTWARD normal equals a -90deg turn about the into-object
    approach axis — hence the -pi/2 below; flip the sign to swap which twist is
    listed first.)"""
    n = np.asarray(outward_normal, dtype=float)
    base = gripper_R_from_approach(-n)                   # approach = -normal (inward)
    twist = base @ Rot.from_euler("z", -math.pi / 2).as_matrix()
    return [Rot.from_matrix(base).as_euler("xyz"),
            Rot.from_matrix(twist).as_euler("xyz")]


def spherical_to_camera_pose(azimuth_deg, elevation_deg, radius, target=np.zeros(3)):
    """4x4 camera-to-world matrix (OpenGL convention: -Z forward, +Y up)."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.sin(el)
    z = radius * math.cos(el) * math.cos(az)
    eye = np.array([x, y, z]) + target

    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = -forward
    cam_to_world[:3, 3] = eye
    return cam_to_world


def look_at(eye, target, world_up=(0.0, 1.0, 0.0)):
    """4x4 camera-to-world (OpenGL: -Z forward, +Y up) from an eye + look target.

    Same convention as spherical_to_camera_pose, but for an arbitrary eye/target
    pair (used to place the camera at oriented-bounding-box corners)."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray(world_up, dtype=float)
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def oriented_bounding_box(points):
    """Oriented bounding box of an (N, 3) point set.

    Returns (center (3,), axes (3, 3) with unit-vector COLUMNS, half_extents (3,)).
    Wraps trimesh.bounds.oriented_bounds (the OBB of the convex hull): that gives
    the transform mapping points to an origin-centered axis-aligned box, so the
    box's placement in world space is its inverse."""
    import trimesh

    to_origin, extents = trimesh.bounds.oriented_bounds(
        np.asarray(points, dtype=float))
    box_to_world = np.linalg.inv(to_origin)
    center = box_to_world[:3, 3]
    axes = box_to_world[:3, :3]
    axes = axes / np.linalg.norm(axes, axis=0, keepdims=True)  # unit columns
    return center, axes, np.asarray(extents, dtype=float) / 2.0


def obb_corners(center, axes, half_extents):
    """The 8 corners (8, 3) of an OBB and their +/- sign pattern (8, 3) along the
    box axes (corner = center + sum_k sign_k * half_k * axis_k)."""
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    corners = np.asarray(center, dtype=float) + (signs * half_extents) @ np.asarray(axes).T
    return corners, signs


def obb_edges(signs):
    """The 12 OBB edges as index pairs into the corner array -- two corners are
    adjacent iff their sign patterns differ along exactly one axis."""
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(signs[i] != signs[j]) == 1:
                edges.append((i, j))
    return edges


def build_intrinsic_matrix(fov_y, width, height):
    """3x3 camera intrinsic matrix K from vertical FOV and image size."""
    fy = height / (2.0 * math.tan(fov_y / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def point_to_pixel(point, coord_scale, img_w, img_h):
    """Convert Gemini [y, x] in 0-{coord_scale} to pixel (u, v)."""
    y_norm, x_norm = point
    pixel_u = x_norm / coord_scale * img_w
    pixel_v = y_norm / coord_scale * img_h
    return pixel_u, pixel_v


def get_depth_at_pixel(depth_map, u, v, radius=DEPTH_SEARCH_RADIUS):
    h, w = depth_map.shape
    ui, vi = int(round(u)), int(round(v))
    if 0 <= vi < h and 0 <= ui < w and depth_map[vi, ui] > 0:
        return depth_map[vi, ui], u, v

    best_depth, best_dist = None, float("inf")
    best_u, best_v = None, None
    for r in range(1, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dy) != r and abs(dx) != r:
                    continue
                ny, nx = vi + dy, ui + dx
                if 0 <= ny < h and 0 <= nx < w and depth_map[ny, nx] > 0:
                    dist = (dy ** 2 + dx ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_depth = depth_map[ny, nx]
                        best_u, best_v = float(nx), float(ny)
        if best_depth is not None:
            return best_depth, best_u, best_v
    return None, None, None


def unproject_pixel_to_world(u, v, depth, K_inv, cam_to_world):
    """Unproject a 2D pixel to 3D world space (handles OpenCV<->OpenGL flip)."""
    pixel_h = np.array([u, v, 1.0])
    cam_cv = depth * (K_inv @ pixel_h)
    # OpenCV [X, Y, Z] -> OpenGL [X, -Y, -Z]
    cam_gl = np.array([cam_cv[0], -cam_cv[1], -cam_cv[2]])
    world_point = cam_to_world @ np.array([*cam_gl, 1.0])
    return world_point[:3]


def project_world_to_pixel(points, cam_to_world, K):
    """Project (N, 3) world points to pixels -- the inverse of
    unproject_pixel_to_world, in the same OpenGL camera convention (-Z forward,
    +Y up), so it lines up with anything rendered through spherical_to_camera_pose
    or look_at.

    Returns ((N, 2) pixels, (N,) valid): `valid` is False for points on or behind
    the camera plane, whose pixel coordinates are meaningless."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, dtype=float))
    cam_gl = (world_to_cam @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    cam_cv = cam_gl * np.array([1.0, -1.0, -1.0])   # OpenGL -> OpenCV
    z = cam_cv[:, 2]
    valid = z > 1e-9
    uvw = (np.asarray(K, dtype=float) @ (cam_cv / np.where(valid, z, 1.0)[:, None]).T).T
    return uvw[:, :2], valid


def estimate_eps(pts, min_samples):
    """Estimate a good eps for DBSCAN using the k-distance graph."""
    k = min(min_samples, len(pts) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(pts)
    distances, _ = nn.kneighbors(pts)
    k_dists = np.sort(distances[:, -1])
    diffs = np.diff(k_dists)
    knee = np.argmax(diffs)
    eps = k_dists[knee]
    log.info(f"Auto eps: {eps:.4f} (knee at point {knee}/{len(pts)})")
    return eps


def cluster_points(points, eps=None, min_samples=DBSCAN_MIN_SAMPLES):
    """Cluster 3D points with DBSCAN, return (target_point, inliers, outliers).

    Replaces the old MAD-based robust_estimate with density-based clustering.
    Selects the best cluster by density score (count / spread) and returns
    its median as the target point.
    """
    points = np.array(points)
    if len(points) == 1:
        return points[0], points, np.array([]).reshape(0, 3)

    if len(points) < min_samples:
        # Too few points for DBSCAN -- just return median
        center = np.median(points, axis=0)
        return center, points, np.array([]).reshape(0, 3)

    if eps is None:
        eps = estimate_eps(points, min_samples)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = db.fit_predict(points)

    valid_labels = sorted(set(cluster_ids) - {-1})
    n_clusters = len(valid_labels)
    n_noise = (cluster_ids == -1).sum()
    log.info(f"DBSCAN: {n_clusters} cluster(s), {n_noise} noise point(s) (eps={eps:.4f})")

    if n_clusters == 0:
        # All noise -- fall back to median of all points
        center = np.median(points, axis=0)
        return center, points, np.array([]).reshape(0, 3)

    # Select best cluster by density score (count / spread)
    best_label = None
    best_score = -1
    for c in valid_labels:
        mask = cluster_ids == c
        count = mask.sum()
        spread = points[mask].var(axis=0).sum()
        score = count / (spread + 1e-6)
        median = np.median(points[mask], axis=0)
        log.info(
            f"  Cluster {c}: {count} points, spread={spread:.6f}, score={score:.1f}, "
            f"median=({median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f})"
        )
        if score > best_score:
            best_score = score
            best_label = c

    best_mask = cluster_ids == best_label
    inliers = points[best_mask]
    outliers = points[~best_mask]
    target_point = np.median(inliers, axis=0)

    log.info(f"  Best cluster: {best_label} (score={best_score:.1f}) (outliers = {len(points)-len(valid_labels)}), cluster amount = {n_clusters}")
    return target_point, inliers, outliers


def unproject_predictions(predictions, metadata):
    """Unproject the VLM's 2D points to 3D and reduce them to two candidates.

    Returns:
        grasp_candidates: np.ndarray of shape (n_clusters + 1, 3) — the mean of
            each HDBSCAN cluster, one row per place a subset of views agreed on,
            in the order the refine loop should try them: sorted by distance to
            the mean of ALL points, nearest first.

            That overall mean is then APPENDED as the final candidate, so it is
            tried only once every cluster has failed. It is deliberately last
            rather than first: averaging disagreeing views can land between them,
            where nobody pointed. It is dropped when a cluster already coincides
            with it (within _MEAN_DEDUPE_M), which is the usual single-cluster
            case — so the row count is n_clusters + 1 only when the mean is
            genuinely somewhere new.

            Falls back to (1, 3) holding that overall mean when HDBSCAN finds no
            cluster (too few points to run it, or every point marked noise) —
            purely so the caller still has something to grasp.
        all_3d_points: list of dicts for diagnostics
    """
    img_w = metadata["image_width"]
    img_h = metadata["image_height"]
    K = np.array(metadata["intrinsic_K"])
    K_inv = np.linalg.inv(K)

    needed_views = set(p["view_index"] for p in predictions)
    depth_maps = {vi: metadata["views"][vi]["depth"] for vi in needed_views}

    all_pts = []
    all_3d_points = []
    for pred in predictions:
        view_idx = pred["view_index"]
        label = pred["label"]
        point = pred["point"]

        u, v = point_to_pixel(point, COORD_SCALE, img_w, img_h)
        view = metadata["views"][view_idx]
        cam_to_world = np.array(view["cam_to_world"])
        depth_val, actual_u, actual_v = get_depth_at_pixel(depth_maps[view_idx], u, v)

        if depth_val is None:
            log.info(f"  [{view_idx:02d}] {view['label']:15s} no depth at ({u:.0f},{v:.0f})")
            continue

        world_pt = unproject_pixel_to_world(actual_u, actual_v, depth_val, K_inv, cam_to_world)
        log.info(
            f"  [{view_idx:02d}] {view['label']:15s} "
            f"({world_pt[0]:.4f}, {world_pt[1]:.4f}, {world_pt[2]:.4f})"
        )
        all_pts.append(world_pt)
        all_3d_points.append({
            "label": label,
            "x": float(world_pt[0]),
            "y": float(world_pt[1]),
            "z": float(world_pt[2]),
        })

    if not all_pts:
        centroid = np.array(metadata["centroid"])
        return centroid[np.newaxis], []

    pts = np.array(all_pts)
    overall_mean = np.mean(pts, axis=0)

    # HDBSCAN, not DBSCAN: the density of these points varies a lot run to run
    # (a part every view can see gets a tight blob, an occluded one a sparse
    # smear), and a single eps cannot fit both. HDBSCAN picks its own scale per
    # cluster, so there is no eps to estimate — which also removes the crash
    # where estimate_eps returned 0.0 on duplicate points and DBSCAN rejected it.
    cluster_centroids = []
    if len(pts) >= DBSCAN_MIN_SAMPLES:
        labels = HDBSCAN(min_cluster_size=DBSCAN_MIN_SAMPLES).fit_predict(pts)
        valid_labels = sorted(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        log.info(f"HDBSCAN: {len(valid_labels)} cluster(s), {n_noise} noise point(s) "
                 f"(min_cluster_size={DBSCAN_MIN_SAMPLES})")
        for c in valid_labels:
            centroid = np.mean(pts[labels == c], axis=0)
            cluster_centroids.append(centroid)
            log.info(
                f"  Cluster {c}: {int((labels==c).sum())} pts, "
                f"centroid=({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})"
            )
    else:
        log.info(f"Too few points for HDBSCAN ({len(pts)} < {DBSCAN_MIN_SAMPLES}), skipping")

    # The candidates are the cluster centroids — one per place a subset of views
    # agreed on — ordered by distance to the overall mean, since the cluster
    # nearest the whole-scene consensus is the most likely to be the real target;
    # the rest follow outward and still get their turn.
    #
    # The overall mean is then appended as a LAST-RESORT candidate. It is not a
    # cluster: averaging disagreeing views can land where nobody pointed (the
    # midpoint of a handle and a rim is neither), which is why it does not
    # compete with the clusters for first pick. But when every cluster has
    # failed it is still a reasonable thing to try, so it gets the final slot.
    if cluster_centroids:
        centroids = np.vstack(cluster_centroids)
        order = np.argsort(np.linalg.norm(centroids - overall_mean, axis=1))
        grasp_candidates = centroids[order]
        # Skip it when a cluster already sits there (e.g. a single cluster
        # holding every point) — the same grasp twice is a wasted refine pass.
        if np.linalg.norm(grasp_candidates - overall_mean, axis=1).min() > _MEAN_DEDUPE_M:
            grasp_candidates = np.vstack([grasp_candidates, overall_mean])
    else:
        # Last resort, not a preference: with no cluster there is nothing to
        # grasp, and returning empty would leave the refine loop with no point
        # to try at all.
        grasp_candidates = overall_mean[np.newaxis]
        log.info("HDBSCAN found no cluster; falling back to the overall mean")

    log.info(f"Grasp candidates ({len(grasp_candidates)} total, cluster centroids "
             f"nearest-to-mean first, overall mean last):")
    for i, c in enumerate(grasp_candidates):
        d = float(np.linalg.norm(c - overall_mean))
        log.info(f"  [{i}] ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})  "
                 f"{d:.4f} m from the overall mean"
                 + ("   <- overall mean" if d <= _MEAN_DEDUPE_M else ""))
    return grasp_candidates, all_3d_points


def rot_matrix_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


# GLB Y-up -> Isaac Z-up (+90 around X)
R_MESH_YUP_TO_ZUP = np.array([[1,  0,  0],
                                [0,  0, -1],
                                [0,  1,  0]], dtype=float)


def _mesh_vertices(mesh):
    """World-space vertices of a mesh given as a path, Scene, or Trimesh."""
    import trimesh

    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(str(mesh), force="mesh")
    elif isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return np.asarray(mesh.vertices, dtype=float)


def _build_glb_to_robot_matrix(obj_entry, mesh=None):
    """Build the 4x4 transform from GLB mesh space to robot base frame.

    `obj_entry`'s translation / rotation_wxyz / scale are already in the robot
    base frame (the sam3d server bakes in the camera->robot C2R extrinsics), so
    no camera->robot conversion happens here — only the GLB Y-up -> Z-up mesh
    axis fix and the uniform scale, both of which are about the mesh's own
    representation, not the frame.

    The translation is used AS-IS, z included. It used to be re-derived from
    `mesh` ("drop the object so its lowest vertex rests at z=0"), which put the
    VLM grasp points 3-13 mm above where the settled object actually sits in
    the IK/Isaac world (table top at robot z=-0.003, collision base planed
    2-10 mm) and dropped a target resting on another object to the table.
    Every caller passes the settled pose from grasp.settle(), so that z is the
    right one. `mesh` is still accepted so callers need not change; it is
    ignored.
    """
    scale = obj_entry["scale"]
    t_robot = np.array(obj_entry["translation"], dtype=float)
    q_robot = np.array(obj_entry["rotation_wxyz"], dtype=float)
    R_spawn = rot_matrix_from_quat(q_robot)
    R_total = R_spawn @ R_MESH_YUP_TO_ZUP

    M = np.eye(4)
    M[:3, :3] = scale * R_total
    M[:3, 3] = t_robot
    return M


def glb_point_to_robot(glb_point, obj_entry, mesh=None):
    """Convert a single GLB-space point to robot base frame."""
    M = _build_glb_to_robot_matrix(obj_entry, mesh)
    return (M @ np.array([*glb_point, 1.0]))[:3]


def glb_points_to_robot(points, obj_entry, mesh=None):
    """Convert an (N, 3) array of GLB-space points to robot base frame -> (N, 3)."""
    M = _build_glb_to_robot_matrix(obj_entry, mesh)
    pts_h = np.hstack([np.asarray(points), np.ones((len(points), 1))])
    return np.asarray((M @ pts_h.T).T[:, :3])


def resolve_obj_entry(transform, object_name=None):
    """Pick a single object entry from a full transforms dict
    ({"objects": {name: entry}}), or pass a single entry through unchanged."""
    if isinstance(transform, dict) and "objects" in transform:
        objs = transform["objects"]
        if object_name is not None:
            return objs[object_name]
        if len(objs) == 1:
            return next(iter(objs.values()))
        raise ValueError("transform has multiple objects; pass object_name to pick one")
    return transform


def glb_to_robot_matrix(transform, mesh=None, object_name=None):
    """Public wrapper for the 4x4 GLB->robot matrix of one object.

    `transform` may be a single object entry or a full transforms dict (resolved
    via `object_name`). The rotation block is scale*R: its column norm is the
    object's uniform scale, and the inverse maps robot-frame points/lengths back
    into the mesh's native GLB frame (as show_grasp_on_mesh does)."""
    return _build_glb_to_robot_matrix(resolve_obj_entry(transform, object_name), mesh=mesh)
