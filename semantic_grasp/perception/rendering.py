"""Rendering helpers: pyrender multi-view generation, view labeling."""

import io
import logging
import math
import os
import threading

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# pyrender under PYOPENGL_PLATFORM=egl is not thread-safe: concurrent
# OffscreenRenderer use across threads segfaults / fails to create the EGL
# context. When several VLM stages render at once (main_thompson runs them in
# parallel), this serializes just the GL-critical section — renderers are quick
# next to the per-view network fan-out, so the parallelism is barely affected.
_RENDER_LOCK = threading.Lock()

import cv2
import numpy as np
import open3d as o3d
import pyrender
import trimesh
from PIL import Image

from ..config import (
    CAMERA_FOV_Y, GRIPPER_MAX_WIDTH_M, IMG_WIDTH, IMG_HEIGHT, VIEWPOINTS,
)
from .geometry import (
    build_intrinsic_matrix, glb_to_robot_matrix, gripper_R_from_approach,
    look_at, obb_corners, obb_edges, orientation_candidates,
    oriented_bounding_box, spherical_to_camera_pose,
)

log = logging.getLogger("pipeline")

# Distinct colors for the 5 OBB-face approach arrows (RGBA). Deliberately no pure
# red, so the arrows never blend with the red grasp-point sphere. Hues are spread
# far apart around the wheel and kept saturated/dark enough to read on the white
# render background, so both the eye and the VLM can tell them apart — INCLUDING
# under the renderer's shading, which darkens every face: the old yellow had to
# be darkened for the white background and then read as orange, and the old
# purple converged with shaded blue, so those two slots are now black and
# magenta, which stay far from every other color at any brightness. The names in
# ARROW_COLOR_NAMES (parallel order) are what the VLM is asked to choose from.
ARROW_COLORS = [
    [255, 100, 0, 255],    # orange
    [0, 150, 0, 255],      # green
    [0, 60, 255, 255],     # blue
    [25, 25, 25, 255],     # black
    [255, 0, 190, 255],    # magenta
]
ARROW_COLOR_NAMES = ["orange", "green", "blue", "black", "magenta"]


def _extract_meshes(mesh):
    """(pose, Trimesh) pairs plus every vertex in the common frame, from either a
    trimesh Scene or a bare Trimesh."""
    if isinstance(mesh, trimesh.Scene):
        pairs = []
        for node_name in mesh.graph.nodes_geometry:
            pose, geom_name = mesh.graph[node_name]
            geom = mesh.geometry[geom_name]
            if isinstance(geom, trimesh.Trimesh):
                pairs.append((pose, geom))
    else:
        pairs = [(np.eye(4), mesh)]
    all_verts = np.vstack([m.vertices @ t[:3, :3].T + t[:3, 3] for t, m in pairs])
    return pairs, all_verts


def _arrow_mesh(start, direction, length, color, shaft_radius):
    """A solid arrow (shaft cylinder + cone head) of `length`, starting at `start`
    and pointing along `direction`."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    start = np.asarray(start, dtype=float)
    head_len = 0.35 * length
    shaft_end = start + direction * (length - head_len)

    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, segment=np.array([start, shaft_end]))
    head = trimesh.creation.cone(radius=shaft_radius * 2.2, height=head_len)
    head.apply_transform(
        trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction))
    head.apply_translation(shaft_end)

    arrow = trimesh.util.concatenate([shaft, head])
    arrow.visual.face_colors = color
    return arrow


# Vertical spacing (robot metres) between the SIDE forks' heights. The four
# horizontal approaches would otherwise draw their forks through the same slab
# of space — an antiparallel pair (same closing axis, opposite sides) coincides
# almost exactly — and the overlapping colors become unreadable. Each side fork
# is lifted/lowered to its own height around the anchor instead. 2 cm clears
# the 1 cm finger width with air to spare while staying small next to the 8 cm
# gap, so the fit-vs-too-wide reading of each fork is unchanged.
FORK_STAGGER_M = 0.02


def _jaw_glyph_mesh(face_pt, n, close_dir, scale, color, finger_len):
    """The gripper's jaws for one approach arrow, drawn TO SCALE as a two-tined
    FORK: two long parallel fingers whose inner faces sit exactly
    GRIPPER_MAX_WIDTH_M apart, joined at the gripper (outer) end by a crossbar.
    The fingers run inward along the approach direction (-n) for `finger_len`, so
    they lie ALONGSIDE the whole depth of the object the gripper would slide onto.

    This makes the feasibility check unmistakable from any side view: if the part
    fits, the object sits cleanly BETWEEN the two fingers with an air gap on each
    side; if it is wider than the 8 cm opening, the object overflows and the
    fingers cut THROUGH it along their length. Short tip-pads (the old glyph) were
    easy to misread, especially seen edge-on. Everything is in GLB render units;
    `scale` converts the metric finger sizes (robot metres -> GLB), `n` is the
    outward face normal and `close_dir` the jaw closing axis (both unit GLB
    vectors), and `finger_len` is how far the fingers reach inward (GLB units,
    sized to the object's extent along the approach so they span it).
    """
    gap = GRIPPER_MAX_WIDTH_M / scale     # true 8 cm opening at object scale
    fin_t = 0.006 / scale                 # finger thickness (along closing axis)
    fin_w = 0.010 / scale                 # finger width (the third axis)
    bar_t = 0.008 / scale                 # crossbar thickness (along approach)
    fdir = -np.asarray(n, dtype=float)    # fingers reach inward, along approach
    t3 = np.cross(n, close_dir)
    t3 /= np.linalg.norm(t3) + 1e-12
    R = np.column_stack([close_dir, t3, fdir])   # box local x/y/z -> close/third/inward

    # Crossbar sits just OUTSIDE the near face (the gripper side, at the arrow
    # tip); the fingers extend inward from there across the object.
    base = np.asarray(face_pt, dtype=float) + n * bar_t
    parts = []
    for s in (+1.0, -1.0):                # the two fingers
        box = trimesh.creation.box(extents=[fin_t, fin_w, finger_len])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = base + s * (gap + fin_t) / 2.0 * close_dir + fdir * (finger_len / 2.0)
        box.apply_transform(T)
        parts.append(box)
    bar = trimesh.creation.box(extents=[gap + 2 * fin_t, fin_w, bar_t])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = base - fdir * (bar_t / 2.0)
    bar.apply_transform(T)
    parts.append(bar)

    glyph = trimesh.util.concatenate(parts)
    glyph.visual.face_colors = color
    return glyph


def render_views(mesh) -> dict:
    """Render every viewpoint of an in-memory trimesh Scene/Trimesh. Returns a
    metadata dict whose per-view entries carry the rendered PNG bytes and depth
    array."""
    meshes, all_verts = _extract_meshes(mesh)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    centroid = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)
    cam_radius = extent * 1.2

    py_scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 0.0],
        ambient_light=[0.3, 0.3, 0.3],
    )
    for pose, m in meshes:
        py_scene.add(pyrender.Mesh.from_trimesh(m, smooth=True), pose=pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = centroid + np.array([extent, extent, extent])
    py_scene.add(light, pose=light_pose)

    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    light2_pose = np.eye(4)
    light2_pose[:3, 3] = centroid + np.array([-extent, -extent * 0.5, -extent])
    py_scene.add(light2, pose=light2_pose)

    camera = pyrender.PerspectiveCamera(
        yfov=CAMERA_FOV_Y,
        aspectRatio=IMG_WIDTH / IMG_HEIGHT,
        znear=cam_radius * 0.01,
        zfar=cam_radius * 10.0,
    )
    cam_node = py_scene.add(camera, pose=np.eye(4))
    K = build_intrinsic_matrix(CAMERA_FOV_Y, IMG_WIDTH, IMG_HEIGHT)

    metadata = {
        "image_width": IMG_WIDTH,
        "image_height": IMG_HEIGHT,
        "fov_y_rad": CAMERA_FOV_Y,
        "intrinsic_K": K.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "centroid": centroid.tolist(),
        "views": [],
    }

    with _RENDER_LOCK:                      # pyrender/EGL: one renderer at a time
        renderer = pyrender.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
        for idx, (label, az, el) in enumerate(VIEWPOINTS):
            cam_to_world = spherical_to_camera_pose(az, el, cam_radius, target=centroid)
            world_to_cam = np.linalg.inv(cam_to_world)
            py_scene.set_pose(cam_node, pose=cam_to_world)
            color, depth = renderer.render(py_scene)

            buf = io.BytesIO()
            Image.fromarray(color).save(buf, format="PNG")

            metadata["views"].append({
                "index": idx,
                "label": label,
                "azimuth_deg": az,
                "elevation_deg": el,
                "cam_to_world": cam_to_world.tolist(),
                "world_to_cam": world_to_cam.tolist(),
                "image_png": buf.getvalue(),
                "depth": depth.astype(np.float32),
            })
            log.info(f"  [{idx:02d}] {label:15s} az={az:4d} el={el:4d}")

        renderer.delete()
    return metadata


def _approach_normals(all_verts, obb_axes, up, g2r=None):
    """The 5 approach face normals: top + 4 sides of the OBB, bottom dropped.

    With `g2r` (GLB->robot matrix) the OBB is Open3D's on the ROBOT-frame verts —
    identical to _candidate_poses_from_obb, so the arrows equal the pipeline's real
    approach candidates — and the bottom face is the most anti-robot-+z one.
    Without it, falls back to the trimesh GLB OBB axes and the `up` direction.

    Returns (glb_normals, report_normals, obb_axes_ext): unit outward normals in
    the GLB render frame (for drawing) and in the frame callers should report
    ("robot" when g2r is given, else the same GLB vectors), plus the box axes and
    full extents in the report frame as (axes (3, 3) unit COLUMNS, extents (3,)) —
    used to draw each jaw glyph at its best-fitting closing axis.
    """
    if g2r is not None:
        scale = float(np.linalg.norm(g2r[:3, 0]))
        rot = g2r[:3, :3] / scale                # pure rotation, GLB -> robot
        verts_robot = (g2r @ np.c_[all_verts, np.ones(len(all_verts))].T).T[:, :3]
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(
                np.ascontiguousarray(verts_robot, dtype=np.float64)))
        R = np.asarray(obb.R, dtype=float)       # columns = box axes (robot frame)
        ext = np.asarray(obb.extent, dtype=float)
        normals = np.array([s * R[:, i] for i in range(3) for s in (+1.0, -1.0)])
        bottom = int(np.argmin(normals @ np.array([0.0, 0.0, 1.0])))
        report = np.delete(normals, bottom, axis=0)
        glb = (rot.T @ report.T).T               # robot -> GLB directions
    else:
        R = obb_axes / np.linalg.norm(obb_axes, axis=0, keepdims=True)
        ext = np.array([np.ptp(all_verts @ R[:, i]) for i in range(3)])
        normals = np.array([s * R[:, i] for i in range(3) for s in (+1.0, -1.0)])
        glb = report = np.delete(normals, int(np.argmin(normals @ up)), axis=0)

    glb = glb / np.linalg.norm(glb, axis=1, keepdims=True)
    report = report / np.linalg.norm(report, axis=1, keepdims=True)
    return glb, report, (R, ext)


# The three stacked scale bars drawn under the object for get_grasp_width, TOP TO
# BOTTOM. Absolute lengths in ROBOT METERS — unlike the approach arrows these are
# not derived from the object; they are the fixed choices the VLM picks between,
# and the red bar doubles as the gripper's full 8 cm opening. The order also
# encodes the vote tie-break: earlier wins (red over green over blue).
RULER_BARS = [
    ("red",   GRIPPER_MAX_WIDTH_M, [220, 0, 0, 255]),
    ("green", 0.05,                [0, 160, 0, 255]),
    ("blue",  0.02,                [0, 90, 255, 255]),
]
RULER_COLOR_NAMES = [name for name, _, _ in RULER_BARS]


def _ruler_rows(lengths, all_verts, cam_pose, anchor, margin, spacing):
    """Where to draw the stacked scale bars in THIS view: one (start, end) pair per
    entry of `lengths`, top to bottom, in the GLB render frame.

    The bars are placed for the camera rather than left lying in the scene, because
    a scale bar is only honest if it is the same size on screen as the thing it
    measures. Each bar is laid out entirely in the image plane — spanning the
    camera's right axis (so it is never foreshortened), at the exact DEPTH of
    `anchor` (the grasp point, else the box center) — and the stack is centered
    under the object and dropped below it.

    Depth is what makes this necessary. A bar resting on the ground sits below the
    object, so a camera looking up from underneath sees it far nearer than the
    object and renders it much too big — measure against that and every width comes
    out proportionally too small. Sliding the bars only along the camera's right/up
    axes moves them within the image without changing their distance, so each bar's
    apparent size stays exactly the apparent size of its length at the object.

    The drop and the centering are computed in SCREEN space, not 3D: each vertex is
    taken at its projected screen position (its in-plane offset divided by its own
    depth), because a part of the object nearer to the camera than the anchor
    spills lower and wider on screen than its 3D coordinates suggest. The stack
    sits fully below the lowest point of the object's silhouette, so the mesh can
    never overdraw it — the bars are depth-tested scene geometry, and the VLM must
    see all of them or the choice they offer grounds nothing.

    `margin` (gap under the silhouette) and `spacing` (drop between bar centerlines)
    are in the same units as `lengths`; `margin` must exceed how far the end ticks
    reach up and `spacing` how far a tick reaches toward its neighbor (_ruler_mesh).
    """
    pose = np.asarray(cam_pose, dtype=float)
    eye, right, cam_up, fwd = pose[:3, 3], pose[:3, 0], pose[:3, 1], -pose[:3, 2]
    anchor = np.asarray(anchor, dtype=float)
    depth = float((anchor - eye) @ fwd)

    rel = np.asarray(all_verts, dtype=float) - eye
    z = rel @ fwd
    vis = z > 1e-9
    sx = (rel[vis] @ right) / z[vis]        # screen positions (angles): where each
    sy = (rel[vis] @ cam_up) / z[vis]       # vertex lands, regardless of its depth

    # In-plane offsets from the anchor at the anchor's depth: centered on the
    # silhouette's horizontal extent, below its lowest screen point.
    a = depth * 0.5 * (float(sx.min()) + float(sx.max())) - float((anchor - eye) @ right)
    b = depth * float(sy.min()) - float((anchor - eye) @ cam_up) - margin

    rows = []
    for i, length in enumerate(lengths):
        base = anchor + a * right + (b - i * spacing) * cam_up
        rows.append((base - 0.5 * length * right, base + 0.5 * length * right))
    return rows


def _ruler_mesh(start, end, tick_dir, tick_len, shaft_radius, color):
    """One scale bar: the rod plus a tick at each end (perpendicular to it, along
    `tick_dir`, `tick_len` end to end), so where the measured length starts and
    stops is unambiguous."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    parts = [trimesh.creation.cylinder(radius=shaft_radius, segment=np.array([start, end]))]
    tick = tick_len * np.asarray(tick_dir, dtype=float)
    for cap in (start, end):
        parts.append(trimesh.creation.cylinder(
            radius=shaft_radius, segment=np.array([cap - 0.5 * tick, cap + 0.5 * tick])))
    ruler = trimesh.util.concatenate(parts)
    ruler.visual.face_colors = color
    return ruler


def render_obb_corners(mesh, standoff=1.8, draw_obb=True,
                       grasp_points=None, sphere_radius=0.008,
                       transform=None, object_name=None,
                       draw_arrows=False, arrow_length=0.04,
                       draw_ruler=False, uniform_radius=False,
                       mesh_alpha=1.0, jaw_axis="bestfit"):
    """Render an in-memory mesh from the 8 corners and 12 edge midpoints of its
    oriented bounding box (20 viewpoints in total).

    Same pyrender/EGL setup, mesh extraction and look-at convention as
    render_views (the renderer behind VLM_Predict_XYZ): the mesh is shown in its
    native GLB frame. The camera sits at each OBB corner and edge midpoint,
    pulled out along the center->point ray by `standoff` (1.0 sits exactly on
    the point and near-clips the object; ~1.8 frames the whole object), aimed
    back at the box center.

    Edge midpoints lie closer to the center than corners do, so on an elongated
    object those views land much nearer the surface and fill the frame. With
    `uniform_radius` every viewpoint is instead pushed out to the same distance
    (`standoff` x the OBB circumradius), which frames all 20 views at a
    comparable scale. Corner views are unchanged either way — they already sit at
    the circumradius.

    `grasp_points` ((N, 3) or (3,)) are drawn as red spheres. When `transform`
    is given (a single object entry or a full transforms dict resolved via
    `object_name`) the points are robot-base-frame and mapped back into the GLB
    frame (radius scaled to match); otherwise they are already GLB-frame.

    `mesh_alpha` < 1.0 renders the object as a translucent shell so grasp points
    that landed INSIDE it stay visible. Leave it at 1.0 for anything a VLM will
    see — the object must look like itself in those photos. It exists for the
    output diagnostics only (_dump_cluster_views).

    With `draw_arrows`, 5 differently-colored arrows are drawn, one per kept OBB
    face (top + 4 sides; see _approach_normals). Each arrow points INWARD along
    its face normal (the approach direction) and sits entirely OUTSIDE the box:
    the tip lands on the OBB face — at the first grasp point's projection onto
    it when `grasp_points` is given, else at the face center — with a shaft
    `arrow_length` m long (robot-frame meters when `transform` is given).

    With `draw_ruler`, three stacked scale bars of FIXED metric length are drawn
    under the object — RULER_BARS, top to bottom: red 8 cm (the gripper's full
    opening), green 5 cm, blue 2 cm — the choices get_grasp_width's VLM picks
    between. Nothing but the colored bars is drawn: the color->length mapping
    rides in the prompt, not on the image. The stack is re-placed per camera
    (_ruler_rows): flat in the image plane, at the grasp point's depth, fully
    below the object's silhouette, so on every photo each bar shows exactly what
    its length looks like at the object. Requires `transform` (the lengths are
    robot meters, meaningless in bare GLB units). Each record then also carries
    "ruler_bars": [{"color", "length_m", "ends" ((2, 3), GLB frame)}, ...], top
    to bottom.

    Returns (records, arrows).

    records: a list of 20 entries, the 8 corners first then the 12 edge
    midpoints, each:
        {"index", "kind" ("corner"|"edge"), "signs" (e.g. "+-+" for a corner,
         "+0-" for an edge whose midpoint is centered along one axis),
         "corner" (3,) the viewpoint position, "eye" (3,),
         "image" (H, W, 3) RGB uint8}

    arrows: the color->approach mapping for the drawn arrows (empty unless
    draw_arrows), one per arrow in ARROW_COLOR_NAMES order:
        {"color" (name), "frame" ("robot" when `transform` given, else "glb"),
         "normal" (3,) outward face normal, "approach" (3,) inward approach
         direction (= -normal, the way the arrow points)}
    When a `transform` is given each entry also carries the gripper orientation
    in the isaac_render_client convention (approach axis antiparallel to the
    outward normal -- the rpy main.py feeds to IK):
        "rpy" (3,) base orientation (xyz euler, robot frame),
        "rpy_candidates" [base, 90deg-twist] -- the two orientations that face.
    """
    meshes, all_verts = _extract_meshes(mesh)
    extent = float(np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0)))

    center, axes, half = oriented_bounding_box(all_verts)
    corners, signs = obb_corners(center, axes, half)
    log.info(f"OBB center={center}, extents={2.0 * half}")

    # High ambient + a camera-following headlight (set per view in the loop
    # below) keeps every viewpoint evenly lit, so the arrow colors read true
    # from all 20 views instead of being darkened by self-shading.
    py_scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 0.0],
        ambient_light=[0.65, 0.65, 0.65],
    )
    for pose, m in meshes:
        if mesh_alpha < 1.0:
            # Glass shell: the grasp spheres are opaque and drawn in the opaque
            # pass, so a point that ended up INSIDE the object still shows
            # through the surface blended over it. doubleSided keeps the far
            # wall drawn too, so the object reads as a volume rather than a
            # cut-away. Diagnostic only — never used for photos sent to a VLM,
            # which must show the object as it really looks.
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.75, 0.75, 0.78, mesh_alpha],
                metallicFactor=0.0, roughnessFactor=0.9,
                alphaMode="BLEND", doubleSided=True)
            py_scene.add(pyrender.Mesh.from_trimesh(m, material=material, smooth=True),
                         pose=pose)
        else:
            py_scene.add(pyrender.Mesh.from_trimesh(m, smooth=True), pose=pose)

    if draw_obb:
        edge_r = max(extent * 0.004, 1e-4)
        for i, j in obb_edges(signs):
            cyl = trimesh.creation.cylinder(
                radius=edge_r, segment=np.array([corners[i], corners[j]]))
            cyl.visual.face_colors = [0, 200, 255, 255]
            py_scene.add(pyrender.Mesh.from_trimesh(cyl, smooth=False))

    # Everything is drawn in the GLB render frame. When a `transform` is given the
    # caller speaks robot frame, so build the GLB<->robot map once: it converts
    # points and gives the uniform scale (robot meters -> GLB lengths).
    g2r, scale = None, 1.0
    up = np.array([0.0, 1.0, 0.0])               # GLB is Y-up (fallback)
    if transform is not None:
        g2r = glb_to_robot_matrix(transform, mesh=mesh, object_name=object_name)
        scale = float(np.linalg.norm(g2r[:3, 0]))

    arrows, grasp_anchor = [], None
    if grasp_points is not None:
        pts = np.asarray(grasp_points, dtype=float).reshape(-1, 3)
        if g2r is not None:
            pts = (np.linalg.inv(g2r) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        grasp_anchor = pts[0]

        for pt in pts:
            sphere = trimesh.creation.uv_sphere(radius=sphere_radius / scale, count=[16, 16])
            sphere.apply_translation(pt)
            sphere.visual.face_colors = [255, 0, 0, 255]  # red, like show_grasp_on_mesh
            py_scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))

    if draw_arrows:
        # Arrows anchor at the first grasp point when one is given, else at the
        # OBB center — projected onto each face, that puts the tips at the face
        # centers.
        anchor = center if grasp_anchor is None else grasp_anchor
        glb_normals, report_normals, (obb_R, obb_ext) = _approach_normals(
            all_verts, axes, up, g2r)
        length, shaft_r = arrow_length / scale, 0.06 * arrow_length / scale

        # Stagger the SIDE forks vertically (see FORK_STAGGER_M): each fork
        # whose approach is roughly horizontal gets its own height step,
        # centred on the anchor, so no two forks share the same slab of space.
        # The top fork keeps the anchor height — nothing overlaps it there.
        # Only meaningful with a transform (the report normals are robot-frame
        # and the step is metric), which is also the only case forks are drawn.
        fork_offsets = [0.0] * len(glb_normals)
        if transform is not None:
            z_glb = (g2r[:3, :3] / scale).T @ np.array([0.0, 0.0, 1.0])
            z_glb /= np.linalg.norm(z_glb) + 1e-12
            side = [abs(float(v @ np.array([0.0, 0.0, 1.0]))) < 0.6
                    for v in report_normals]
            n_side, k = sum(side), 0
            for i, is_side in enumerate(side):
                if is_side:
                    fork_offsets[i] = (k - (n_side - 1) / 2.0) * FORK_STAGGER_M / scale
                    k += 1

        for i, (n, vec, color, cname) in enumerate(zip(
                glb_normals, report_normals, ARROW_COLORS, ARROW_COLOR_NAMES)):
            # Whole arrow OUTSIDE the OBB, pointing inward: project the anchor
            # onto this face's plane (the verts' supporting plane along n) and
            # land the tip right on the face.
            face_pt = anchor + (np.max(all_verts @ n) - anchor @ n) * n
            arrow = _arrow_mesh(face_pt + n * length, -n, length, color, shaft_r)
            py_scene.add(pyrender.Mesh.from_trimesh(arrow, smooth=False))
            entry = {
                "color": cname,
                "frame": "robot" if transform is not None else "glb",
                "normal": vec.tolist(),         # outward face normal
                "approach": (-vec).tolist(),    # inward approach direction
            }
            if transform is not None:
                # Gripper orientation for this face, in the same convention as
                # the pipeline's approach candidates (orientation_candidates):
                # [base, 90deg-twist], base first -- the rpy main.py feeds to IK.
                rpys = [r.tolist() for r in orientation_candidates(vec)]
                entry["rpy"] = rpys[0]
                entry["rpy_candidates"] = rpys
                # Jaws drawn to scale at this arrow, closing across the object's
                # best-fitting axis so the glyph is an HONEST feasibility cue: of
                # the two OBB axes perpendicular to this approach, the fingers
                # would rotate to close across the SMALLER one (its extent is the
                # tightest the jaws must span). If even that is wider than the 8 cm
                # gap, no approach from this arrow can grip the part. `jaw_axis`
                # ("base") falls back to the base gripper orientation's closing
                # axis (arbitrary), which can show a feasible arrow's jaws spanning
                # the long side and read as a bad grip.
                if jaw_axis == "bestfit":
                    comp = np.abs(obb_R.T @ np.asarray(vec))   # |normal . axis_i|
                    approach_ax = int(np.argmax(comp))         # axis the arrow runs along
                    perp = [i for i in range(3) if i != approach_ax]
                    close_ax = perp[int(np.argmin([obb_ext[i] for i in perp]))]
                    close_robot = obb_R[:, close_ax]
                else:
                    close_robot = gripper_R_from_approach(-np.asarray(vec))[:, 1]
                close_glb = (g2r[:3, :3] / scale).T @ close_robot
                close_glb /= np.linalg.norm(close_glb) + 1e-12
                # Fingers reach in alongside the object far enough to read as a
                # fork, but capped so a long-axis approach doesn't draw bars the
                # full length of the object and clutter the frame. The 8 cm gap at
                # the crossbar (not the finger length) is what shows the fit.
                gap_glb = GRIPPER_MAX_WIDTH_M / scale
                finger_len = min(float(np.ptp(all_verts @ n)), 1.5 * gap_glb) + 0.4 * gap_glb
                glyph = _jaw_glyph_mesh(face_pt + fork_offsets[i] * z_glb, n,
                                        close_glb, scale, color, finger_len)
                py_scene.add(pyrender.Mesh.from_trimesh(glyph, smooth=False))
            arrows.append(entry)

    # The scale bars are the one thing not baked into the scene here: they are
    # re-placed for each camera below (_ruler_rows), so each bar's size on screen
    # always means its stated number of centimeters at the object. Their lengths
    # are absolute robot meters, so they need the GLB<->robot scale to exist.
    if draw_ruler:
        if transform is None:
            raise ValueError("draw_ruler requires `transform`: the scale bars are "
                             "absolute metric lengths (see RULER_BARS)")
        bar_glb = [m / scale for _, m, _ in RULER_BARS]
        bar_L = max(bar_glb)                    # sizing/spacing reference (GLB units)

    # Soft headlight, re-aimed with the camera each view so the visible side is
    # always lit head-on (a DirectionalLight shines along its pose's -z, which is
    # also the camera's look direction). Kept gentle so the bright ambient stays
    # dominant and shading barely darkens the colors.
    headlight = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.2)
    head_node = py_scene.add(headlight, pose=np.eye(4))

    circumradius = float(np.linalg.norm(corners[0] - center))   # equal for all 8
    cam_dist = standoff * circumradius
    camera = pyrender.PerspectiveCamera(
        yfov=CAMERA_FOV_Y,
        aspectRatio=IMG_WIDTH / IMG_HEIGHT,
        znear=cam_dist * 0.01,
        zfar=cam_dist * 10.0,
    )
    cam_node = py_scene.add(camera, pose=np.eye(4))

    # Viewpoints: the 8 OBB corners plus the midpoint of each of the 12 edges,
    # 20 in total. The camera sits at each, pushed out from the center by
    # `standoff`, looking back at the center.
    viewpoints = [(corner, s, "corner") for corner, s in zip(corners, signs)]
    for i, j in obb_edges(signs):
        midpoint = (corners[i] + corners[j]) / 2.0
        # The two corners differ along exactly one axis; that axis reads 0 at the
        # edge midpoint, the other two keep their shared sign.
        edge_sign = np.where(signs[i] == signs[j], signs[i], 0.0)
        viewpoints.append((midpoint, edge_sign, "edge"))

    def _sign_str(vec):
        return "".join("+" if v > 0 else "-" if v < 0 else "0" for v in vec)

    records = []
    with _RENDER_LOCK:                      # pyrender/EGL: one renderer at a time
        renderer = pyrender.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
        for idx, (pos, s, kind) in enumerate(viewpoints):
            ray = pos - center
            if uniform_radius:
                ray = circumradius * ray / np.linalg.norm(ray)
            eye = center + standoff * ray
            cam_pose = look_at(eye, center)
            py_scene.set_pose(cam_node, pose=cam_pose)
            py_scene.set_pose(head_node, pose=cam_pose)   # headlight follows the camera

            ruler_nodes, ruler_rows = [], None
            if draw_ruler:
                ruler_rows = _ruler_rows(
                    bar_glb, all_verts, cam_pose,
                    anchor=center if grasp_anchor is None else grasp_anchor,
                    margin=0.15 * bar_L, spacing=0.25 * bar_L)
                for (_name, _m, rgba), ends in zip(RULER_BARS, ruler_rows):
                    ruler_nodes.append(py_scene.add(pyrender.Mesh.from_trimesh(
                        _ruler_mesh(*ends, cam_pose[:3, 1], 0.14 * bar_L,
                                    0.018 * bar_L, rgba),
                        smooth=False)))

            color, _ = renderer.render(py_scene)
            for node in ruler_nodes:
                py_scene.remove_node(node)             # re-placed for the next camera

            image = np.asarray(color)
            sign_str = _sign_str(s)
            record = {
                "index": idx,
                "kind": kind,
                "signs": sign_str,
                "corner": pos.tolist(),
                "eye": eye.tolist(),
                "image": image,
            }
            if draw_ruler:
                record["ruler_bars"] = [
                    {"color": name, "length_m": m, "ends": np.asarray(ends)}
                    for (name, m, _), ends in zip(RULER_BARS, ruler_rows)]
            records.append(record)
            log.info(f"  {kind} {idx} ({sign_str}) eye={eye}")
        renderer.delete()
    return records, arrows


def tile_grid(tiles, cols):
    """Stack same-sized BGR tiles into a `cols`-wide grid, padding the last row
    with black."""
    rows = math.ceil(len(tiles) / cols)
    tiles = list(tiles) + [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
    return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])


def corner_contact_sheet(records, cols=4, tile_w=400):
    """BGR contact sheet of render_obb_corners() records, each tile labeled with
    its corner index and +/- sign pattern."""
    first = np.asarray(records[0]["image"])
    tile_h = int(round(first.shape[0] * tile_w / first.shape[1]))
    tiles = []
    for rec in records:
        bgr = cv2.cvtColor(np.asarray(rec["image"]), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (tile_w, tile_h))
        label = f"{rec['index']} ({rec['signs']})"
        cv2.putText(bgr, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(bgr, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(bgr, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 2)
        tiles.append(bgr)
    return tile_grid(tiles, cols)
