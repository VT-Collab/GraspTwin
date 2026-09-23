"""
=====================================================================
 ZMQ LangSAM Segmentation Server  —  REQ / REP
=====================================================================
 Receives an image + a list of text object prompts, runs language-guided
 segmentation (LangSAM), and sends the per-object masks back.

 SETUP
 -----
   pip install pyzmq opencv-python pillow numpy
   # plus lang-segment-anything, per its own install instructions

 RUN (two terminals)
 -------------------
   Terminal 1:  python seg_server.py
   Terminal 2:  python seg_client.py
=====================================================================
"""

import os
import sys

# Running `python servers/perception_server.py` (from the repo root OR from
# inside servers/) puts the script's own dir on sys.path, not the repo root, so
# `import semantic_grasp` fails. Add the repo root (parent of servers/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle

import cv2
import numpy as np
import zmq
from PIL import Image
from lang_sam import LangSAM


print("Loading LangSAM model (this can take a bit)...")
model = LangSAM()
print("Model ready.")
def segment_text(rgb, objects):
    """Text-prompt path ("langsam" request): LangSAM runs GroundingDINO on each
    name to find a box, then SAM. Fallback for names the Gemini detection call
    returned no box for — distinct from semantic_grasp.perception.segment_objects,
    the client-side dispatcher that decides which path each name takes."""
    image_pil = Image.fromarray(rgb) #putting into PIL form
    masks = {}
    for obj in objects:
        results = model.predict([image_pil], [obj])
        print(obj)
        raw_image = results[0]['masks'][0]
        uint8_image = (raw_image*255).astype(np.uint8)
        masks[obj] = uint8_image
    return masks


def segment_boxes(rgb, boxes):
    """{name: [x0, y0, x1, y1]} pixel boxes -> {name: uint8 mask}, by prompting
    the SAM inside LangSAM with each box (no GroundingDINO). One set_image, one
    batched box prompt. The caller (semantic_grasp.perception.segment_objects)
    gets the boxes from the Gemini detection call, which localizes every named
    object jointly — the text-query path above can hand two different names
    the same region."""
    names = list(boxes)
    if not names:
        return {}
    h, w = rgb.shape[:2]
    xyxy = np.array([boxes[n] for n in names], dtype=np.float32).reshape(-1, 4)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)
    masks_np, scores, _ = model.sam.predict(np.ascontiguousarray(rgb), xyxy)
    masks_np = np.asarray(masks_np).reshape(len(names), h, w)
    out = {}
    for n, m, s in zip(names, masks_np, np.asarray(scores).reshape(-1)):
        print(f"{n}: box={np.round(boxes[n], 1).tolist()} sam_score={float(s):.3f} "
              f"px={int(np.count_nonzero(m))}")
        out[n] = ((m > 0) * 255).astype(np.uint8)
    return out


# ── 3. ZMQ SETUP ─────────────────────────────────────────────────────
from semantic_grasp.config import PERCEPTION, bind
context = zmq.Context()
socket = context.socket(zmq.REP)        # REPLY socket = server half
bind(socket, PERCEPTION)                # we own the address; clients connect
print(f"Segmentation server listening on {PERCEPTION}  (Ctrl+C to stop)")


# ── 4. REQUEST LOOP ──────────────────────────────────────────────────
#   {"model": "langsam",   "objects": [name, ...],              "image": rgb}
#   {"model": "sam_boxes", "boxes":   {name: [x0, y0, x1, y1]}, "image": rgb}
# Both reply {name: uint8 mask} (0/255, HxW), or {"error": str}.
while True:
    request = pickle.loads(socket.recv())
    print("received")
    rgb = request["image"]
    kind = request.get("model")

    if kind == "langsam":
        objects = request["objects"]
        print(objects)
        try:
            masks = segment_text(rgb, objects)
            payload = pickle.dumps(masks)
        except Exception as e:
            payload = pickle.dumps({"error": str(e)})
            print(f"  !! error: {e}")

    elif kind == "sam_boxes":
        boxes = request["boxes"]
        print(list(boxes))
        try:
            masks = segment_boxes(rgb, boxes)
            payload = pickle.dumps(masks)
        except Exception as e:
            payload = pickle.dumps({"error": str(e)})
            print(f"  !! error: {e}")

    else:
        payload = pickle.dumps({"error": f"unknown model {kind!r} "
                                         f"(expected 'langsam' or 'sam_boxes')"})
        print(f"  !! unknown model {kind!r}")


    # 4c. REPLY ---------------------------------------------------------
    socket.send(payload)
    print("  -> sent masks back\n")