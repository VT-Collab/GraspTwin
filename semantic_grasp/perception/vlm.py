"""VLM/LLM API calls and response parsing."""
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import logging
import re
from pathlib import Path

import yaml

from ..config import (
    GEMINI_API_KEY, GEMINI_BASE, VLM_MODEL,
    TROUBLESHOOTING_FILE,
)

log = logging.getLogger("pipeline")

VLM_SYSTEM_PROMPT = (
    "You are a precise spatial-reasoning assistant for robotic grasping. "
    "You will be shown a rendered view of a 3D object. "
    "Point to the requested feature by returning its coordinates as JSON. "
    "The point MUST lie directly on the surface of the object, not in empty space. "
    "Never explain. Never use markdown. Only output valid JSON."
)


def _log_to_troubleshooting(label, model, provider, system=None, user_content=None,
                            prompt=None, n_images=None, response=None):
    """Append prompt/response details to the troubleshooting file."""
    from datetime import datetime
    with open(TROUBLESHOOTING_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if response is not None:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{ts}] {label} -- {model} ({provider}) -- RESPONSE\n")
            f.write(f"{'='*80}\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")
        else:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{ts}] {label} -- {model} ({provider}) -- PROMPT\n")
            f.write(f"{'='*80}\n")
            if system:
                f.write(f"SYSTEM PROMPT:\n{system}\n")
                f.write(f"{'-'*80}\n")
            if user_content:
                if isinstance(user_content, str):
                    f.write(f"USER PROMPT:\n{user_content}\n")
                elif isinstance(user_content, list):
                    for item in user_content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            f.write(f"USER PROMPT:\n{item['text']}\n")
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            f.write("[IMAGE attached]\n")
            if prompt:
                f.write(f"PROMPT:\n{prompt}\n")
            if n_images is not None:
                f.write(f"[{n_images} image(s) attached]\n")
            f.write(f"{'='*80}\n\n")


def call_gemini(url, payload, label, model, timeout=60):
    for attempt in range(4):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)

            if resp.status_code in (429, 503) and attempt < 3:
                wait = 2 ** attempt
                log.warning(f"Gemini {resp.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            parts = resp.json()["candidates"][0]["content"]["parts"]

            # prefer a non-"thought" text part, else fall back to the last text part
            content = next(
                (p["text"] for p in parts if "text" in p and "thought" not in p),
                ""
            )
            if not content:
                content = next(
                    (p["text"] for p in reversed(parts) if "text" in p),
                    ""
                )

            _log_to_troubleshooting(label, model, "gemini", response=content)
            return content

        except Exception as e:
            if attempt < 3:
                wait = 2 ** attempt
                log.warning(f"Gemini error: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise

        except Exception as e:
            if attempt < 3:
                wait = 2 ** attempt
                log.warning(f"Gemini error: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise

def build_user_prompt(template, cfg):
    return template.format(
        object_description=cfg["task"]["object_description"],
        scale=cfg["coordinates"]["scale"],
    )


def parse_predictions(raw_content, view_index=0):
    """Parse VLM response. Handles both array and dict formats."""
    cleaned = re.sub(r"```(?:json)?", "", raw_content).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(f"Could not parse VLM JSON: {raw_content[:300]}")
        return []

    # Robotics ER returns a list: [{"point": [y,x], "label": "..."}]
    # Old Gemini returned: {"predictions": [...]}
    if isinstance(data, list):
        preds = data
    elif isinstance(data, dict):
        preds = data.get("predictions", [])
    else:
        return []

    # Ensure each prediction has view_index and confidence
    for p in preds:
        p.setdefault("view_index", view_index)
        p.setdefault("confidence", 1.0)
    return preds


def query_vlm_view(view, prompt_template, cfg, url):
    """Query Gemini for a single rendered view. Returns list of predictions."""
    idx = view["index"]
    label = view["label"]
    image_b64 = base64.b64encode(view["image_png"]).decode("utf-8")

    user_prompt = build_user_prompt(prompt_template, cfg)
    full_prompt = VLM_SYSTEM_PROMPT + "\n\n" + user_prompt
    payload = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                    {"text": full_prompt},
                ],
            },
        ],
        "generationConfig": {
            "temperature": 0.5,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    content = call_gemini(url, payload, f"VLM view {idx} {label}", VLM_MODEL)
    return parse_predictions(content, view_index=idx)


def query_grasp_semantics(task, target):
    """Ask Gemini Robotics ER for the specific part/region of the target to grasp.

    Returns a short noun phrase (e.g. "outer edge of the bowl rim") that is fed
    into the object_description of the per-view VLM prompt.
    """
    config_path = Path(__file__).parents[2] / "config" / "grasp_target_part_prompt.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    system = cfg["system_prompt"].strip()
    user_text = cfg["user_prompt_template"].format(task=task, target=target)
    full_prompt = system + "\n\n" + user_text

    url = f"{GEMINI_BASE}/models/{VLM_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": full_prompt}]},
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": cfg.get("max_tokens", 100),
        },
    }

    content = call_gemini(url, payload, "GraspSemantics", VLM_MODEL)
    grasp_target = content.strip().strip('"').strip("'").strip()
    log.info(f"Semantic grasp target: '{grasp_target}'")
    return grasp_target


def query_target_object(task, objects):
    """Ask Gemini Robotics ER which detected object the robot should interact
    with FIRST for `task`.

    `objects` is the name list from get_pertinent_objects. Returns the chosen
    name as the model wrote it (the caller matches it back against the list),
    or None when the model answers NONE.
    """
    config_path = Path(__file__).parents[2] / "config" / "target_object_selection_prompt.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    system = cfg["system_prompt"].strip()
    user_text = cfg["user_prompt_template"].format(
        task=task, objects=", ".join(objects))
    full_prompt = system + "\n\n" + user_text

    url = f"{GEMINI_BASE}/models/{VLM_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": full_prompt}]},
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": cfg.get("max_tokens", 200),
        },
    }

    content = call_gemini(url, payload, "TargetObject", VLM_MODEL)
    choice = content.strip().strip('"').strip("'").strip()
    log.info(f"Target object for task {task!r}: '{choice}'")
    return None if choice.upper() == "NONE" else choice


def query_approach_semantics(task, target, grasp_part=None):
    """Ask Gemini Robotics ER how the gripper should approach the grasp, in words.

    Returns a short phrase (e.g. "from directly above, jaws closing across the
    rim") that conditions the numeric arrow-vote stage (VLM_Predict_RPY), the
    way query_grasp_semantics' part phrase conditions the pointing stage.
    """
    config_path = Path(__file__).parents[2] / "config" / "approach_semantic_prompt.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    system = cfg["system_prompt"].strip()
    user_text = cfg["user_prompt_template"].format(
        task=task, target=target, grasp_part=grasp_part or target)
    full_prompt = system + "\n\n" + user_text

    url = f"{GEMINI_BASE}/models/{VLM_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": full_prompt}]},
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": cfg.get("max_tokens", 100),
        },
    }

    content = call_gemini(url, payload, "ApproachSemantics", VLM_MODEL)
    approach = content.strip().strip('"').strip("'").strip()
    log.info(f"Semantic approach: '{approach}'")
    return approach


def query_vlm_views(metadata, task, target, grasp_part=None):
    """Query Gemini for every rendered view.

    Returns (predictions, grasp_target): the combined per-view predictions and
    the grasp-semantics phrase (the specific part/region of the target to grasp)
    that was used as the object description. `grasp_part` supplies that phrase
    when the semantic stage already ran (VLM_Predict_XYZ_Semantic); when omitted
    it is derived here via query_grasp_semantics.
    """
    config_path = Path(__file__).parents[2] / "config" / "grasp_point_localization_prompt.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # The specific part/region to point at: caller-supplied, else ask Gemini
    # Robotics ER for it here; either way it becomes the object description in
    # the per-view pointing prompt.
    grasp_target = grasp_part if grasp_part is not None else query_grasp_semantics(task, target)

    # Point each view at the specific part chosen by the grasp-target-part stage.
    cfg["task"]["object_description"] = grasp_target
    prompt_template = cfg["prompt_template"]

    url = f"{GEMINI_BASE}/models/{VLM_MODEL}:generateContent?key={GEMINI_API_KEY}"

    views = metadata["views"]

    # The text prompt is identical for every view (only the image changes),
    # so print it once.
    sample_prompt = VLM_SYSTEM_PROMPT + "\n\n" + build_user_prompt(prompt_template, cfg)
    print(f"\n{'='*80}\n[VLM views] PROMPT (same for all {len(views)} views):"
          f"\n{'-'*80}\n{sample_prompt}\n{'='*80}\n")

    with ThreadPoolExecutor(max_workers=len(views)) as pool:
        results = pool.map(
            lambda view: query_vlm_view(view, prompt_template, cfg, url), views
        )

    predictions = []
    for preds in results:
        predictions.extend(preds)
    return predictions, grasp_target
