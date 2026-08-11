"""
ScreenSeekeR Position Inference (Planner Module)
Leverages Multimodal LLM via OpenRouter (or fallback) to analyze desktop context and infer target candidate areas.
"""

import io
import base64
import json
import logging
from typing import List, Tuple, Dict, Any
from PIL import Image
import httpx

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PLANNER_MODEL, SCREEN_RESOLUTION

logger = logging.getLogger(__name__)

def encode_image_to_base64(image: Image.Image) -> str:
    """Converts PIL Image to base64 JPEG string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def _get_visual_description(instruction: str) -> str:
    """Provides visual feature hints for common desktop icons to maximize MLLM vision accuracy."""
    inst_lower = instruction.lower()
    if any(k in inst_lower for k in ["text", "editor", "notepad", "gedit"]):
        return f'"{instruction}" (Look for a DESKTOP SHORTCUT ICON on the desktop wallpaper background showing a document/notepad symbol or text label "Text Editor" / "Notepad")'
    elif any(k in inst_lower for k in ["chrome", "browser", "web"]):
        return f'"{instruction}" (Look for a circular red, yellow, green, blue browser icon on the desktop or dock)'
    elif any(k in inst_lower for k in ["terminal", "cmd", "bash"]):
        return f'"{instruction}" (Look for a dark square terminal icon on the desktop or dock)'
    elif any(k in inst_lower for k in ["folder", "explorer", "file"]):
        return f'"{instruction}" (Look for a yellow folder shortcut icon on the desktop wallpaper)'
    return f'"{instruction}"'


def position_inference(instruction: str, image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Position Inference (ScreenSeekeR Paper Section 4):
    Asks the MLLM planner to analyze the screenshot and return candidate bounding box regions
    [x_min, y_min, x_max, y_max] in absolute 1920x1080 pixel coordinates.
    """
    width, height = image.size
    
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. Using heuristic desktop grid planner.")
        return _heuristic_position_inference(image)
        
    base64_image = encode_image_to_base64(image)
    
    visual_desc = _get_visual_description(instruction)
    
    prompt = f"""You are a GUI Desktop Grounding Planner.
Instruction: Locate the exact target desktop shortcut icon or application launcher icon {visual_desc} on the desktop.

CRITICAL RULES:
1. Target MUST be a desktop shortcut icon (located on the wallpaper surface) or a taskbar/dock launcher icon.
2. DO NOT select open window title bars, window bodies, text areas, or headers of already open applications (such as an open Text Editor or browser window).
3. Identify candidate regions [xmin, ymin, xmax, ymax] enclosing ONLY the desktop shortcut icon or launcher button.

Return 1 to 3 candidate bounding box areas [xmin, ymin, xmax, ymax] (scaled 0 to 1000) for "{instruction}".
Return ONLY valid JSON:
```json
{{
  "candidate_areas": [
    {{"box_1000": [xmin, ymin, xmax, ymax], "description": "reasoning"}}
  ]
}}
```
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/automatic-cursor-notebad",
        "Content-Type": "application/json"
    }

    # Model fallback chain: Gemini 3 Flash Preview -> Gemini 2.5 Flash -> Qwen 2.5 VL -> GPT-4o
    candidate_models = [PLANNER_MODEL, "google/gemini-3-flash-preview", "google/gemini-2.5-flash", "qwen/qwen-2.5-vl-72b-instruct", "openai/gpt-4o"]
    # De-duplicate while preserving order
    candidate_models = list(dict.fromkeys(candidate_models))

    for model in candidate_models:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                res_json = response.json()
                
                content = res_json['choices'][0]['message']['content']
                candidates = _parse_planner_response(content, width, height)
                if candidates:
                    logger.info(f"Successfully grounded '{instruction}' using model '{model}'.")
                    return candidates
        except Exception as e:
            logger.warning(f"Position Inference call failed for model '{model}': {e}. Trying fallback model...")

    logger.error(f"All MLLM models failed for position inference. Falling back to heuristic grid search.")
        
    return _heuristic_position_inference(image)


def _parse_planner_response(response_text: str, width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Parses JSON candidate boxes from LLM output string and converts to pixel bounds."""
    try:
        text = response_text.strip()
        # Clean markdown code blocks if present
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p_str = p.strip()
                if p_str.startswith("json"):
                    p_str = p_str[4:].strip()
                if (p_str.startswith("{") and p_str.endswith("}")) or (p_str.startswith("[") and p_str.endswith("]")):
                    text = p_str
                    break

        # Replace single quotes with double quotes for valid JSON parsing
        cleaned_json = text.replace("'", '"')
        
        start_bracket = min([i for i in [cleaned_json.find("{"), cleaned_json.find("[")] if i != -1], default=-1)
        end_bracket = max([cleaned_json.rfind("}"), cleaned_json.rfind("]")])
        
        raw_boxes = []
        if start_bracket != -1 and end_bracket != -1:
            json_str = cleaned_json[start_bracket:end_bracket + 1]
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list) and len(item) == 4:
                            raw_boxes.append(item)
                        elif isinstance(item, dict):
                            b = item.get("box_1000") or item.get("box") or item.get("bbox")
                            if isinstance(b, list) and len(b) == 4:
                                raw_boxes.append(b)
                elif isinstance(data, dict):
                    candidates = data.get("candidate_areas") or data.get("candidates") or [data]
                    if isinstance(candidates, list):
                        for item in candidates:
                            if isinstance(item, list) and len(item) == 4:
                                raw_boxes.append(item)
                            elif isinstance(item, dict):
                                b = item.get("box_1000") or item.get("box") or item.get("bbox")
                                if isinstance(b, list) and len(b) == 4:
                                    raw_boxes.append(b)
            except Exception:
                pass

        # Regex fallback matching [num, num, num, num]
        if not raw_boxes:
            import re
            matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
            for m in matches:
                raw_boxes.append([int(v) for v in m])

        boxes = []
        for b in raw_boxes:
            c1, c2, c3, c4 = b
            # If c1 < c2, c1 is xmin, c2 is ymin. If c1 > c2 and c1 > 300, c1 is ymin.
            # Convert to absolute pixel bounds: (xmin, ymin, xmax, ymax)
            if c1 > c2 and c1 > 300: # ymin, xmin, ymax, xmax format
                ymin, xmin, ymax, xmax = c1, c2, c3, c4
            else: # xmin, ymin, xmax, ymax format
                xmin, ymin, xmax, ymax = c1, c2, c3, c4

            pixel_box = (
                max(0, min(int(xmin * width / 1000.0), width - 1)),
                max(0, min(int(ymin * height / 1000.0), height - 1)),
                max(1, min(int(xmax * width / 1000.0), width)),
                max(1, min(int(ymax * height / 1000.0), height))
            )
            boxes.append(pixel_box)

        if boxes:
            return boxes
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}")
    return []


def _heuristic_position_inference(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Fallback Heuristic Position Inference:
    Generates desktop candidate search areas prioritizing desktop wallpaper shortcuts, left dock, and center.
    """
    w, h = image.size
    return [
        (int(w * 0.15), int(h * 0.1), int(w * 0.6), int(h * 0.8)), # Primary Desktop Surface Grid (where shortcuts live)
        (0, 0, int(w * 0.1), h),                                   # Left Launcher Dock / Taskbar column
        (0, 0, int(w * 0.35), int(h * 0.5)),                       # Top-Left desktop grid
        (int(w * 0.3), int(h * 0.2), int(w * 0.7), int(h * 0.8)), # Center desktop area
    ]
