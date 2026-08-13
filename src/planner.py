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

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PLANNER_MODEL, GOOGLE_API_KEY, GEMINI_MODEL, SCREEN_RESOLUTION

logger = logging.getLogger(__name__)


def encode_image_to_base64(image: Image.Image) -> str:
    """Converts PIL Image to base64 JPEG string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _get_visual_description(instruction: str) -> str:
    """Provides visual feature hints for Ubuntu Linux desktop icons to maximize MLLM vision accuracy."""
    inst_lower = instruction.lower()
    if any(k in inst_lower for k in ["text", "editor", "notepad", "gedit"]):
        return f'"{instruction}" (Ubuntu Linux Text Editor / Notepad: Look for a desktop wallpaper shortcut icon or dock icon showing a white document sheet with a pencil/notepad symbol, labeled "Text Editor" or "Notepad")'
    elif any(k in inst_lower for k in ["word", "libreoffice", "writer", "doc"]):
        return f'"{instruction}" (Ubuntu LibreOffice Writer: Look for a blue/white document icon labeled "LibreOffice Writer" or "Writer")'
    elif any(k in inst_lower for k in ["excel", "calc", "spreadsheet"]):
        return f'"{instruction}" (Ubuntu LibreOffice Calc: Look for a green spreadsheet icon labeled "LibreOffice Calc" or "Calc")'
    elif any(k in inst_lower for k in ["chrome", "browser", "web", "firefox"]):
        return f'"{instruction}" (Web Browser: Look for a Firefox orange fox icon or Google Chrome circular logo icon on desktop/dock)'
    elif any(k in inst_lower for k in ["terminal", "cmd", "bash"]):
        return f'"{instruction}" (Ubuntu Terminal: Look for a dark square icon with prompt symbol ">_" labeled "Terminal")'
    elif any(k in inst_lower for k in ["folder", "explorer", "file", "files"]):
        return f'"{instruction}" (Ubuntu Files / Nautilus: Look for a purple or yellow folder icon labeled "Files" or "Home")'
    return f'"{instruction}"'


def _call_google_gemini_api(prompt: str, base64_image: str, width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Directly invokes Google Gemini API via official endpoint using GOOGLE_API_KEY."""
    gemini_models = [GEMINI_MODEL, "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    # De-duplicate while preserving order
    gemini_models = list(dict.fromkeys(gemini_models))

    for model_name in gemini_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    res_json = response.json()
                    candidates_data = res_json.get("candidates", [])
                    if candidates_data:
                        parts = candidates_data[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            content = parts[0]["text"]
                            candidates = _parse_planner_response(content, width, height)
                            if candidates:
                                logger.info(f"Successfully grounded '{prompt[:30]}...' using Direct Google Gemini model '{model_name}'.")
                                return candidates
        except Exception as e:
            logger.warning(f"Direct Google Gemini API call failed for model '{model_name}': {e}")
    return []


def position_inference(instruction: str, image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Position Inference (ScreenSeekeR Paper Section 4):
    Asks the MLLM planner to analyze the screenshot and return candidate bounding box regions
    [x_min, y_min, x_max, y_max] in absolute 1920x1080 pixel coordinates.
    """
    width, height = image.size
    base64_image = encode_image_to_base64(image)
    visual_desc = _get_visual_description(instruction)
    
    prompt = f"""You are a GUI Desktop Grounding Planner operating on an Ubuntu Linux Desktop environment.
Instruction: Locate the target desktop shortcut icon or application launcher icon {visual_desc} on the Ubuntu desktop screen.

UBUNTU LINUX DESKTOP MAPPINGS & RULES:
1. Environment: Ubuntu Linux (Gnome desktop).
2. "Notepad" in Ubuntu is called "Text Editor" (gnome-text-editor/gedit) with a document/pencil shortcut icon.
3. "Word" in Ubuntu is called "LibreOffice Writer" (blue document icon).
4. Target MUST be a desktop shortcut icon (on the wallpaper surface) or taskbar/dock application launcher button.
5. DO NOT select open application window bodies, text areas, or title bars of already open windows.
6. Identify candidate regions [xmin, ymin, xmax, ymax] enclosing ONLY the target desktop shortcut icon or dock button.

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

    # 1. Primary: If GOOGLE_API_KEY is available, use direct Google Gemini SDK / REST API
    if GOOGLE_API_KEY:
        gemini_candidates = _call_google_gemini_api(prompt, base64_image, width, height)
        if gemini_candidates:
            return gemini_candidates

    # 2. Secondary: If OpenRouter API key is set, use OpenRouter API
    if OPENROUTER_API_KEY:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/automatic-cursor-notebad",
            "Content-Type": "application/json"
        }

        candidate_models = [PLANNER_MODEL, "google/gemini-3-flash-preview", "google/gemini-2.5-flash", "qwen/qwen-2.5-vl-72b-instruct", "openai/gpt-4o"]
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
                        logger.info(f"Successfully grounded '{instruction}' using OpenRouter model '{model}'.")
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
