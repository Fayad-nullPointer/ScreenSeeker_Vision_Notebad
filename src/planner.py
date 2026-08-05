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

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, SCREEN_RESOLUTION

logger = logging.getLogger(__name__)

def encode_image_to_base64(image: Image.Image) -> str:
    """Converts PIL Image to base64 JPEG string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
        
    try:
        base64_image = encode_image_to_base64(image)
        
        prompt = f"""You are a GUI Desktop Grounding Planner.
Instruction: Find the target UI element "{instruction}" on the 1920x1080 desktop.

Analyze the screenshot and output 1 to 3 candidate bounding box areas where the target is most likely located.
Return ONLY valid JSON in the following format:
```json
{{
  "candidate_areas": [
    {{"box_1000": [ymin, xmin, ymax, xmax], "description": "reasoning"}},
    ...
  ]
}}
```
where [ymin, xmin, ymax, xmax] are normalized coordinates scaled from 0 to 1000.
"""

        payload = {
            "model": OPENROUTER_MODEL,
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
            "temperature": 0.1,
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/automatic-cursor-notebad",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=15.0) as client:
            response = client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            res_json = response.json()
            
            content = res_json['choices'][0]['message']['content']
            candidates = _parse_planner_response(content, width, height)
            if candidates:
                return candidates

    except Exception as e:
        logger.error(f"Position Inference API call failed: {e}. Falling back to heuristic grid search.")
        
    return _heuristic_position_inference(image)


def _parse_planner_response(response_text: str, width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Parses JSON candidate boxes from LLM output string and converts to pixel bounds."""
    try:
        # Extract json code block
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            boxes = []
            for item in data.get("candidate_areas", []):
                b = item.get("box_1000", [])
                if len(b) == 4:
                    ymin, xmin, ymax, xmax = b
                    pixel_box = (
                        int(xmin * width / 1000.0),
                        int(ymin * height / 1000.0),
                        int(xmax * width / 1000.0),
                        int(ymax * height / 1000.0)
                    )
                    boxes.append(pixel_box)
            return boxes
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}")
    return []


def _heuristic_position_inference(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Fallback Heuristic Position Inference:
    Generates standard desktop quadrant search boxes (Top-Left, Desktop Left Grid, Center, Bottom-Right).
    """
    w, h = image.size
    return [
        (0, 0, int(w * 0.4), int(h * 0.5)),            # Top-Left quadrant
        (0, 0, int(w * 0.25), int(h * 0.9)),           # Desktop Icon Left Grid
        (int(w * 0.3), int(h * 0.2), int(w * 0.7), int(h * 0.8)), # Center area
        (int(w * 0.6), int(h * 0.5), w, h)             # Bottom-Right quadrant
    ]
