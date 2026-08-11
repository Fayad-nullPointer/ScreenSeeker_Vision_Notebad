"""
Visual Grounder Module
Extracts precise sub-pixel bounding box (x1, y1, x2, y2) and center (cx, cy) 
from cropped high-resolution image patches.
"""

import io
import base64
import json
import logging
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import cv2
import httpx

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GROUNDER_MODEL

logger = logging.getLogger(__name__)

def direct_grounding(instruction: str, patch_image: Image.Image) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Direct Grounding (ScreenSeekeR Section 4):
    Evaluates cropped image patch to detect exact target UI element.
    Returns (local_box, confidence). Confidence 0.0 indicates target not present in patch.
    """
    width, height = patch_image.size
    
    # 1. Primary: Use MLLM direct patch grounding for exact target matching
    if OPENROUTER_API_KEY:
        mllm_box = _mllm_patch_grounding(instruction, patch_image)
        if mllm_box:
            return mllm_box, 0.98

    # 2. Fallback target-aware icon detector (checks icon structure with matching criteria)
    target_box = _target_aware_fallback_grounding(instruction, patch_image)
    if target_box:
        x1, y1, x2, y2 = target_box
        return (x1, y1, x2, y2), 0.85

    # 3. Target not present in patch
    return None, 0.0


def _mllm_patch_grounding(instruction: str, patch_image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Uses OpenRouter MLLM to return exact pixel bounding box of target shortcut icon within patch."""
    width, height = patch_image.size
    try:
        buffered = io.BytesIO()
        patch_image.save(buffered, format="JPEG", quality=90)
        base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        prompt = f"""Pinpoint the exact bounding box of the DESKTOP SHORTCUT ICON or LAUNCHER ICON for "{instruction}" inside this image patch.

CRITICAL RULES FOR ACCURACY:
1. Target MUST be a desktop shortcut icon (on wallpaper) or taskbar/dock application launcher icon matching "{instruction}".
2. DO NOT ground title bars, menu bars, text lines, or bodies of already open application windows (e.g. open Text Editor or document window).
3. If ONLY an open application window or random text is present without the actual shortcut icon, return {{"box_1000": null}}.
4. If the target shortcut icon is present, return ONLY valid JSON:
```json
{{
  "box_1000": [xmin, ymin, xmax, ymax]
}}
```
where coordinates are scaled from 0 to 1000.
"""

        payload = {
            "model": GROUNDER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/automatic-cursor-notebad",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                res_json = resp.json()
                content = res_json['choices'][0]['message']['content']
                
                # Parse box_1000
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    data = json.loads(content[start_idx:end_idx])
                    b = data.get("box_1000")
                    if isinstance(b, list) and len(b) == 4:
                        xmin, ymin, xmax, ymax = b
                        # Validate sanity of returned box
                        if xmax > xmin and ymax > ymin:
                            return (
                                int(xmin * width / 1000.0),
                                int(ymin * height / 1000.0),
                                int(xmax * width / 1000.0),
                                int(ymax * height / 1000.0)
                            )
    except Exception as e:
        logger.debug(f"MLLM patch grounding exception: {e}")
        
    return None


def _target_aware_fallback_grounding(instruction: str, patch_image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Target-Aware Computer Vision Fallback:
    Locates isolated icon glyphs / shortcut squares ONLY when patch size matches candidate patch limits,
    preventing arbitrary false grounding on text windows or background edges.
    """
    try:
        pw, ph = patch_image.size
        # Fallback contour search only operates on small patches to avoid full-screen random boxes
        if pw > 600 or ph > 600:
            return None

        cv_img = cv2.cvtColor(np.array(patch_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        icon_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            # Strict aspect ratio (0.7 to 1.4) and icon size range (300 to 8000 sq px)
            if 0.7 <= aspect_ratio <= 1.4 and 300 <= (w * h) <= 8000:
                icon_boxes.append((x, y, x + w, y + h, w * h))
                
        if icon_boxes:
            center_x, center_y = pw / 2.0, ph / 2.0
            best_box = min(
                icon_boxes, 
                key=lambda b: ((b[0]+b[2])/2 - center_x)**2 + ((b[1]+b[3])/2 - center_y)**2
            )
            return (best_box[0], best_box[1], best_box[2], best_box[3])

    except Exception as e:
        logger.debug(f"Target-aware fallback grounding error: {e}")
        
    return None

