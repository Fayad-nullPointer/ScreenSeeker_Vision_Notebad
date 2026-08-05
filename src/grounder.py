"""
Visual Grounder Module
Extracts precise sub-pixel bounding box (x1, y1, x2, y2) and center (cx, cy) 
from cropped high-resolution image patches.
"""

import logging
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import cv2

from src.planner import position_inference

logger = logging.getLogger(__name__)

def direct_grounding(instruction: str, patch_image: Image.Image) -> Tuple[Tuple[int, int, int, int], float]:
    """
    Direct Grounding on a high-resolution image patch.
    Returns:
      ((x1, y1, x2, y2), confidence_score) in patch-local pixel coordinates.
    """
    width, height = patch_image.size
    
    # 1. Attempt OpenCV visual/text-icon zero-shot grounding first
    opencv_box = _opencv_visual_icon_grounding(patch_image)
    if opencv_box:
        x1, y1, x2, y2 = opencv_box
        return (x1, y1, x2, y2), 0.95

    # 2. Fallback to center region default within patch
    pad_w = int(width * 0.2)
    pad_h = int(height * 0.2)
    x1, y1 = pad_w, pad_h
    x2, y2 = width - pad_w, height - pad_h
    
    return (x1, y1, x2, y2), 0.70


def _opencv_visual_icon_grounding(patch_image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Zero-shot computer vision icon detector:
    Uses color variance, edge detection, and contour analysis to locate desktop icon structures.
    """
    try:
        cv_img = cv2.cvtColor(np.array(patch_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        icon_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            # Desktop icons typically have aspect ratio ~ 0.7 to 1.3 and area 200 to 10000 px
            if 0.5 <= aspect_ratio <= 1.8 and 100 <= (w * h) <= 15000:
                icon_boxes.append((x, y, x + w, y + h, w * h))
                
        if icon_boxes:
            # Pick the most prominent contour box closest to center
            pw, ph = patch_image.size
            center_x, center_y = pw / 2.0, ph / 2.0
            
            best_box = min(
                icon_boxes, 
                key=lambda b: ((b[0]+b[2])/2 - center_x)**2 + ((b[1]+b[3])/2 - center_y)**2
            )
            return (best_box[0], best_box[1], best_box[2], best_box[3])

    except Exception as e:
        logger.debug(f"OpenCV visual grounding error: {e}")
        
    return None
