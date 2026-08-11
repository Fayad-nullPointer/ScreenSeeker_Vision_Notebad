"""
ScreenSeekeR Engine Implementation (Algorithm 1 in arXiv:2504.07981)
Implements Box Dilation, Gaussian Centrality Scoring, Non-Maximum Suppression (NMS), 
and Recursive Coarse-to-Fine Visual Search.
"""

import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image

from src.config import SIGMA, D_MAX, S_MIN, DILATION_FACTOR, NMS_THRESHOLD
from src.screen import crop_sub_image, project_crop_to_screen
from src.planner import position_inference
from src.grounder import direct_grounding

logger = logging.getLogger(__name__)

def box_dilation(boxes: List[Tuple[int, int, int, int]], image_size: Tuple[int, int], factor: float = DILATION_FACTOR) -> List[Tuple[int, int, int, int]]:
    """
    Box Dilation (ScreenSeekeR Section 4):
    Expands candidate bounding boxes outward to prevent cutting off target icon edges.
    """
    w_img, h_img = image_size
    dilated_boxes = []
    
    for (x1, y1, x2, y2) in boxes:
        bw = x2 - x1
        bh = y2 - y1
        
        dx = int(bw * factor)
        dy = int(bh * factor)
        
        nx1 = max(0, x1 - dx)
        ny1 = max(0, y1 - dy)
        nx2 = min(w_img, x2 + dx)
        ny2 = min(h_img, y2 + dy)
        
        dilated_boxes.append((nx1, ny1, nx2, ny2))
        
    return dilated_boxes


def gaussian_centrality_score(box: Tuple[int, int, int, int], candidate_boxes: List[Tuple[int, int, int, int]], sigma: float = SIGMA) -> float:
    """
    Gaussian Centrality Scoring (Equation 1 & 2 in ScreenSeekeR paper):
    s = exp( - ((x' - 0.5)^2 + (y' - 0.5)^2) / (2 * sigma^2) )
    
    Assigns higher score to candidates whose center aligns with predicted voting centers.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    total_score = 0.0
    for cand in candidate_boxes:
        cx_c = (cand[0] + cand[2]) / 2.0
        cy_c = (cand[1] + cand[3]) / 2.0
        w_c = max(1, cand[2] - cand[0])
        h_c = max(1, cand[3] - cand[1])
        
        # Normalized relative position inside candidate
        x_prime = (cx - cand[0]) / w_c
        y_prime = (cy - cand[1]) / h_c
        
        if 0.0 <= x_prime <= 1.0 and 0.0 <= y_prime <= 1.0:
            dist_sq = (x_prime - 0.5)**2 + (y_prime - 0.5)**2
            score = math.exp(-dist_sq / (2 * (sigma**2)))
            total_score += score
            
    return total_score if total_score > 0 else 0.5


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[0])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def non_maximum_suppression(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_threshold: float = NMS_THRESHOLD) -> List[Tuple[int, int, int, int]]:
    """
    Non-Maximum Suppression (NMS):
    Filters out highly overlapping candidate boxes, retaining the highest scoring ones.
    """
    if not boxes:
        return []
        
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while idxs:
        current = idxs.pop(0)
        keep.append(boxes[current])
        
        idxs = [i for i in idxs if calculate_iou(boxes[current], boxes[i]) < iou_threshold]
        
    return keep


def visual_search(
    instruction: str, 
    image: Image.Image, 
    viewport_offset: Tuple[int, int] = (0, 0), 
    depth: int = 0
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int], List[Dict[str, Any]]]:
    """
    Algorithm 1: ScreenSeekeR Visual Search (arXiv:2504.07981)
    Recursively narrows search area and grounds target UI element across candidate patches.
    """
    width, height = image.size
    trace = []
    
    # Base case: if max depth reached or patch size is small enough
    if depth >= D_MAX or (width * height) <= (S_MIN * S_MIN):
        local_box, confidence = direct_grounding(instruction, image)
        
        if local_box is not None and confidence > 0.0:
            lx1, ly1, lx2, ly2 = local_box
            lcx = (lx1 + lx2) // 2
            lcy = (ly1 + ly2) // 2
            
            gx, gy = project_crop_to_screen((lcx, lcy), (viewport_offset[0], viewport_offset[1], viewport_offset[0] + width, viewport_offset[1] + height))
            gbx = (
                viewport_offset[0] + lx1,
                viewport_offset[1] + ly1,
                viewport_offset[0] + lx2,
                viewport_offset[1] + ly2
            )
            
            trace.append({
                "depth": depth,
                "action": "Direct Grounding Success",
                "patch_size": (width, height),
                "local_box": local_box,
                "global_center": (gx, gy)
            })
            return (gx, gy), gbx, trace
        else:
            # Target not found in this sub-patch
            return None, None, trace

    # Step 1: Position Inference (Planner)
    candidates = position_inference(instruction, image)
    trace.append({"depth": depth, "action": "Position Inference", "candidates_count": len(candidates)})

    if not candidates:
        candidates = [(0, 0, width, height)]

    # Step 2: Box Dilation
    dilated_candidates = box_dilation(candidates, (width, height))

    # Step 3: Candidate Scoring & NMS
    scores = [gaussian_centrality_score(box, candidates) for box in dilated_candidates]
    nms_candidates = non_maximum_suppression(dilated_candidates, scores)

    # Step 4: Iterative candidate patch evaluation loop (Algorithm 1 lines 14-20)
    for candidate_box in nms_candidates:
        cropped_patch = crop_sub_image(image, candidate_box)
        new_viewport = (viewport_offset[0] + candidate_box[0], viewport_offset[1] + candidate_box[1])
        
        trace.append({
            "depth": depth,
            "action": "Evaluate Candidate Patch",
            "crop_box": candidate_box,
            "new_viewport": new_viewport
        })

        res_center, res_box, sub_trace = visual_search(instruction, cropped_patch, viewport_offset=new_viewport, depth=depth + 1)
        trace.extend(sub_trace)
        
        if res_center is not None:
            return res_center, res_box, trace

    # If direct patch grounding failed for sub-patches (depth > 0), return None to signal parent search loop
    if depth > 0:
        return None, None, trace

    # Root level fallback (depth == 0): If no patch yielded direct grounding, return top candidate box center
    if nms_candidates:
        top_box = nms_candidates[0]
        gx = viewport_offset[0] + (top_box[0] + top_box[2]) // 2
        gy = viewport_offset[1] + (top_box[1] + top_box[3]) // 2
        gbx = (
            viewport_offset[0] + top_box[0],
            viewport_offset[1] + top_box[1],
            viewport_offset[0] + top_box[2],
            viewport_offset[1] + top_box[3]
        )
        logger.warning(f"Direct patch grounding inconclusive. Utilizing Planner candidate region center: ({gx}, {gy})")
        return (gx, gy), gbx, trace

    logger.warning("Target element not detected on screen.")
    return None, None, trace

