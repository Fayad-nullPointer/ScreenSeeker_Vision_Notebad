"""
Visual Debug Annotator Module
Draws ScreenSeekeR search trace boxes, bounding boxes, target center click dot,
and generates the 3 mandatory annotated screenshot deliverables.
"""

from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from src.config import ANNOTATED_OUTPUT_DIR

def draw_grounding_annotations(
    screenshot: Image.Image,
    target_center: Tuple[int, int],
    target_box: Tuple[int, int, int, int],
    label: str = "Notepad Icon",
    trace_info: Optional[List[Dict[str, Any]]] = None
) -> Image.Image:
    """
    Draws visual grounding bounding box, center click target dot, and search trace info on image.
    """
    annotated = screenshot.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    gx, gy = target_center
    x1, y1, x2, y2 = target_box
    
    # Ensure bounding box has valid width and height for visualization
    if x2 <= x1 + 10:
        x2 = max(x1 + 60, gx + 30)
        x1 = max(0, gx - 30)
    if y2 <= y1 + 10:
        y2 = max(y1 + 60, gy + 30)
        y1 = max(0, gy - 30)

    # 1. Draw target bounding box (Bright Cyan/Green outline)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 200, 255), width=4)
    
    # 2. Draw Target Center Point Dot (Red crosshair & filled circle)
    radius = 8
    draw.ellipse([gx - radius, gy - radius, gx + radius, gy + radius], fill=(255, 0, 50, 255), outline=(255, 255, 255, 255), width=2)
    draw.line([gx - 15, gy, gx + 15, gy], fill=(255, 255, 255, 255), width=2)
    draw.line([gx, gy - 15, gx, gy + 15], fill=(255, 255, 255, 255), width=2)
    
    # 3. Draw Text Badge above bounding box
    badge_text = f"GROUNDED: {label} ({gx}, {gy})"
    badge_y1 = max(5, y1 - 28)
    draw.rectangle([x1, badge_y1, x1 + 340, badge_y1 + 25], fill=(0, 150, 120, 230))
    draw.text((x1 + 10, badge_y1 + 4), badge_text, fill=(255, 255, 255, 255))

    composite = Image.alpha_composite(annotated, overlay)
    return composite.convert("RGB")


def save_annotated_deliverable(
    screenshot: Image.Image,
    target_center: Tuple[int, int],
    target_box: Tuple[int, int, int, int],
    position_name: str
) -> Path:
    """
    Saves mandatory deliverable annotated screenshot (top_left, bottom_right, center).
    """
    annotated_img = draw_grounding_annotations(
        screenshot,
        target_center,
        target_box,
        label=position_name
    )
    
    output_path = ANNOTATED_OUTPUT_DIR / f"annotated_{position_name.lower().replace('-', '_')}.png"
    annotated_img.save(output_path)
    return output_path
