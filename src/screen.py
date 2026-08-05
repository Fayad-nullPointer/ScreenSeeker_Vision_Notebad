"""
Screenshot capture and coordinate mapping utilities
"""

from typing import Tuple
from PIL import Image
import mss
import mss.tools
from src.config import SCREEN_RESOLUTION

def capture_desktop_screenshot() -> Image.Image:
    """
    Captures full 1920x1080 resolution desktop screenshot using mss.
    Returns PIL Image in RGB mode.
    """
    with mss.mss() as sct:
        # Capture primary monitor
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        sct_img = sct.grab(monitor)
        
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # Resize to target resolution if needed
        if img.size != SCREEN_RESOLUTION:
            img = img.resize(SCREEN_RESOLUTION, Image.Resampling.LANCZOS)
            
        return img

def crop_sub_image(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crops sub-image patch from full image using bounding box (x_min, y_min, x_max, y_max).
    Preserves 100% pixel detail without downsampling.
    """
    x_min, y_min, x_max, y_max = box
    
    # Boundary clamp
    width, height = image.size
    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(x_min + 1, min(x_max, width))
    y_max = max(y_min + 1, min(y_max, height))
    
    return image.crop((x_min, y_min, x_max, y_max))

def project_crop_to_screen(local_point: Tuple[float, float], crop_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Projects local coordinates (local_x, local_y) within a cropped patch back to 
    absolute 1920x1080 screen pixel coordinates.
    
    Formula:
      global_x = x_min + local_x
      global_y = y_min + local_y
    """
    x_min, y_min, _, _ = crop_box
    local_x, local_y = local_point
    
    global_x = int(x_min + local_x)
    global_y = int(y_min + local_y)
    
    return global_x, global_y
