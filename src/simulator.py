"""
Desktop Icon Simulator & Test Harness Module
Generates synthetic 1920x1080 desktop screens with custom target icons (Notepad, Recycle Bin, File Explorer, Chrome, etc.)
placed in Top-Left, Bottom-Right, and Center areas to test generalized icon selection.
"""

from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from src.config import SCREEN_RESOLUTION

def create_synthetic_desktop_image(icon_position: Tuple[int, int], target_label: str = "Notepad") -> Image.Image:
    """
    Renders a realistic 1920x1080 desktop canvas with multiple desktop icons and places
    the requested target icon at specified (x, y) coordinates.
    """
    w, h = SCREEN_RESOLUTION
    
    # 1. Create desktop wallpaper gradient (Dark modern theme)
    desktop = Image.new("RGB", (w, h), (20, 24, 33))
    draw = ImageDraw.Draw(desktop)
    
    # Draw desktop wallpaper background shapes
    draw.polygon([(0, 0), (w, 0), (0, h)], fill=(28, 34, 48))
    draw.ellipse([w//4, h//4, w*3//4, h*3//4], fill=(25, 30, 42))
    
    # Draw taskbar at bottom
    draw.rectangle([0, h - 48, w, h], fill=(15, 18, 24))
    
    # 2. Draw surrounding desktop icons (This PC, Recycle Bin, File Explorer, etc.)
    all_icons = ["This PC", "Recycle Bin", "File Explorer", "Chrome", "Terminal", "Notepad"]
    
    # Distribute surrounding non-target icons on grid
    grid_coords = [
        (80, 80), (80, 200), (80, 320), (80, 440), (80, 560)
    ]
    
    for idx, (ix, iy) in enumerate(grid_coords):
        label_name = all_icons[idx % len(all_icons)]
        if label_name.lower() != target_label.lower():
            _draw_desktop_icon(draw, ix, iy, label_name, is_target=False)
        
    # 3. Draw target icon at target requested coordinates
    tx, ty = icon_position
    _draw_desktop_icon(draw, tx, ty, target_label, is_target=True)
    
    return desktop


def _draw_desktop_icon(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, is_target: bool = False):
    """Draws a realistic desktop icon matching the target label name."""
    size = 48
    x1, y1 = x - size//2, y - size//2
    x2, y2 = x + size//2, y + size//2
    
    label_lower = label.lower()
    
    # Custom icon colors based on icon type
    if "recycle" in label_lower or "trash" in label_lower:
        bg_color = (40, 180, 100)
    elif "explorer" in label_lower or "file" in label_lower or "folder" in label_lower:
        bg_color = (230, 170, 40)
    elif "chrome" in label_lower or "browser" in label_lower:
        bg_color = (220, 60, 60)
    elif "terminal" in label_lower or "cmd" in label_lower:
        bg_color = (40, 40, 40)
    else:
        bg_color = (40, 120, 220) if is_target else (100, 110, 125)
        
    draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=(255, 255, 255), width=2)
    
    # Draw icon pattern inside box
    draw.line([x1 + 10, y1 + 12, x2 - 10, y1 + 12], fill=(255, 255, 255), width=2)
    draw.line([x1 + 10, y1 + 22, x2 - 10, y1 + 22], fill=(255, 255, 255), width=2)
    draw.line([x1 + 10, y1 + 32, x2 - 18, y1 + 32], fill=(255, 255, 255), width=2)
    
    # Draw shortcut arrow badge
    draw.rectangle([x1, y2 - 12, x1 + 12, y2], fill=(255, 255, 255))
    draw.polygon([(x1 + 2, y2 - 2), (x1 + 10, y2 - 6), (x1 + 6, y2 - 10)], fill=(0, 0, 0))
    
    # Draw label text below icon
    draw.rectangle([x1 - 15, y2 + 5, x2 + 15, y2 + 25], fill=(0, 0, 0, 180))
    draw.text((x1 - 10, y2 + 8), label, fill=(255, 255, 255))
