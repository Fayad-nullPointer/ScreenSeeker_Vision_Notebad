"""
Desktop Icon Simulator & Test Harness Module
Generates synthetic 1920x1080 desktop screens with Notepad icons placed in 
Top-Left, Bottom-Right, and Center areas to generate mandatory annotated screenshot deliverables.
"""

from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from src.config import SCREEN_RESOLUTION

def create_synthetic_desktop_image(icon_position: Tuple[int, int], position_label: str = "Notepad") -> Image.Image:
    """
    Renders a realistic 1920x1080 desktop canvas with desktop icons and places
    the target Notepad icon at specified (x, y) coordinates.
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
    
    # 2. Draw surrounding system icons (This PC, Recycle Bin, File Explorer)
    system_icons = [
        (80, 80, "This PC"),
        (80, 200, "Recycle Bin"),
        (80, 320, "File Explorer")
    ]
    
    for (ix, iy, label) in system_icons:
        _draw_desktop_icon(draw, ix, iy, label, is_target=False)
        
    # 3. Draw target Notepad icon at target coordinates
    tx, ty = icon_position
    _draw_desktop_icon(draw, tx, ty, "Notepad", is_target=True)
    
    return desktop


def _draw_desktop_icon(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, is_target: bool = False):
    """Draws a realistic shortcut desktop icon."""
    size = 48
    x1, y1 = x - size//2, y - size//2
    x2, y2 = x + size//2, y + size//2
    
    # Draw icon background notebook box
    bg_color = (40, 120, 220) if is_target else (100, 110, 125)
    draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=(255, 255, 255), width=2)
    
    # Draw notebook lines inside icon
    draw.line([x1 + 10, y1 + 12, x2 - 10, y1 + 12], fill=(255, 255, 255), width=2)
    draw.line([x1 + 10, y1 + 22, x2 - 10, y1 + 22], fill=(255, 255, 255), width=2)
    draw.line([x1 + 10, y1 + 32, x2 - 18, y1 + 32], fill=(255, 255, 255), width=2)
    
    # Draw shortcut arrow badge
    draw.rectangle([x1, y2 - 12, x1 + 12, y2], fill=(255, 255, 255))
    draw.polygon([(x1 + 2, y2 - 2), (x1 + 10, y2 - 6), (x1 + 6, y2 - 10)], fill=(0, 0, 0))
    
    # Draw label text below icon
    draw.rectangle([x1 - 10, y2 + 5, x2 + 10, y2 + 25], fill=(0, 0, 0, 180))
    draw.text((x1 - 5, y2 + 8), label, fill=(255, 255, 255))
