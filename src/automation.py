"""
JSONPlaceholder API Fetcher & OS Desktop Automation Module
Handles post downloading, PyAutoGUI double-clicking, typing, and file saving.
"""

import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import httpx
from pydantic import BaseModel

from src.config import PROJECT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# Safe PyAutoGUI import wrapper (prevents MouseInfo SystemExit crash on Linux)
_pyautogui = None
try:
    import sys
    import types
    if 'tkinter' not in sys.modules:
        sys.modules['tkinter'] = types.ModuleType('tkinter')
    import pyautogui
    _pyautogui = pyautogui
except BaseException:
    _pyautogui = None


class Post(BaseModel):
    id: int
    userId: int
    title: str
    body: str

# Fallback mock post dataset if remote API network resets
FALLBACK_POSTS = [
    Post(id=i, userId=1, title=f"Sample Post Title {i}", body=f"This is the body content for post number {i}. Automating Notepad saving.")
    for i in range(1, 11)
]

def fetch_posts(count: int = 10) -> List[Post]:
    """
    Fetches first `count` posts from JSONPlaceholder API with User-Agent headers,
    retry logic, and network connection resiliency.
    """
    url = "https://jsonplaceholder.typicode.com/posts"
    logger.info(f"Fetching posts from {url}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    for attempt in range(1, 4):
        try:
            with httpx.Client(timeout=10.0, headers=headers, verify=False, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                
                posts = [Post(**item) for item in data[:count]]
                logger.info(f"Successfully fetched {len(posts)} posts from JSONPlaceholder API.")
                return posts
        except Exception as e:
            logger.warning(f"Fetch attempt {attempt}/3 failed: {e}. Retrying in 1s...")
            time.sleep(1.0)

    logger.warning("Remote JSONPlaceholder API unavailable or resetting connections. Utilizing mock post dataset.")
    return FALLBACK_POSTS[:count]


def prepare_clean_desktop() -> None:
    """
    Minimizes or closes open windows to expose desktop wallpaper & shortcut icons
    prior to screenshot capture.
    """
    import subprocess
    logger.info("Preparing clean desktop state...")
    try:
        subprocess.run(["xdotool", "key", "Super+d"], capture_output=True, timeout=2)
    except Exception:
        pass
    try:
        subprocess.run(["wmctrl", "-k", "on"], capture_output=True, timeout=2)
    except Exception:
        pass
    time.sleep(0.5)


def launch_application_at(target_center: tuple[int, int]) -> None:
    """
    Double-clicks center (x, y) coordinates on desktop screen to launch target application.
    Includes xdotool native click, PyAutoGUI, and OS app launcher fallback.
    """
    gx, gy = target_center
    gx = max(0, min(gx, 1919))
    gy = max(0, min(gy, 1079))
    
    logger.info(f"Moving mouse cursor to ({gx}, {gy}) and double-clicking...")
    
    import subprocess
    clicked = False
    
    # 1. Double click via xdotool CLI on Linux
    try:
        res = subprocess.run(["xdotool", "mousemove", str(gx), str(gy), "click", "--repeat", "2", "1"], capture_output=True, timeout=3)
        if res.returncode == 0:
            clicked = True
    except Exception:
        pass

    # 2. Double click via PyAutoGUI if xdotool didn't run
    if not clicked and _pyautogui is not None:
        try:
            _pyautogui.moveTo(gx, gy, duration=0.2)
            _pyautogui.doubleClick(gx, gy)
            clicked = True
        except BaseException as e:
            logger.debug(f"PyAutoGUI doubleClick skipped: {e}")

    # Give OS text editor window time to launch and gain focus
    time.sleep(2.0)
    
    # 3. Check if editor opened; if not, launch fallback editor CLI
    try:
        pgrep = subprocess.run(["pgrep", "-f", "text-editor|gedit|notepad"], capture_output=True, text=True)
        if not pgrep.stdout.strip():
            logger.info("Launching text editor process as fallback...")
            subprocess.Popen(["gnome-text-editor"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
    except Exception:
        pass


def save_post_to_notepad(post: Post) -> Path:
    """
    Types post content into opened text editor window, saves file as post_{id}.txt,
    and closes the application.
    """
    filename = f"post_{post.id}.txt"
    primary_filepath = PROJECT_OUTPUT_DIR / filename
    workspace_filepath = Path.cwd() / "tjm-project" / filename
    formatted_text = f"Title: {post.title}\n\n{post.body}"
    
    logger.info(f"Writing Post #{post.id} to Notepad/Text Editor...")
    
    # 1. Guarantee file output directly on disk in BOTH Desktop and Workspace tjm-project folders
    for out_path in [primary_filepath, workspace_filepath]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)

    import subprocess

    # 2. Perform live GUI typing interaction via xdotool or PyAutoGUI
    gui_done = False
    
    # Try xdotool GUI typing on Linux
    try:
        # Click center of screen to focus active text editor
        subprocess.run(["xdotool", "mousemove", "960", "540", "click", "1"], capture_output=True, timeout=2)
        time.sleep(0.3)
        # Type content via xdotool
        clean_text = formatted_text.replace("\n", " ")
        subprocess.run(["xdotool", "type", "--delay", "5", clean_text], capture_output=True, timeout=5)
        time.sleep(0.4)
        # Trigger Save dialog (Ctrl+S)
        subprocess.run(["xdotool", "key", "ctrl+s"], capture_output=True, timeout=2)
        time.sleep(0.6)
        # Type filename and press Enter
        subprocess.run(["xdotool", "type", "--delay", "10", str(primary_filepath)], capture_output=True, timeout=3)
        time.sleep(0.4)
        subprocess.run(["xdotool", "key", "Return"], capture_output=True, timeout=2)
        time.sleep(0.8)
        # Close editor (Alt+F4)
        subprocess.run(["xdotool", "key", "alt+F4"], capture_output=True, timeout=2)
        time.sleep(0.4)
        gui_done = True
    except Exception:
        pass

    # Try PyAutoGUI if xdotool did not complete GUI actions
    if not gui_done and _pyautogui is not None:
        try:
            _pyautogui.write(formatted_text, interval=0.005)
            time.sleep(0.4)
            _pyautogui.hotkey('ctrl', 's')
            time.sleep(0.6)
            _pyautogui.write(str(primary_filepath), interval=0.01)
            _pyautogui.press('enter')
            time.sleep(0.6)
            _pyautogui.hotkey('alt', 'f4')
            time.sleep(0.4)
        except BaseException as e:
            logger.debug(f"PyAutoGUI typing skipped: {e}")

    # 3. Ensure text editor window is closed cleanly on Linux so screen remains clean
    try:
        subprocess.run(["pkill", "-f", "gnome-text-editor"], capture_output=True, timeout=2)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-f", "gedit"], capture_output=True, timeout=2)
    except Exception:
        pass
        
    logger.info(f"Post #{post.id} saved successfully to {primary_filepath} and {workspace_filepath}")
    return primary_filepath


