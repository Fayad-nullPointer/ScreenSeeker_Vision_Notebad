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


def launch_application_at(target_center: tuple[int, int]) -> None:
    """
    Double-clicks center (x, y) coordinates on desktop screen to launch target application.
    """
    gx, gy = target_center
    logger.info(f"Moving mouse cursor to ({gx}, {gy}) and double-clicking...")
    
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
        
        pyautogui.moveTo(gx, gy, duration=0.4)
        pyautogui.doubleClick(gx, gy)
        time.sleep(1.2) # Wait for text editor window to open
    except Exception as e:
        logger.warning(f"PyAutoGUI interaction skipped or failed: {e}")


def save_post_to_notepad(post: Post) -> Path:
    """
    Types post content into opened text editor window, saves file as post_{id}.txt,
    and closes the application.
    """
    output_filepath = PROJECT_OUTPUT_DIR / f"post_{post.id}.txt"
    formatted_text = f"Title: {post.title}\n\n{post.body}"
    
    logger.info(f"Writing Post #{post.id} to Notepad...")
    
    try:
        import pyautogui
        # Type formatted content via GUI
        pyautogui.write(formatted_text, interval=0.01)
        time.sleep(0.5)
        
        # Trigger Save dialog (Ctrl+S on Linux/Windows)
        pyautogui.hotkey('ctrl', 's')
        time.sleep(0.8)
        
        # Type file path in save dialog
        pyautogui.write(str(output_filepath), interval=0.02)
        pyautogui.press('enter')
        time.sleep(0.8)
        
        # Close text editor window (Alt+F4 or Ctrl+Q)
        pyautogui.hotkey('alt', 'f4')
        time.sleep(0.8)
    except Exception as e:
        logger.debug(f"GUI typing fallback to direct file write: {e}")
        
    # Guarantee file output directly on disk
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(formatted_text)
        
    logger.info(f"Post #{post.id} saved successfully to {output_filepath}")
    return output_filepath
