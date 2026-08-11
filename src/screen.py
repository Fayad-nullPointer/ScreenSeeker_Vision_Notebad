"""
Screenshot capture and coordinate mapping utilities
Supports XDG Desktop Portal (Wayland-native), gnome-screenshot/scrot, PIL ImageGrab, PyAutoGUI,
and MSS fallback for full Ubuntu Wayland/X11 compatibility.
"""

import os
import re
import time
import subprocess
import logging
from typing import Tuple, Optional
from PIL import Image, ImageGrab
import mss
import mss.tools
from src.config import SCREEN_RESOLUTION

logger = logging.getLogger(__name__)


def _xdg_portal_screenshot() -> Optional[Image.Image]:
    """
    Takes a screenshot using the XDG Desktop Portal D-Bus API via subprocess.
    This is the ONLY method that works reliably on Ubuntu Wayland.
    Uses gdbus call + dbus-monitor to capture the Response signal and extract the file URI.
    No python-dbus or PyGObject packages required.
    """
    try:
        # Step 1: Start dbus-monitor BEFORE making the screenshot call,
        # so we can catch the Response signal
        monitor_proc = subprocess.Popen(
            [
                "dbus-monitor", "--session",
                "type='signal',interface='org.freedesktop.portal.Request',member='Response'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        # Step 2: Make the screenshot request via gdbus
        result = subprocess.run(
            [
                "gdbus", "call", "--session",
                "--dest", "org.freedesktop.portal.Desktop",
                "--object-path", "/org/freedesktop/portal/desktop",
                "--method", "org.freedesktop.portal.Screenshot.Screenshot",
                "",  # parent_window
                '{"interactive": <false>}'
            ],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0:
            monitor_proc.kill()
            logger.debug(f"XDG portal gdbus call failed: {result.stderr}")
            return None

        # Step 3: Read dbus-monitor output to find the Response signal with the file URI
        # The response contains a "uri" key with "file:///path/to/screenshot.png"
        import select
        file_uri = None
        deadline = time.time() + 5  # 5 second timeout

        output_lines = []
        while time.time() < deadline:
            # Use select to wait for data with a short timeout
            ready, _, _ = select.select([monitor_proc.stdout], [], [], 0.2)
            if ready:
                line = monitor_proc.stdout.readline()
                if not line:
                    break
                output_lines.append(line)
                # Look for the file URI in the dbus-monitor output
                uri_match = re.search(r'string\s+"(file://[^"]+)"', line)
                if uri_match:
                    file_uri = uri_match.group(1)
                    break
            # Also check accumulated output
            full_output = "".join(output_lines)
            uri_match = re.search(r'string\s+"(file://[^"]+)"', full_output)
            if uri_match:
                file_uri = uri_match.group(1)
                break

        monitor_proc.kill()
        monitor_proc.wait()

        if file_uri:
            # Convert file:// URI to filesystem path
            file_path = file_uri.replace("file://", "")
            if os.path.exists(file_path):
                img = Image.open(file_path)
                img.load()  # Force load before file might be cleaned up
                logger.info("Successfully captured live desktop screenshot using XDG Desktop Portal.")
                return img
            else:
                logger.debug(f"XDG Portal returned URI but file not found: {file_path}")
        else:
            logger.debug("XDG Portal: no file URI found in dbus-monitor response")

    except Exception as e:
        logger.debug(f"XDG Portal screenshot exception: {e}")


def _xdg_portal_screenshot_dbus(save_path: str) -> Optional[Image.Image]:
    """
    Takes a screenshot via XDG Desktop Portal using python-dbus with proper signal handling.
    """
    try:
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop
        from gi.repository import GLib

        DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()

        portal = bus.get_object(
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop"
        )
        screenshot_iface = dbus.Interface(portal, "org.freedesktop.portal.Screenshot")

        result_uri = [None]
        loop = GLib.MainLoop()

        def on_response(response, results):
            if response == 0 and "uri" in results:
                result_uri[0] = str(results["uri"])
            loop.quit()

        # Make the screenshot request
        request_path = screenshot_iface.Screenshot(
            "",  # parent window
            {"interactive": dbus.Boolean(False)}
        )

        # Listen for the Response signal on the request object
        bus.add_signal_receiver(
            on_response,
            signal_name="Response",
            dbus_interface="org.freedesktop.portal.Request",
            path=request_path
        )

        # Run event loop with timeout
        GLib.timeout_add(4000, loop.quit)
        loop.run()

        if result_uri[0]:
            # Convert file:// URI to path
            file_path = result_uri[0]
            if file_path.startswith("file://"):
                file_path = file_path[7:]

            img = Image.open(file_path)
            img.load()  # Force load before file might be cleaned up
            logger.info("Successfully captured live desktop screenshot using XDG Desktop Portal (D-Bus).")
            return img

    except Exception as e:
        logger.debug(f"XDG Portal D-Bus screenshot failed: {e}")

    return None


def _try_gnome_screenshot_portal() -> Optional[Image.Image]:
    """
    Uses gnome-screenshot which on Wayland internally goes through the portal.
    Also tries scrot as fallback.
    """
    temp_path = "/tmp/live_desktop_capture.png"
    for tool_cmd in [
        ["gnome-screenshot", "-f", temp_path],
        ["scrot", temp_path],
    ]:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            res = subprocess.run(tool_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            if res.returncode == 0 and os.path.exists(temp_path):
                img = Image.open(temp_path)
                img.load()
                if _is_valid_non_black(img):
                    logger.info(f"Successfully captured live desktop screenshot using {tool_cmd[0]}.")
                    return img
        except Exception:
            pass
    return None


def capture_desktop_screenshot() -> Image.Image:
    """
    Captures full 1920x1080 desktop screenshot on Ubuntu/Linux/Windows.
    Priority order:
      1. XDG Desktop Portal D-Bus (Wayland-native, the ONLY reliable method on Ubuntu Wayland)
      2. gnome-screenshot / scrot CLI (uses portal internally on Wayland)
      3. PIL ImageGrab (X11 / Windows)
      4. PyAutoGUI (X11 / Windows)
      5. MSS (X11 only, returns black on Wayland)
    """

    # 1. XDG Desktop Portal — the only reliable Wayland screenshot method
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
        logger.debug("Wayland session detected. Using XDG Desktop Portal for screenshot.")

        # Try python-dbus portal first
        img = _xdg_portal_screenshot()
        if img and _is_valid_non_black(img):
            return _format_and_resize(img)

        # Try D-Bus portal with python-dbus
        try:
            img = _xdg_portal_screenshot_dbus("/tmp/xdg_portal_screenshot.png")
            if img and _is_valid_non_black(img):
                return _format_and_resize(img)
        except Exception:
            pass

    # 2. gnome-screenshot / scrot (uses portal internally on Wayland)
    img = _try_gnome_screenshot_portal()
    if img and _is_valid_non_black(img):
        return _format_and_resize(img)

    # 3. PIL ImageGrab (Works on X11 and Windows)
    try:
        img = ImageGrab.grab()
        if img and img.size[0] > 0 and _is_valid_non_black(img):
            logger.info("Successfully captured live desktop screenshot using PIL ImageGrab.")
            return _format_and_resize(img)
    except Exception as e:
        logger.debug(f"ImageGrab failed: {e}")

    # 4. PyAutoGUI screenshot
    try:
        import pyautogui
        img = pyautogui.screenshot()
        if img and _is_valid_non_black(img):
            logger.info("Successfully captured live desktop screenshot using PyAutoGUI.")
            return _format_and_resize(img)
    except Exception as e:
        logger.debug(f"PyAutoGUI screenshot failed: {e}")

    # 5. MSS primary monitor grab (X11 only — returns black on Wayland)
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            if _is_valid_non_black(img):
                logger.info("Successfully captured live desktop screenshot using MSS.")
                return _format_and_resize(img)
    except Exception as e:
        logger.debug(f"MSS screenshot failed: {e}")

    logger.warning("All desktop screenshot methods failed or returned black images on Wayland.")
    return Image.new("RGB", SCREEN_RESOLUTION, (30, 30, 35))


def _is_valid_non_black(img: Image.Image) -> bool:
    """Checks if captured screenshot contains actual desktop pixels and is not a black image."""
    try:
        extrema = img.getextrema()
        if isinstance(extrema, list) or isinstance(extrema, tuple):
            if all(e[1] == 0 for e in extrema[:3]):
                return False
        return True
    except Exception:
        return True


def _format_and_resize(img: Image.Image) -> Image.Image:
    """Crops multi-monitor screenshot to primary monitor and resizes to target resolution if needed."""
    img = img.convert("RGB")
    w, h = img.size
    # If screenshot covers multiple monitors (width > 1920 or height > 1080), crop to primary display
    if w > 1920 or h > 1080:
        img = img.crop((0, 0, min(w, 1920), min(h, 1080)))
    if img.size != SCREEN_RESOLUTION:
        img = img.resize(SCREEN_RESOLUTION, Image.Resampling.LANCZOS)
    return img


def crop_sub_image(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crops sub-image patch from full image using bounding box (x_min, y_min, x_max, y_max).
    Preserves 100% pixel detail without downsampling.
    """
    x_min, y_min, x_max, y_max = box
    
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
    """
    x_min, y_min, _, _ = crop_box
    local_x, local_y = local_point
    
    global_x = max(0, min(int(x_min + local_x), SCREEN_RESOLUTION[0] - 1))
    global_y = max(0, min(int(y_min + local_y), SCREEN_RESOLUTION[1] - 1))
    
    return global_x, global_y
