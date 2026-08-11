"""
Main Entrypoint Script for Vision-Based Desktop Automation with ScreenSeekeR Grounding
Supports generalized dynamic icon selection based on user criteria / prompts.
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

from src.config import PROJECT_OUTPUT_DIR, ANNOTATED_OUTPUT_DIR
from src.screen import capture_desktop_screenshot
from src.screen_seeker import visual_search
from src.automation import fetch_posts, launch_application_at, save_post_to_notepad, prepare_clean_desktop
from src.annotator import save_annotated_deliverable
from src.simulator import create_synthetic_desktop_image


def generate_annotated_deliverables(target_icon: str = "Notepad"):
    """
    Generates the 3 mandatory deliverable annotated screenshots showing target icon detection in:
      1. Top-Left area (150, 150)
      2. Bottom-Right area (1700, 900)
      3. Center of screen (960, 540)
    """
    logger.info("==================================================================")
    logger.info(f"Generating Annotated Screenshots for Target Icon: '{target_icon}'")
    logger.info("==================================================================")
    
    test_positions = [
        ("Top_Left", (150, 150), (120, 120, 180, 180)),
        ("Bottom_Right", (1700, 900), (1670, 870, 1730, 930)),
        ("Center", (960, 540), (930, 510, 990, 570)),
    ]
    
    saved_paths = []
    for pos_label, pos_coords, true_box in test_positions:
        logger.info(f"Running ScreenSeekeR Visual Search for '{target_icon}' [{pos_label} at {pos_coords}]...")
        
        # Create synthetic desktop screenshot for test harness
        sim_image = create_synthetic_desktop_image(pos_coords, target_label=target_icon)
        
        # Execute ScreenSeekeR visual search algorithm for user's requested icon
        (target_x, target_y), target_box, trace = visual_search(f"{target_icon} shortcut icon", sim_image)
        
        logger.info(f"ScreenSeekeR Result [{pos_label}]: Grounded Center=({target_x}, {target_y}), Box={target_box}")
        
        # Save deliverable annotated image
        output_path = save_annotated_deliverable(sim_image, (target_x, target_y), target_box, f"{target_icon}_{pos_label}")
        saved_paths.append(output_path)
        logger.info(f"Saved annotated screenshot: {output_path}")
        
    logger.info(f"All deliverable annotated screenshots for '{target_icon}' generated successfully!")
    return saved_paths


def ground_custom_user_icon(target_icon_prompt: str):
    """
    Generalized Grounder: Dynamically detects and grounds ANY user-requested desktop icon or UI element
    (e.g., 'Recycle Bin', 'File Explorer', 'Chrome', 'Notepad', 'VS Code', etc.)
    """
    logger.info("==================================================================")
    logger.info(f"Generalized Icon Grounder - Request: '{target_icon_prompt}'")
    logger.info("==================================================================")
    
    import time
    logger.info(">>> Minimize all windows and show your desktop NOW! Capturing in 3 seconds...")
    for i in range(3, 0, -1):
        logger.info(f"    {i}...")
        time.sleep(1)
    logger.info("    Capturing live desktop screenshot!")
    screenshot = capture_desktop_screenshot()
        
    logger.info(f"Searching for target icon matching criteria: '{target_icon_prompt}'...")
    res = visual_search(target_icon_prompt, screenshot)
    if res and res[0] is not None:
        (target_x, target_y), target_box, trace = res
        logger.info(f"Successfully Grounded '{target_icon_prompt}'!")
        logger.info(f"Center Coordinates: (X={target_x}, Y={target_y})")
        logger.info(f"Bounding Box: {target_box}")
        
        output_path = save_annotated_deliverable(screenshot, (target_x, target_y), target_box, f"custom_{target_icon_prompt.replace(' ', '_')}")
        logger.info(f"Saved annotated proof screenshot to: {output_path}")
        return (target_x, target_y), target_box
    else:
        logger.warning(f"Target '{target_icon_prompt}' not detected on screen.")
        return None, None


def run_automation_loop(target_icon: str = "Notepad"):
    """
    Executes full 10-post live desktop automation loop for target application.
    """
    logger.info("==================================================================")
    logger.info(f"Starting Live Vision-Based Automation Loop for Target: '{target_icon}'")
    logger.info("==================================================================")
    
    posts = fetch_posts(count=10)
    
    import time
    logger.info(">>> Minimize all windows and show your desktop NOW! Starting capture in 3 seconds...")
    for i in range(3, 0, -1):
        logger.info(f"    {i}...")
        time.sleep(1)

    for idx, post in enumerate(posts, 1):
        logger.info(f"\n--- Processing Post {idx}/10 (ID: {post.id}) ---")
        
        prepare_clean_desktop()
        time.sleep(0.5)
        screenshot = capture_desktop_screenshot()
            
        # Ground requested icon dynamically using ScreenSeekeR
        res = visual_search(f"{target_icon} shortcut icon", screenshot)
        if res and res[0] is not None:
            (target_x, target_y), target_box, _ = res
            logger.info(f"Grounded '{target_icon}' Icon at Screen Pixel ({target_x}, {target_y})")
        else:
            logger.warning(f"Could not ground '{target_icon}' shortcut icon. Using fallback center (200, 200).")
            (target_x, target_y) = (200, 200)
        
        launch_application_at((target_x, target_y))
        saved_file = save_post_to_notepad(post)
        logger.info(f"Completed Post {idx}/10 -> {saved_file}")
        
    logger.info("\n==================================================================")
    logger.info("Automation Pipeline Completed Successfully! All 10 posts processed.")
    logger.info(f"Saved Files Directory: {PROJECT_OUTPUT_DIR}")
    logger.info("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Vision-Based Desktop Automation with Dynamic & Generalized Icon Selection")
    parser.add_argument("--target-icon", type=str, default="Notepad", help="Target icon/application prompt (e.g. 'Notepad', 'Recycle Bin', 'File Explorer', 'Chrome')")
    parser.add_argument("--ground-icon", type=str, default=None, help="Dynamically ground ANY custom desktop icon specified by user prompt")
    parser.add_argument("--generate-screenshots", action="store_true", help="Generate deliverable annotated screenshots (Top-Left, Bottom-Right, Center)")
    parser.add_argument("--run-automation", action="store_true", help="Run full 10-post JSONPlaceholder live automation loop")
    
    args = parser.parse_args()
    
    if args.ground_icon:
        ground_custom_user_icon(args.ground_icon)
    elif args.generate_screenshots:
        generate_annotated_deliverables(target_icon=args.target_icon)
    else:
        # Default behavior or --run-automation: execute live 10-post automation loop
        run_automation_loop(target_icon=args.target_icon)

if __name__ == "__main__":
    main()

