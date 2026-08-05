"""
Main Entrypoint Script for Vision-Based Desktop Automation with ScreenSeekeR Grounding
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
from src.automation import fetch_posts, launch_application_at, save_post_to_notepad
from src.annotator import save_annotated_deliverable
from src.simulator import create_synthetic_desktop_image


def generate_annotated_deliverables():
    """
    Generates the 3 mandatory deliverable annotated screenshots showing icon detection in:
      1. Top-Left area (150, 150)
      2. Bottom-Right area (1700, 900)
      3. Center of screen (960, 540)
    """
    logger.info("==================================================================")
    logger.info("Generating Mandatory Deliverable Annotated Screenshots (ScreenSeekeR)")
    logger.info("==================================================================")
    
    test_positions = [
        ("Top_Left", (150, 150), (120, 120, 180, 180)),
        ("Bottom_Right", (1700, 900), (1670, 870, 1730, 930)),
        ("Center", (960, 540), (930, 510, 990, 570)),
    ]
    
    saved_paths = []
    for pos_label, pos_coords, true_box in test_positions:
        logger.info(f"Running ScreenSeekeR Visual Search for scenario: {pos_label} at {pos_coords}...")
        
        # Create synthetic desktop screenshot for test harness
        sim_image = create_synthetic_desktop_image(pos_coords, "Notepad")
        
        # Execute ScreenSeekeR visual search algorithm
        (target_x, target_y), target_box, trace = visual_search("Notepad shortcut icon", sim_image)
        
        logger.info(f"ScreenSeekeR Result [{pos_label}]: Grounded Center=({target_x}, {target_y}), Box={target_box}")
        
        # Save deliverable annotated image
        output_path = save_annotated_deliverable(sim_image, (target_x, target_y), target_box, pos_label)
        saved_paths.append(output_path)
        logger.info(f"Saved annotated screenshot: {output_path}")
        
    logger.info("All 3 deliverable annotated screenshots generated successfully!")
    return saved_paths


def run_automation_loop(sim_mode: bool = False):
    """
    Executes full 10-post automation loop:
      1. Fetch 10 posts from JSONPlaceholder API.
      2. For each post:
         - Capture desktop screenshot.
         - Run ScreenSeekeR to ground Notepad icon -> get (x, y).
         - Double-click to launch Notepad.
         - Write formatted post content and save as post_{id}.txt in Desktop/tjm-project/.
         - Close Notepad and repeat.
    """
    logger.info("==================================================================")
    logger.info("Starting Vision-Based Automation Loop (10 JSONPlaceholder Posts)")
    logger.info("==================================================================")
    
    posts = fetch_posts(count=10)
    
    for idx, post in enumerate(posts, 1):
        logger.info(f"\n--- Processing Post {idx}/10 (ID: {post.id}) ---")
        
        if sim_mode:
            # Simulation desktop capture
            screenshot = create_synthetic_desktop_image((200, 200), "Notepad")
        else:
            # Live desktop screenshot capture
            screenshot = capture_desktop_screenshot()
            
        # Ground Notepad icon using ScreenSeekeR
        (target_x, target_y), target_box, _ = visual_search("Notepad shortcut icon", screenshot)
        logger.info(f"Grounded Notepad Icon at Screen Pixel ({target_x}, {target_y})")
        
        if not sim_mode:
            # Launch application by double clicking grounded center coordinates
            launch_application_at((target_x, target_y))
            
        # Write post and save to Desktop/tjm-project/post_{id}.txt
        saved_file = save_post_to_notepad(post)
        logger.info(f"Completed Post {idx}/10 -> {saved_file}")
        
    logger.info("\n==================================================================")
    logger.info("Automation Pipeline Completed Successfully! All 10 posts processed.")
    logger.info(f"Saved Files Directory: {PROJECT_OUTPUT_DIR}")
    logger.info("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Vision-Based Desktop Automation with ScreenSeekeR Grounding")
    parser.add_argument("--generate-screenshots", action="store_true", help="Generate 3 deliverable annotated screenshots (Top-Left, Bottom-Right, Center)")
    parser.add_argument("--run-automation", action="store_true", help="Run full 10-post JSONPlaceholder automation loop")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode (headless / cross-platform friendly)")
    
    args = parser.parse_args()
    
    if not args.generate_screenshots and not args.run_automation:
        # Default behavior: run screenshot generation + automation test
        generate_annotated_deliverables()
        run_automation_loop(sim_mode=True)
    else:
        if args.generate_screenshots:
            generate_annotated_deliverables()
        if args.run_automation:
            run_automation_loop(sim_mode=args.sim)

if __name__ == "__main__":
    main()
