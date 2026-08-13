"""
Configuration parameters for Vision-Based Desktop Automation & ScreenSeekeR Framework
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Target OS & Display settings
SCREEN_RESOLUTION = (1920, 1080)
TARGET_APP_NAME = "Notepad"

# Google GenAI & OpenRouter API Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "google/gemini-3-flash-preview")
GROUNDER_MODEL = os.getenv("GROUNDER_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_MODEL = PLANNER_MODEL

# ScreenSeekeR Paper Hyperparameters (arXiv:2504.07981)
SIGMA = 0.3                # Gaussian centrality scoring variance parameter (Equation 1 in paper)
D_MAX = 2                  # Maximum search depth for recursive visual search
S_MIN = 1280               # Minimum patch size below which direct grounding is invoked (pixels)
DILATION_FACTOR = 0.15     # Box dilation ratio to expand candidate regions (15%)
NMS_THRESHOLD = 0.5        # Non-Maximum Suppression IoU threshold

# Output Directories
DESKTOP_PATH = Path.home() / "Desktop"
PROJECT_OUTPUT_DIR = DESKTOP_PATH / "tjm-project"
ANNOTATED_OUTPUT_DIR = Path.cwd() / "screenshots"

# Ensure required directories exist
PROJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
