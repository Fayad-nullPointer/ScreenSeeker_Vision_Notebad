# Vision-Based Desktop Automation with Dynamic Icon Grounding (ScreenSeekeR)

This repository implements a vision-based desktop automation system in Python that dynamically locates and interacts with desktop application icons (Notepad) on a 1920x1080 resolution environment (Linux Ubuntu / Windows 10/11). 

The visual grounding engine is directly adapted from the **ScreenSeekeR** research paper ([arXiv:2504.07981](https://arxiv.org/pdf/2504.07981)), utilizing coarse-to-fine position inference, box dilation, Gaussian centrality scoring, NMS, and recursive high-resolution sub-image grounding.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Install `uv` package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Project Dependencies
Synchronize dependencies using `uv`:
```bash
uv sync
```

### 3. Environment Configuration (OpenRouter API)
Create a `.env` file in the project root directory:
```env
OPENROUTER_API_KEY=sk-or-v1-YOUR_ACTUAL_KEY_HERE
OPENROUTER_MODEL=openai/gpt-4o-2024-11-20
```

---

## 💻 Commands to Run

### Command A: Generate Mandatory Deliverable Screenshots
Generates the 3 mandatory annotated screenshots showing icon detection in Top-Left, Bottom-Right, and Center desktop locations:
```bash
uv run main.py --generate-screenshots
```
*Outputs saved to `./screenshots/` directory:*
- `annotated_top_left.png`
- `annotated_bottom_right.png`
- `annotated_center.png`

### Command B: Run 10-Post Automation Loop (Simulation Mode)
Executes the full pipeline in headless/simulated mode (ideal for testing without moving physical mouse):
```bash
uv run main.py --run-automation --sim
```

### Command C: Run Full Desktop Automation Loop (Live Mode)
Executes the full pipeline on your active Ubuntu/Windows desktop screen:
```bash
uv run main.py --run-automation
```
*Outputs saved to `~/Desktop/tjm-project/` directory:*
- `post_1.txt` ... `post_10.txt`

---

## 📂 Repository Structure

```
.
├── design_doc.md         # Part 1: Formal Design Document
├── pyproject.toml        # uv package configuration
├── main.py               # Main CLI entrypoint
└── src/
    ├── config.py         # Hyperparameters (sigma=0.3, D_max=2, S_min=1280)
    ├── screen.py         # Screenshot capture & coordinate projection
    ├── planner.py        # Position Inference (OpenRouter MLLM Planner)
    ├── grounder.py       # High-precision Visual Grounder
    ├── screen_seeker.py  # Algorithm 1: ScreenSeekeR Engine (Dilation, Scoring, NMS)
    ├── automation.py     # JSONPlaceholder API fetcher & OS GUI driver
    ├── annotator.py      # Visual debug annotator
    └── simulator.py      # Desktop icon simulator for test harness
```
