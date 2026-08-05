# Design Document: Vision-Based Desktop Automation with ScreenSeekeR Dynamic Icon Grounding

## 1. Executive Summary & Objective

This document outlines the architectural design for a vision-based desktop automation system built in Python. The system dynamically locates and interacts with desktop application icons—specifically the **Notepad** shortcut—on a 1920x1080 resolution environment (supporting Linux Ubuntu and Windows 10/11). 

Rather than relying on brittle hardcoded coordinates or rigid template matching (which fail under theme changes, scaling, or pop-ups), our visual grounding engine implements the **ScreenSeekeR** framework ([arXiv:2504.07981](https://arxiv.org/pdf/2504.07981)). ScreenSeekeR uses a coarse-to-fine visual search driven by a Multimodal Large Language Model (MLLM) Planner and a Visual Grounder to achieve zero-shot, robust UI element localization.

Once grounded, the automation engine double-clicks the target icon to launch Notepad, fetches 10 posts from the JSONPlaceholder API (`https://jsonplaceholder.typicode.com/posts`), types the post contents, saves each post as `post_{id}.txt` inside `Desktop/tjm-project/`, closes Notepad, and repeats the loop with fresh screenshot grounding.

---

## 2. System Architecture & Component Diagram

The system follows a modular, decoupled architecture comprising five core subsystems:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Automation Orchestrator                              │
│                                   (main.py)                                      │
└──────┬────────────────────────────────────────────────────────────────────┬──────┘
       │                                                                    │
       ▼                                                                    ▼
┌───────────────┐                                                  ┌────────────────┐
│ API Integrator│                                                  │ Screen Capture │
│ (JSONPlaceholder)                                                │   (mss / PIL)  │
└──────┬────────┘                                                  └────────┬───────┘
       │                                                                    │
       │ Fetch 10 posts                                    1080p Screenshot │
       ▼                                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            ScreenSeekeR Search Engine                            │
│                        (src/screen_seeker.py & planner.py)                       │
│                                                                                  │
│ 1. Position Inference (Planner via OpenRouter / MLLM)                             │
│    -> Identifies candidate interest regions & neighbor anchors                    │
│ 2. Box Dilation                                                                  │
│    -> Expands search bounds by dilation factor (15%)                             │
│ 3. Gaussian Centrality Scoring & NMS                                             │
│    -> Ranks patches using s = exp(-((x'-0.5)^2 + (y'-0.5)^2)/2σ^2)               │
│ 4. Crop & High-Resolution Sub-Image Search                                       │
│    -> Preserves 100% pixel detail without downsampling degradation               │
│ 5. Local Grounding & Coordinate Translation                                      │
│    -> Maps patch (cx, cy) to global (X_screen, Y_screen)                         │
└────────────────────────────────────────────────────────┬─────────────────────────┘
                                                         │
                                                         │ Absolute (X, Y)
                                                         ▼
                                               ┌──────────────────┐
                                               │ OS Interaction   │
                                               │ (PyAutoGUI / OS) │
                                               └──────────────────┘
```

---

## 3. Detailed Grounding Strategy: ScreenSeekeR Integration

### 3.1 The High-Resolution Grounding Bottleneck
Standard MLLMs fail on high-resolution GUI grounding (accuracy drops to ~18.9% on ScreenSpot-Pro) because 1920x1080 screenshots are downsampled to lower resolutions (e.g. 448x448) before being tokenized. Small desktop icons (32x32 to 48x48 pixels) degenerate into unidentifiable blurry patches.

### 3.2 The 5-Step ScreenSeekeR Pipeline

1. **Position Inference (Planner Step)**:
   - The screenshot $I_{full}$ (1920x1080) and instruction $T$ ("Notepad shortcut icon") are sent to the OpenRouter MLLM Planner (e.g. `openai/gpt-4o` or `qwen/qwen-2.5-vl-72b-instruct`).
   - The Planner returns bounding box candidate regions $[y_{min}, x_{min}, y_{max}, x_{max}]$ in normalized scale, leveraging desktop spatial layout knowledge (e.g., desktop grid edges, proximity to system icons).

2. **Box Dilation**:
   - To prevent cutting off icon edges due to imperfect initial predictions, candidate boxes are expanded:
     $$\text{Box}_{dilated} = [x_{min} - \Delta x, y_{min} - \Delta y, x_{max} + \Delta x, y_{max} + \Delta y]$$
   - Default dilation margin: $15\%$ of box dimension.

3. **Gaussian Centrality Scoring & NMS**:
   - Each candidate box receives a centrality score based on how close predicted points fall to the center of candidate bounds:
     $$s = \exp\left(-\frac{(x' - 0.5)^2 + (y' - 0.5)^2}{2\sigma^2}\right)$$
     where $x', y' \in [0, 1]$ are relative point coordinates within the candidate patch, and $\sigma = 0.3$.
   - Non-Maximum Suppression (NMS) with IoU threshold $0.5$ eliminates overlapping candidate boxes.

4. **Sub-Image Crop & Direct Grounding**:
   - The top candidate patch is cropped directly from the original uncompressed 1080p image buffer.
   - Because the crop size is smaller than the model's native maximum patch size ($S_{min} \le 1280$), zero image downsampling occurs.
   - The Visual Grounder extracts sub-pixel bounding box $(x_1, y_1, x_2, y_2)_{patch}$ and center point $(cx_{patch}, cy_{patch})$.

5. **Coordinate Translation**:
   - Patch-local coordinates are projected back to global 1920x1080 desktop coordinates:
     $$X_{global} = X_{crop\_start} + cx_{patch}$$
     $$Y_{global} = Y_{crop\_start} + cy_{patch}$$

---

## 4. Key Trade-offs & Failure Modes Analysis

| Scenario / Challenge | Failure Risk | ScreenSeekeR Mitigation Strategy |
| :--- | :--- | :--- |
| **Unexpected Pop-up Windows** | High for template matching; pop-up covers icon | Planner re-evaluates screen context; if icon is occluded, selects alternate candidate or dismisses pop-up. |
| **Dark / Light Desktop Themes** | High for pixel matching; colors change | Semantic visual grounding relies on icon geometry & text labels ("Notepad") rather than raw RGB color masks. |
| **Desktop Icon Size Variations** | Medium; small vs large desktop icons | Box dilation expands search bounds; cropped sub-image search scales naturally regardless of icon size. |
| **Multiple Identical Text Documents** | Medium; wrong text file opened | Prompting planner to identify the specific shortcut arrow overlay / "Notepad" application launcher icon. |
| **API Latency vs Accuracy** | High latency if searching recursively | Search depth capped at $D_{max} = 2$. High confidence candidate skips secondary recursion. |

---

## 5. Automation Workflow & API Specifications

1. **HTTP API Fetching**:
   - Endpoint: `GET https://jsonplaceholder.typicode.com/posts`
   - Data structure parsed into Pydantic models: `Post(id=int, title=str, body=str)`.

2. **Notepad Interaction Loop**:
   - For $i \in \{1 \dots 10\}$:
     1. Capture fresh desktop screenshot (`mss`).
     2. Invoke `ScreenSeekeR.ground("Notepad icon")` $\rightarrow$ returns $(X_{target}, Y_{target})$.
     3. Double-click $(X_{target}, Y_{target})$.
     4. Wait $1.0\text{s}$ for window focus.
     5. Type formatted post text:
        ```text
        Title: {post.title}

        {post.body}
        ```
     6. Trigger Save (`Ctrl+S` or `Ctrl+O`), enter file path: `Desktop/tjm-project/post_{id}.txt`, press `Enter`.
     7. Close editor (`Alt+F4` or `Ctrl+Q`).

---

## 6. Deliverables & Verification Plan

### Required Artifacts:
1. **GitHub Repository Structure** managed via `uv`.
2. **Design Document** (`design_doc.md`).
3. **3 Annotated Screenshots**:
   - `annotated_top_left.png`: Notepad icon detected in Top-Left quadrant.
   - `annotated_bottom_right.png`: Notepad icon detected in Bottom-Right quadrant.
   - `annotated_center.png`: Notepad icon detected in Center screen area.
4. **Automated File Output**: 10 text files saved in `~/Desktop/tjm-project/post_1.txt` through `post_10.txt`.
