# Banned by 21 - Project Context

## Project Overview
"Banned by 21" is a project dedicated to creating an interactive application that helps public sector employees in Quebec determine their job eligibility under the constraints of **Bill 21** (An Act respecting the laicity of the State) and **Bill 94** (An Act to, in particular, reinforce laicity in the education network). The app evaluates images and job titles against these laws to provide an eligibility status.

## Directory Overview
This workspace is currently in the **Planning and Specification phase**. It contains the foundational documentation, logic rules, and UI requirements needed to implement the Phase 1 Web Application and Phase 2 iOS Application.

## Key Files
- `project-specifications.md`: The primary "source of truth" document.
- `GEMINI.md`: This file, providing instructional context for AI interactions.
- `app.py`: Gradio implementation for the Hugging Face Space.

## Critical Next Steps (Pre-Implementation)
Before coding begins, the following "Platform Phase 1" requirements must be addressed:
- [x] **CV Model Selection:** (PROPOSED) Use **CLIP** zero-shot detection on Hugging Face Spaces.

## Planned Architecture & Conventions
- **Platform Phase 1:** Hugging Face Space (Gradio SDK).
- **Platform Phase 2:** iOS Application (Swift/SwiftUI/CoreML).
- **Styling:** Vanilla CSS with responsive CSS Grid (`grid-template-columns: repeat(auto-fit, minmax(300px, 1fr))`).
- **UI Components:** Data-driven donation components mapped from the JSON object in the specifications.
- **Logic:** Strict adherence to the "Logic Engine Rules" (Rules 1-2) defined in the specs.

## Usage
The contents of this directory should be used as the definitive guide for building the "Banned by 21" application. Any implementation must strictly follow the eligibility logic and civil liberties advocacy goals outlined in the specifications.
