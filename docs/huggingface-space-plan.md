# Hugging Face Space Implementation Plan: Banned by 21

This document outlines the implementation of the "Banned by 21" project as a **Hugging Face Space** using **Gradio**.

## 1. Technical Stack
- **SDK:** Gradio (`gradio`)
- **Backend:** Python 3.10+
- **Machine Learning:** `transformers`, `torch`, `Pillow`
- **Model:** Zero-shot Image Classification (`openai/clip-vit-base-patch32`)
- **Hosting:** Hugging Face Spaces (CPU or GPU tier)

## 2. Component Mapping

| Original Requirement | Hugging Face / Gradio Implementation |
| :--- | :--- |
| **Platform Phase 1: Web App** | **Hugging Face Space (Gradio SDK)** |
| **Computer Vision Module** | **CLIP Zero-Shot Classifier**. Labels including: "person wearing a hijab", "person wearing a kippah", "person wearing a turban", "person wearing a niqab/burqa", "person with face covered". |
| **Job Selection Dropdown** | `gr.Dropdown` populated with restricted professions. |
| **Result UI (Check/X)** | `gr.HTML` with dynamic styling and color-coded eligibility status. |
| **CTA / Donation Page** | `gr.HTML` with a responsive grid for NCCM and CCLA donations. |

## 3. Logic Engine Implementation (Python)

```python
def check_eligibility(image, job):
    # 1. Run CLIP model to detect religious symbols or face coverings
    # 2. Check if the selected job is in the restricted list
    # 3. Return status (Eligible/Ineligible) and UI components
    pass
```

## 4. UI Layout (Gradio Blocks)

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Banned by 21: Eligibility Checker")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Photo")
            job_input = gr.Dropdown(label="Select Profession", choices=ALL_JOBS)
            
    check_btn = gr.Button("Check Eligibility", variant="primary")
    
    # Results Area
    result_output = gr.HTML()
    
    # CTA Area (Donations)
    cta_output = gr.HTML()
```

## 5. Deployment Files
- `app.py`: Main logic and Gradio UI.
- `requirements.txt`: `gradio`, `transformers`, `torch`, `pillow`.
- `README.md`: Metadata for the Hugging Face Space.

## 6. Security & Privacy
- Images are processed in-memory and not stored.
- A disclaimer is provided to the user regarding AI detection and privacy.
