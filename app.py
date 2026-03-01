import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import glob
import time
import base64

# --- 1. Configuration & AI Model Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data/test_images")

model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

DETECTION_LABELS = [
    "a person wearing a religious hijab",
    "a person wearing a religious kippah or taqiya cap",
    "a person wearing a religious sikh turban",
    "a person wearing a religious niqab or burqa face covering",
    "a person with their face covered by a cloth or mask",
    "a person wearing a religious cross or crucifix necklace",
    "a person wearing a religious sikh kirpan ceremonial dagger",
    "a person wearing a religious sikh kara bracelet",
    "a person wearing a religious catholic nun's habit",
    "a person wearing a religious buddhist monk's saffron robe",
    "a person wearing a religious clerical collar",
    "a person with a religious bindi, tilak, or tilakah marking on their forehead",
    "a person wearing religious jewish tzitzit tassels at their waist",
    "a person wearing a medical face mask",
    "a person wearing a baseball cap",
    "a person wearing large headphones",
    "a person wearing a winter hat",
    "a person with no religious headwear",
    "a person with no religious symbols",
    "a regular professional portrait with no accessories",
    "a standard business headshot of an uncovered face",
    "a person with no religions markings",
    "a person with no religions attire",
    "a secular person"
]

# --- 2. Asset Helpers ---
def get_base64_img(filename, style=""):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{data}" style="{style}">'
    return ""

# --- 3. Core Logic Functions ---
def analyze_image(image):
    if image is None: return None, 0.0
    inputs = processor(text=DETECTION_LABELS, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    max_idx = probs.argmax().item()
    return DETECTION_LABELS[max_idx], probs[0][max_idx].item()

def get_eligibility(image, job):
    if image is None: return "Upload a photo to see your eligibility status."
    time.sleep(0.5) 
    label, confidence = analyze_image(image)

    religious_items = ["a person wearing a religious hijab", "a person wearing a religious kippah or taqiya cap", "a person wearing a religious sikh turban", "a person wearing a religious cross or crucifix necklace", "a person wearing a religious sikh kirpan ceremonial dagger", "a person wearing a religious sikh kara bracelet", "a person wearing a religious catholic nun's habit", "a person wearing a religious buddhist monk's saffron robe", "a person wearing a religious clerical collar", "a person with a religious bindi, tilak, or tilakah marking on their forehead", "a person wearing religious jewish tzitzit tassels at their waist"]
    face_coverings = ["a person wearing a religious niqab or burqa face covering", "a person with their face covered by a cloth or mask"]
    
    baseline_items = ["a person with no religious headwear", "a person with no religious symbols", "a regular professional portrait with no accessories", "a standard business headshot of an uncovered face", "a person with no religions markings", "a person with no religions attire", "a secular person", "a person wearing a medical face mask", "a person wearing a baseball cap", "a person wearing large headphones", "a person wearing a winter hat"]

    status, color = "Eligible ✅", "green"
    reason = f"Detected: {label} ({confidence:.1%})"

    if label in religious_items and job in RESTRICTED_JOBS:
        status, color = "Ineligible ❌", "red"
        reason = f"Detected: {label}. Restricted under Bill 21 / Bill 94 for this role."
    elif label in face_coverings:
        status, color = "Ineligible ❌", "red"
        reason = f"Detected: {label}. Face coverings are restricted in the Quebec education network (Bill 94)."
    elif label in baseline_items:
        reason = "The AI detected a clear appearance with no restricted symbols."
    elif confidence < 0.35:
        reason = "No religious symbols detected with high confidence."

    return f"## {status}\n**Details:** {reason}"

# --- 4. UI Data ---
RESTRICTED_JOBS = ["Principals, vice-principals, and teachers of public educational institutions", "School service centre personnel (support staff, daycare workers, supervisors)", "Special education technicians", "Janitors or maintenance staff on school premises", "Volunteers or contractors providing services to students", "Any person who regularly provides services on school premises", "Peace officers exercising functions mainly in Quebec", "Administrative justices of the peace, clerks, sheriffs, and bankruptcy registrars", "Government-employed lawyers, notaries, and criminal/penal prosecuting attorneys", "Members or commissioners of provincial boards and tribunals (e.g., Human Rights Tribunal)", "President and Vice-Presidents of the National Assembly"]

EXAMPLE_IMAGES = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
EXAMPLES = [[img] for img in EXAMPLE_IMAGES]

# --- 5. Layout Logic ---
def build_action_html():
    nccm_logo = get_base64_img("nccm-logo.png", "height: 50px; margin: 0 auto 15px auto; display: block;")
    ccla_logo = get_base64_img("CCLA-logo.png", "height: 50px; margin: 0 auto 15px auto; display: block;")
    return f"""
    <div style="margin-top: 40px; border-top: 2px solid var(--border-color-primary); padding-top: 30px;">
        <h2 style="text-align: center; font-weight: 800; margin-bottom: 20px;">Defend Civil Liberties: Take Action</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
            <div style="background: var(--background-fill-secondary); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color-primary); text-align: center;">
                {nccm_logo}
                <h3 style="font-weight: 800;">NCCM</h3>
                <p style="font-size: 0.9em; margin-bottom: 15px;">Defending the rights of public sector employees and providing legal aid.</p>
                <a href="https://www.nccm.ca/donate/" target="_blank" style="display: block; background: var(--button-primary-background-fill); color: var(--button-primary-text-color); padding: 10px; border-radius: 6px; text-decoration: none; font-weight: 600;">Donate to NCCM</a>
            </div>
            <div style="background: var(--background-fill-secondary); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color-primary); text-align: center;">
                {ccla_logo}
                <h3 style="font-weight: 800;">CCLA</h3>
                <p style="font-size: 0.9em; margin-bottom: 15px;">Acting as a vigilant watchdog for rights and freedoms across Canada.</p>
                <a href="https://ccla.org/donate/" target="_blank" style="display: block; background: var(--button-primary-background-fill); color: var(--button-primary-text-color); padding: 10px; border-radius: 6px; text-decoration: none; font-weight: 600;">Donate to CCLA</a>
            </div>
        </div>
    </div>
    """

custom_css = """
.gradio-container { max-width: 1200px !important; }
.stat-box { background: var(--background-fill-secondary); padding: 24px; border-radius: 12px; text-align: center; border: 1px solid var(--border-color-primary); margin-bottom: 16px; }
.stat-box h2 { font-size: 2.5em !important; font-weight: 900 !important; color: #ef4444 !important; }
.result-box { background: var(--background-fill-secondary); padding: 20px; border-radius: 12px; border: 2px solid var(--border-color-primary); }
.glossary-box { background: var(--background-fill-secondary); padding: 15px; border-radius: 8px; border-left: 4px solid var(--color-accent); margin-bottom: 10px; }
"""

with gr.Blocks(title="Banned by 21", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Banned by 21: Quebec Employment Eligibility")
    
    with gr.Tabs() as tabs:
        with gr.Tab("Home", id="home"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML(get_base64_img("banned-by-21.png", "max-height: 350px; border-radius: 12px;"))
                    gr.Markdown("## The Fight for Civil Liberties\nIn 2019, Quebec passed **Bill 21**, followed by **Bill 94** in 2025. These laws prohibit public sector employees from wearing religious symbols.")
                    start_btn = gr.Button("Enter Eligibility Checker", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML('<div class="stat-box"><h2>71%</h2><p>consider leaving Quebec.</p></div><div class="stat-box"><h2>88%</h2><p>felt less welcoming since 2019.</p></div>')
            gr.HTML(build_action_html())

        with gr.Tab("Eligibility Checker", id="checker"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Photo Input")
                    image_input = gr.Image(label="Upload or Webcam", type="pil", height=400, sources=["upload", "webcam"])
                    gr.Examples(examples=EXAMPLES, inputs=[image_input], label="Try Example", examples_per_page=12)
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Job & Status")
                    job_dropdown = gr.Dropdown(label="Select Profession", choices=RESTRICTED_JOBS, value=RESTRICTED_JOBS[0])
                    submit_btn = gr.Button("Check My Status", variant="primary")
                    status_output = gr.Markdown("Status will appear here.")
            gr.HTML(build_action_html())

        with gr.Tab("About the Laws", id="laws"):
            gr.HTML(get_base64_img("banned-by-21-at-work.png", "height: 200px; width: auto; display: block; margin: 0 auto 20px auto;"))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### The Legislation\n**Bill 21:** Prohibits symbols for authority figures.\n**Bill 94:** Expands ban to all school staff.")
                with gr.Column():
                    gr.Markdown("### Key Terms")
                    gr.HTML("<div class='glossary-box'><strong>Laicity:</strong> State neutrality toward all religions.</div>")
            gr.HTML(build_action_html())

    start_btn.click(fn=lambda: gr.Tabs(selected="checker"), outputs=tabs)
    submit_btn.click(fn=get_eligibility, inputs=[image_input, job_dropdown], outputs=[status_output], show_progress="full")

    gr.Markdown("---")
    gr.Markdown("*Created by Bilal Shirazi (bilalshirazi.com)*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
