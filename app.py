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
    "a person with no religious symbols, markings, or headwear"
]

# --- 2. Helper Functions for Assets ---
def get_img_html(filename, height="auto", width="auto", style=""):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{data}" style="height: {height}; width: {width}; {style} border-radius: 8px; display: block; margin: 0 auto;">'
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
    if image is None: return "## Step 1 Required\nPlease upload a photo or use the webcam to check your status."
    
    time.sleep(0.5) 
    label, confidence = analyze_image(image)

    religious_items = ["a person wearing a religious hijab", "a person wearing a religious kippah or taqiya cap", "a person wearing a religious sikh turban", "a person wearing a religious cross or crucifix necklace", "a person wearing a religious sikh kirpan ceremonial dagger", "a person wearing a religious sikh kara bracelet", "a person wearing a religious catholic nun's habit", "a person wearing a religious buddhist monk's saffron robe", "a person wearing a religious clerical collar", "a person with a religious bindi, tilak, or tilakah marking on their forehead", "a person wearing religious jewish tzitzit tassels at their waist"]
    face_coverings = ["a person wearing a religious niqab or burqa face covering"]
    safe_items = ["a person wearing a medical face mask", "a person wearing a baseball cap", "a person wearing large headphones", "a person wearing a winter hat", "a person with no religious symbols, markings, or headwear"]

    status, color = "Eligible ✅", "green"
    reason = f"Detected: {label} ({confidence:.1%})"

    if label in religious_items and job in RESTRICTED_JOBS:
        status, color = "Ineligible ❌", "red"
        reason = f"Detected: {label}. This attire is restricted for this role under Bill 21 / Bill 94."
    elif label in face_coverings:
        status, color = "Ineligible ❌", "red"
        reason = f"Detected: {label}. Face coverings are restricted in the Quebec education network (Bill 94)."
    elif label in safe_items:
        reason = f"The AI detected a non-restricted appearance."
    elif confidence < 0.35:
        reason = "No clear religious symbols detected."

    return f"## {status}\n**Details:** {reason}"

# --- 4. UI Data ---
RESTRICTED_JOBS = [
    "Principals, vice-principals, and teachers of public educational institutions",
    "School service centre personnel (support staff, daycare workers, supervisors)",
    "Special education technicians",
    "Janitors or maintenance staff on school premises",
    "Volunteers or contractors providing services to students",
    "Any person who regularly provides services on school premises",
    "Peace officers exercising functions mainly in Quebec",
    "Administrative justices of the peace, clerks, sheriffs, and bankruptcy registrars",
    "Government-employed lawyers, notaries, and criminal/penal prosecuting attorneys",
    "Members or commissioners of provincial boards and tribunals (e.g., Human Rights Tribunal)",
    "President and Vice-Presidents of the National Assembly"
]

# --- 5. UI Layout ---
custom_css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.stat-box { background: var(--background-fill-secondary); padding: 24px; border-radius: 12px; text-align: center; border: 1px solid var(--border-color-primary); margin-bottom: 16px; }
.stat-box h2 { font-size: 2.5em !important; font-weight: 900 !important; color: #ef4444 !important; margin: 0 !important; }
.stat-box p { font-weight: 600 !important; font-size: 1.1em !important; margin: 5px 0 0 0 !important; }
.action-card { background: var(--background-fill-secondary); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color-primary); text-align: center; margin-bottom: 15px; }
.glossary-box { background: var(--background-fill-secondary); padding: 15px; border-radius: 8px; border-left: 4px solid var(--color-accent); margin-bottom: 10px; }
.result-display { padding: 20px; border-radius: 12px; background: var(--background-fill-secondary); border: 2px solid var(--border-color-primary); min-height: 100px; }
"""

with gr.Blocks(title="Banned by 21", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Banned by 21: Quebec Employment Eligibility")
    
    with gr.Tabs() as tabs:
        # --- HOME ---
        with gr.Tab("Home", id="home"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML(get_img_html("banned-by-21.png", style="max-height: 350px;"))
                    gr.Markdown("## The Fight for Civil Liberties in Quebec\nIn 2019, Quebec passed **Bill 21**, followed by **Bill 94** in 2025. These laws prohibit public sector employees from wearing religious symbols while exercising their functions.")
                    start_btn = gr.Button("Enter Eligibility Checker", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML('<div class="stat-box"><h2>71%</h2><p>consider leaving Quebec.</p></div><div class="stat-box"><h2>88%</h2><p>felt less welcoming since 2019.</p></div>')
            
        # --- ELIGIBILITY CHECKER ---
        with gr.Tab("Eligibility Checker", id="checker"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Photo Input")
                    image_input = gr.Image(label="Upload or Use Webcam", type="pil", height=400, sources=["upload", "webcam"])
                    gr.Markdown("*Note: For demonstration, you can upload any of the sample images from the project folder.*")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Job & Status")
                    job_dropdown = gr.Dropdown(label="Select Profession", choices=RESTRICTED_JOBS, value=RESTRICTED_JOBS[0])
                    submit_btn = gr.Button("Check My Status", variant="primary")
                    
                    with gr.Group(elem_classes="result-display"):
                        status_output = gr.Markdown("Waiting for input...")

        # --- HUMAN IMPACT ---
        with gr.Tab("Human Impact", id="human_impact"):
            gr.Markdown("## Real Stories: The Human Cost\nThis video highlights the stories of women who have already lost their jobs following the expansion of the law in 2025.")
            gr.HTML('<div style="margin-bottom: 24px; border-radius: 12px; overflow: hidden;"><iframe width="100%" height="500" src="https://www.youtube.com/embed/urcZnCopNAc" frameborder="0" allowfullscreen></iframe></div>')

        # --- ABOUT THE LAWS ---
        with gr.Tab("About the Laws", id="laws"):
            gr.HTML(get_img_html("banned-by-21-at-work.png", height="300px"))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### The Legislation\n**Bill 21:** Prohibits symbols for authority figures.\n**Bill 94:** Expands ban to all school staff.\n\n### Legal Resources\n* [Bill 21 Official Text](https://www.legisquebec.gouv.qc.ca/en/document/cs/L-0.3)\n* [Bill 94 Official Text](https://www.assnat.qc.ca/en/travaux-parlementaires/projets-loi/projet-loi-94-43-1.html)\n* [Canadian Charter of Rights](https://www.canada.ca/en/canadian-heritage/services/how-rights-protected/guide-canadian-charter-rights-freedoms.html)")
                with gr.Column():
                    gr.Markdown("### Key Terms")
                    gr.HTML("<div class='glossary-box'><strong>Notwithstanding Clause:</strong> Section 33, allowing governments to override Charter rights.</div><div class='glossary-box'><strong>Laicity:</strong> State neutrality toward all religions.</div>")

    # --- PERMANENT ADVOCACY SECTION (RENDERED ON EVERY TAB) ---
    gr.Markdown("---")
    gr.Markdown("## Defend Civil Liberties: Take Action", elem_classes="cta-header")
    gr.Markdown("The NCCM and CCLA have partnered to take the civil liberties battle to the Supreme Court of Canada.", elem_classes="cta-subtext")
    
    with gr.Row():
        with gr.Column(elem_classes="action-card"):
            gr.HTML(get_img_html("nccm-logo.png", height="50px"))
            gr.Markdown("**National Council of Canadian Muslims**\n\nChallenging Quebec's secularism laws to defend the rights of public sector employees.")
            gr.Button("Donate to NCCM", link="https://www.nccm.ca/donate/", variant="primary")
            
        with gr.Column(elem_classes="action-card"):
            gr.HTML(get_img_html("CCLA-logo.png", height="50px"))
            gr.Markdown("**Canadian Civil Liberties Association**\n\nActing as a vigilant watchdog for rights and freedoms across Canada.")
            gr.Button("Donate to CCLA", link="https://ccla.org/donate/", variant="primary")

    # Event Wiring
    start_btn.click(fn=lambda: gr.Tabs(selected="checker"), outputs=tabs)
    submit_btn.click(fn=get_eligibility, inputs=[image_input, job_dropdown], outputs=[status_output], show_progress="full")

    gr.Markdown("---")
    gr.Markdown("*Disclaimer: This tool is for informational and advocacy purposes. Images are processed locally in memory and not stored on any server.*")
    gr.Markdown("*Created by Bilal Shirazi (bilalshirazi.com)*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
