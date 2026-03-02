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
    "a person wearing a sikh kirpan ceremonial dagger",
    "a person wearing a sikh kara bracelet",
    "a person wearing a catholic nun's habit",
    "a person wearing a religious buddhist monk's saffron robe",
    "a person wearing a clerical collar",
    "a person with a religious bindi, tilak, or tilakah marking on their forehead",
    "a person wearing religious jewish tzitzit tassels",
    "a person wearing a medical face mask",
    "a person wearing a baseball cap",
    "a person wearing large headphones",
    "a person wearing a winter hat",
    "a person with no religious symbols, markings, or headwear"
]

# --- 2. Asset Helpers ---
def get_img_html(filename, height="auto", width="auto", max_height="none"):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{data}" style="height: {height}; width: {width}; max-width: 100%; max-height: {max_height}; object-fit: contain; margin: 0 auto; display: block; border-radius: 12px;">'
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
    if image is None: return "## Status: Pending\nPlease upload an image to begin."
    
    time.sleep(0.5) 
    label, confidence = analyze_image(image)

    religious_items = ["a person wearing a religious hijab", "a person wearing a religious kippah or taqiya cap", "a person wearing a religious sikh turban", "a person wearing a religious cross or crucifix necklace", "a person wearing a sikh kirpan ceremonial dagger", "a person wearing a sikh kara bracelet", "a person wearing a catholic nun's habit", "a person wearing a religious buddhist monk's saffron robe", "a person wearing a clerical collar", "a person with a religious bindi, tilak, or tilakah marking on their forehead", "a person wearing religious jewish tzitzit tassels"]
    face_coverings = ["a person wearing a religious niqab or burqa face covering"]
    safe_items = ["a person wearing a medical face mask", "a person wearing a baseball cap", "a person wearing large headphones", "a person wearing a winter hat", "a person with no religious symbols, markings, or headwear"]

    status, color = "Eligible ✅", "green"
    reason = f"Detected: {label} ({confidence:.1%})"

    if label in religious_items and job in RESTRICTED_JOBS:
        status, color = "Ineligible ❌", "red"
        reason = f"Detected: {label}. Restricted under Bill 21 / Bill 94 for this role."
    elif label in face_coverings:
        status, color = "Ineligible ❌", "red"
        reason = f"Detected: {label}. Face coverings are restricted in the Quebec education network (Bill 94)."
    elif label in safe_items:
        reason = f"The AI detected a non-restricted appearance."
    elif confidence < 0.30:
        reason = "No clear religious symbols detected."

    return f"## {status}\n**Details:** {reason}"

# --- 4. Legislative Data ---
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

PERSPECTIVE_GALLERY = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))

# --- 5. UI Layout ---
custom_css = """
/* Mobile Expert Reset */
* { box-sizing: border-box !important; }
body { overflow-x: hidden !important; width: 100vw; }

/* Core Container & Typography */
.gradio-container { 
    max-width: 100% !important; 
    margin: 0 auto !important; 
    padding: 10px !important; 
}
h1 { 
    font-size: 2.2em !important; 
    text-align: center; 
    margin-bottom: 20px !important; 
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}
h2 { font-size: 1.8em !important; margin-top: 15px !important; overflow-wrap: anywhere !important; }

/* Dashboard / Stats Box */
.stat-box { 
    background: var(--background-fill-secondary); 
    padding: 24px; 
    border-radius: 12px; 
    text-align: center; 
    border: 1px solid var(--border-color-primary); 
    margin-bottom: 16px; 
    width: 100% !important;
    box-sizing: border-box !important;
}
.stat-box h2 { font-size: 2.8em !important; font-weight: 900 !important; color: #ef4444 !important; margin: 0 !important; line-height: 1; }
.stat-box p { font-weight: 600 !important; font-size: 1.1em !important; margin: 8px 0 0 0 !important; }

/* Input Groups & Spacing */
.input-group { 
    background: var(--background-fill-secondary); 
    padding: 20px; 
    border-radius: 12px; 
    border: 1px solid var(--border-color-primary); 
    margin-bottom: 20px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Action / Donation Cards */
.action-card { 
    background: var(--background-fill-secondary); 
    padding: 24px; 
    border-radius: 12px; 
    border: 1px solid var(--border-color-primary); 
    text-align: center; 
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 100% !important;
    box-sizing: border-box !important;
}
.action-card img { max-width: 100% !important; margin-bottom: 15px !important; }

/* Eligibility Checker Results */
.result-display { 
    padding: 20px; 
    border-radius: 12px; 
    background: var(--background-fill-primary); 
    border: 2px solid var(--border-color-primary); 
    margin-top: 15px;
    min-height: 80px; 
}

/* Gallery Optimization */
.gallery-container { 
    width: 100% !important; 
    padding: 15px; 
    background: var(--background-fill-secondary); 
    border-radius: 12px;
    margin-top: 10px;
    box-sizing: border-box !important;
}
#perspective-gallery { height: auto !important; max-height: 400px; overflow-y: auto; }

/* Mobile Specific Tweaks (< 768px) */
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    h1 { font-size: 1.5em !important; }
    h2 { font-size: 1.3em !important; }
    .stat-box h2 { font-size: 2.2em !important; }
    .stat-box { padding: 16px; }
    .action-card { padding: 16px; margin-bottom: 12px; }
    .input-group { padding: 15px; }
    
    /* Force stacking for ANY row on mobile */
    .mobile-stack, .mobile-stack > .row { 
        display: flex !important;
        flex-direction: column !important; 
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    .mobile-stack > .column { 
        width: 100% !important; 
        max-width: 100% !important; 
        min-width: 100% !important; 
        padding: 0 !important;
        margin-bottom: 15px !important;
    }
}
.subtext-small { font-size: 0.85em !important; color: var(--body-text-color-subdued); margin-top: -10px !important; margin-bottom: 10px !important; text-align: center; display: block; }
"""

with gr.Blocks(title="Banned by 21") as demo:
    gr.Markdown("# Banned by 21: Quebec Employment Eligibility")
    
    with gr.Tabs() as tabs:
        # --- HOME ---
        with gr.Tab("Home", id="home"):
            with gr.Row(elem_classes="mobile-stack"):
                with gr.Column(scale=2):
                    gr.HTML(get_img_html("banned-by-21.png", max_height="350px"))
                    gr.Markdown("## The Fight for Civil Liberties\nIn 2019, Quebec passed **Bill 21**, followed by the expansion of **Bill 94** in 2025. These laws prohibit public sector employees from wearing religious symbols while exercising their functions.")
                    start_btn = gr.Button("Enter Eligibility Checker", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML('<div class="stat-box"><h2>71%</h2><p>consider leaving Quebec.</p></div><div class="stat-box"><h2>88%</h2><p>felt less welcoming since 2019.</p></div>')
            
        # --- ELIGIBILITY CHECKER ---
        with gr.Tab("Eligibility Checker", id="checker"):
            with gr.Column(elem_classes="mobile-stack"):
                # Step 1: Image Input
                with gr.Group(elem_classes="input-group"):
                    gr.Markdown("### 1. Photo Input")
                    gr.Markdown("*(We check for religious symbols like hijabs, turbans, crosses, etc., or face coverings)*", elem_classes="subtext-small")
                    image_input = gr.Image(label="Upload or Use Webcam", type="pil", height=350, sources=["upload", "webcam"])
                
                # Step 2: Job & Status (Now directly below image for better mobile UX)
                with gr.Group(elem_classes="input-group"):
                    gr.Markdown("### 2. Job & Status")
                    job_dropdown = gr.Dropdown(label="Select Your Profession", choices=RESTRICTED_JOBS, value=RESTRICTED_JOBS[0])
                    submit_btn = gr.Button("Check My Eligibility", variant="primary", size="lg")
                    
                    with gr.Group(elem_classes="result-display"):
                        status_output = gr.Markdown("Waiting for input...")

                # Step 3: Gallery (Moved to bottom as it's secondary to the check)
                with gr.Column(elem_classes="gallery-container"):
                    gr.Markdown("### 📸 Community Perspectives\nClick an image to see how the law affects different Canadians.")
                    gallery = gr.Gallery(
                        value=PERSPECTIVE_GALLERY,
                        columns=[6, 8, 12], # 6 mobile, 8 tablet, 12 desktop
                        rows=2,
                        height="auto",
                        allow_preview=False,
                        show_label=False,
                        elem_id="perspective-gallery"
                    )

        # --- HOW IT WORKS ---
        with gr.Tab("How it Works", id="how_it_works"):
            gr.Markdown("## How the Determination is Made")
            with gr.Row(elem_classes="mobile-stack"):
                with gr.Column():
                    gr.Markdown("### 🛡️ 1. AI Detection (CLIP)\nThe app uses a computer vision model to scan the image for specific religious symbols and face coverings prohibited under Bill 21 and Bill 94.")
                    gr.Markdown("**What we look for:**\n* Religious headwear (Hijab, Turban, Kippah, etc.)\n* Religious jewelry (Crosses, Kara, etc.)\n* Face coverings (Niqab, Burqa)\n* Religious markings (Bindi, Tilakah)")
                with gr.Column():
                    gr.Markdown("### ⚖️ 2. Legislative Match\nThe detected items are then compared against the **Restricted Jobs** list. If you wear a symbol and work in a covered role (like teaching), the law deems you 'Ineligible'.")
                    gr.Markdown("### 🔒 3. Privacy First\n**No data is stored.** Your image is processed in your browser's memory and deleted the moment you close the tab.")

        # --- HUMAN IMPACT ---
        with gr.Tab("Human Impact", id="human_impact"):
            gr.Markdown("## Real Stories: The Human Cost\nThis video highlights the stories of women who have already lost their jobs following the expansion of the law in 2025.")
            gr.HTML('<div class="video-container"><iframe src="https://www.youtube.com/embed/urcZnCopNAc" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>')

        # --- ABOUT THE LAWS ---
        with gr.Tab("About the Laws", id="laws"):
            gr.HTML(get_img_html("banned-by-21-at-work.png", height="350px"))
            with gr.Row(elem_classes="mobile-stack"):
                with gr.Column():
                    gr.Markdown("### The Legislation\n**Bill 21:** Prohibits symbols for authority figures.\n**Bill 94:** Expands ban to all school staff.\n\n### Legal Resources\n* [Bill 21 Official Text](https://www.legisquebec.gouv.qc.ca/en/document/cs/L-0.3)\n* [Bill 94 Official Text](https://www.assnat.qc.ca/en/travaux-parlementaires/projets-loi/projet-loi-94-43-1.html)\n* [Canadian Charter of Rights](https://www.canada.ca/en/canadian-heritage/services/how-rights-protected/guide-canadian-charter-rights-freedoms.html)")
                with gr.Column():
                    gr.Markdown("### Key Terms")
                    gr.HTML("<div class='glossary-box'><strong>Notwithstanding Clause:</strong> Section 33, allowing governments to override Charter rights.</div><div class='glossary-box'><strong>Laicity:</strong> State neutrality toward all religions.</div>")

    # --- PERMANENT ADVOCACY SECTION ---
    gr.Markdown("---")
    gr.Markdown("## Defend Civil Liberties: Take Action", elem_classes="cta-header")
    gr.Markdown("The NCCM and CCLA have partnered to take the civil liberties battle to the Supreme Court of Canada.")
    
    with gr.Row(elem_classes="mobile-stack"):
        with gr.Column(elem_classes="action-card"):
            gr.HTML(get_img_html("nccm-logo.png", height="50px"))
            gr.Markdown("**National Council of Canadian Muslims**\n\nChallenging Quebec's secularism laws to defend the rights of public sector employees.")
            gr.Button("Donate to NCCM", link="https://www.nccm.ca/donate/", variant="primary")
            
        with gr.Column(elem_classes="action-card"):
            gr.HTML(get_img_html("CCLA-logo.png", height="50px"))
            gr.Markdown("**Canadian Civil Liberties Association**\n\nActing as a vigilant watchdog for rights and freedoms across Canada.")
            gr.Button("Donate to CCLA", link="https://ccla.org/donate/", variant="primary")

    # Event Wiring
    start_btn.click(fn=lambda: gr.Tabs(selected="checker"), outputs=tabs, api_name=False)
    
    # DEFENSIVE SELECT HANDLER: Fixes the 'NoneType' and 'AttributeError' bugs
    def on_select(evt: gr.SelectData):
        if evt is not None and hasattr(evt, 'value'):
            # Return the path of the selected image
            return evt.value['image']['path']
        return None

    gallery.select(fn=on_select, outputs=image_input, api_name=False)
    
    submit_btn.click(fn=get_eligibility, inputs=[image_input, job_dropdown], outputs=[status_output], show_progress="full", api_name=False)

    gr.Markdown("---")
    gr.Markdown("*Disclaimer: This tool is for informational and advocacy purposes. Images are processed locally in memory and not stored on any server.*")
    gr.Markdown("*Created by Bilal Shirazi (bilalshirazi.com)*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft(), css=custom_css)
