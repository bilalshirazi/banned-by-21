import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import glob
import time
import base64
import json
import logging
from packaging import version

# Suppress transformers loading warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- 1. Configuration & AI Model Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data/test_images")

model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy Loading Pattern to save memory on Startup
_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        gr.Info("Initializing AI model for the first time... please wait a few seconds.")
        print("Loading CLIP model (lazy load)...")
        # Load in float32 for CPU stability, but keep memory usage low
        _model = CLIPModel.from_pretrained(model_id, low_cpu_mem_usage=True).to(device)
        _processor = CLIPProcessor.from_pretrained(model_id)
    return _model, _processor

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

RELIGIOUS_ITEMS = [
    "a person wearing a religious hijab", 
    "a person wearing a religious kippah or taqiya cap", 
    "a person wearing a religious sikh turban", 
    "a person wearing a religious cross or crucifix necklace", 
    "a person wearing a sikh kirpan ceremonial dagger", 
    "a person wearing a sikh kara bracelet", 
    "a person wearing a catholic nun's habit", 
    "a person wearing a religious buddhist monk's saffron robe", 
    "a person wearing a clerical collar", 
    "a person with a religious bindi, tilak, or tilakah marking on their forehead", 
    "a person wearing religious jewish tzitzit tassels"
]

FACE_COVERINGS = ["a person wearing a religious niqab or burqa face covering", "a person with their face covered by a cloth or mask"]

# --- 2. Data & Assets ---
RESTRICTED_JOBS = [
    "Principals, vice-principals, and teachers of public educational institutions",
    "School service centre personnel (support, daycare, supervisors, technicians)",
    "Any person who regularly provides services on school premises",
    "Peace officers exercising functions mainly in Quebec",
    "Administrative justices of the peace, clerks, sheriffs, and registrars",
    "Government-employed lawyers, notaries, and prosecutors",
    "Members or commissioners of provincial boards and tribunals",
    "President and Vice-Presidents of the National Assembly"
]

DONATION_DATA = [
    {
        "organization": "National Council of Canadian Muslims (NCCM)",
        "logo": "nccm-logo.png",
        "description": "The NCCM is challenging Bill 21 to defend the rights of public sector employees who are banned from wearing religious symbols, and is providing legal aid and advocacy against Islamophobia.",
        "address": "PO Box 77062, Markham ON, L3P 0C8",
        "email": "info@nccm.ca",
        "actionButtonText": "Donate to NCCM",
        "actionButtonLink": "https://www.nccm.ca/donate/"
    },
    {
        "organization": "Canadian Civil Liberties Association (CCLA)",
        "logo": "CCLA-logo.png",
        "description": "The CCLA acts as a vigilant watchdog for rights and freedoms in Canada. Donations ensure they can continue their work challenging Bill 21 at the Supreme Court.",
        "address": "124 Merton St, Suite #400, Toronto, ON M4S 2Z2",
        "email": "donations@ccla.org",
        "phone": "(416) 646-1407",
        "charityNumber": "Charitable Registration: 75480 2288 RR0001",
        "actionButtonText": "Donate to CCLA",
        "actionButtonLink": "https://ccla.org/donate/"
    }
]

def get_base64_img(filename):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

def generate_donation_cards():
    html = '<div class="donation-grid">'
    for card in DONATION_DATA:
        img_src = get_base64_img(card['logo'])
        meta_info = f'<p class="meta-text">{card.get("address", "")}<br>{card.get("email", "")}'
        if "phone" in card: meta_info += f'<br>{card["phone"]}'
        meta_info += '</p>'
        
        html += f"""
        <div class="donation-card">
            <img src="{img_src}" alt="{card['organization']}" class="card-logo">
            <h3>{card['organization']}</h3>
            <p>{card['description']}</p>
            {meta_info}
            <a href="{card['actionButtonLink']}" target="_blank" class="cta-button">{card['actionButtonText']}</a>
        </div>
        """
    html += '</div>'
    return html

PERSPECTIVE_GALLERY = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))

# --- 3. Logic Engine ---
def analyze_image(image):
    if image is None: return None, 0.0
    
    # Use lazy model loading
    clip_model, clip_processor = get_model()
    
    inputs = clip_processor(text=DETECTION_LABELS, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    max_idx = probs.argmax().item()
    return DETECTION_LABELS[max_idx], probs[0][max_idx].item()

def get_eligibility(image, job):
    if image is None: return "## Status: Pending\nPlease provide an image to begin."
    
    label, confidence = analyze_image(image)

    status, color = "Eligible ✅", "#22c55e"
    reason = f"Detected: {label} ({confidence:.1%})"

    if label in RELIGIOUS_ITEMS and job in RESTRICTED_JOBS:
        status, color = "Ineligible ❌", "#ef4444"
        reason = f"Detected: {label}. Restricted under Bill 21/94 for {job}."
    elif label in FACE_COVERINGS:
        status, color = "Ineligible ❌", "#ef4444"
        reason = f"Detected: {label}. Face coverings are restricted in public service settings (Bill 94)."

    return f"""<div class="result-box" style="border-left: 8px solid {color};">
        <h2 style="color: {color}; margin: 0;">{status}</h2>
        <p style="margin: 10px 0 0 0; font-size: 1.1em;">{reason}</p>
    </div>"""

# --- 4. Custom Styling ---
custom_css = """
/* Reset & Base */
:root {
    --primary-color: #ef4444;
    --accent-color: #1d4ed8;
    --bg-secondary: var(--background-fill-secondary);
    --border-color: var(--border-color-primary);
}

.gradio-container { 
    font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    background-color: var(--background-fill-primary) !important;
}

.main-wrap { 
    max-width: 900px !important; 
    margin: 0 auto !important; 
    padding: 16px 16px calc(16px + env(safe-area-inset-bottom)) 16px !important; 
}

/* Typography */
h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.02em !important; line-height: 1.2 !important; margin-bottom: 0.5rem !important; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; margin-top: 1.5rem !important; }
p { line-height: 1.6 !important; }

/* Remove Double Scrollbars & Fix Height */
body, html { overflow-x: hidden !important; margin: 0; padding: 0; }
.gradio-container { min-height: 100vh !important; height: auto !important; }

/* Mobile-First Tabs (Segmented Control Style) */
.tabs { border: none !important; background: transparent !important; }
.tab-nav { 
    display: flex !important; 
    overflow-x: auto !important; 
    white-space: nowrap !important; 
    gap: 8px !important; 
    padding: 4px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.tab-nav::-webkit-scrollbar { display: none; }

.tab-nav button { 
    flex: 1 1 auto !important;
    border: none !important;
    background: transparent !important;
    padding: 8px 16px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--body-text-color-subdued) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected { 
    background: var(--background-fill-primary) !important;
    color: var(--body-text-color) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

/* Hero Section */
.hero-section { text-align: center; margin-bottom: 2rem; padding: 1rem 0; }
.hero-image { 
    max-width: 320px; 
    width: 90%; 
    border-radius: 16px; 
    box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
    margin: 0 auto 1.5rem; 
    display: block; 
}

/* Stat Boxes */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 1rem 0; }
.stat-box { 
    background: var(--bg-secondary); 
    padding: 1.5rem; 
    border-radius: 16px; 
    text-align: center; 
    border: 1px solid var(--border-color); 
}
.stat-box h2 { font-size: 2.5rem !important; color: var(--primary-color) !important; margin: 0 !important; font-weight: 800; line-height: 1 !important; }
.stat-box p { font-size: 0.9rem !important; margin-top: 0.5rem !important; font-weight: 500; }

/* Result UX */
.result-box { 
    background: var(--background-fill-primary); 
    padding: 24px; 
    border-radius: 16px; 
    margin-top: 1.5rem; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid var(--border-color);
}

/* Donation Cards Grid */
.donation-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 2rem; }
.donation-card { 
    background: var(--bg-secondary); 
    padding: 24px; 
    border-radius: 20px; 
    border: 1px solid var(--border-color); 
    text-align: center; 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
}
.card-logo { height: 48px; margin-bottom: 1rem; object-fit: contain; }
.donation-card h3 { margin: 0 0 0.75rem 0; font-size: 1.25rem; font-weight: 700; }
.donation-card p { font-size: 0.9rem !important; color: var(--body-text-color-subdued); margin-bottom: 1rem; line-height: 1.5; flex-grow: 1; }
.meta-text { font-size: 0.75rem !important; color: var(--body-text-color-subdued); margin-bottom: 1.5rem; line-height: 1.4; }
.cta-button { 
    background: var(--accent-color) !important; 
    color: white !important; 
    padding: 14px 24px !important; 
    border-radius: 12px !important; 
    text-decoration: none !important; 
    font-weight: 700 !important; 
    width: 100% !important; 
    display: block;
}

/* Mobile Optimizations */
@media (max-width: 640px) {
    .main-wrap { padding: 12px !important; }
    h1 { font-size: 1.75rem !important; }
    .hero-image { max-width: 280px; }
    .stat-grid { grid-template-columns: 1fr; }
    .donation-grid { grid-template-columns: 1fr; }
    .tab-nav { padding: 4px !important; border-radius: 10px !important; }
    .tab-nav button { padding: 6px 12px !important; font-size: 0.85rem !important; }
}

/* Video Wrapper */
.video-wrapper { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.15); }
.video-wrapper iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

.glossary-box { background: var(--bg-secondary); padding: 16px; border-radius: 12px; border-left: 4px solid var(--accent-color); margin-bottom: 1rem; }
"""

# Determine where to pass 'css' based on Gradio version
is_v6 = version.parse(gr.__version__) >= version.parse("6.0.0")
blocks_kwargs = {"title": "Banned by 21"}
launch_kwargs = {"server_name": "0.0.0.0"}

if is_v6:
    launch_kwargs["css"] = custom_css
else:
    blocks_kwargs["css"] = custom_css

with gr.Blocks(**blocks_kwargs) as demo:
    with gr.Column(elem_classes="main-wrap"):
        gr.HTML(f"""
            <div class="hero-section">
                <img src="{get_base64_img('banned-by-21.png')}" class="hero-image">
                <h1>Banned by 21: Quebec Employment Eligibility</h1>
            </div>
        """)
        
        with gr.Tabs() as tabs:
            with gr.Tab("Home", id="home"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("## The Fight for Civil Liberties")
                        gr.Markdown("In 2019, Quebec passed **Bill 21**, followed by the expansion of **Bill 94** in 2025. These laws prohibit public sector employees from wearing religious symbols while exercising their functions.")
                        start_btn = gr.Button("Enter Eligibility Checker", variant="primary", size="lg")
                    with gr.Column(scale=2, elem_classes="stat-grid"):
                        gr.HTML("""
                            <div class="stat-box"><h2>71%</h2><p>of professionals consider leaving.</p></div>
                            <div class="stat-box"><h2>88%</h2><p>felt less welcoming.</p></div>
                        """)

            with gr.Tab("Checker", id="checker"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="mobile-stack"):
                        gr.Markdown("### 1. Provide Image")
                        with gr.Group(elem_classes="unified-input-container"):
                            with gr.Tabs() as input_sources:
                                with gr.Tab("Your Photo", id="upload_tab"):
                                    image_input = gr.Image(label="Identify Your Appearance", type="pil", sources=["upload", "webcam"])
                                
                                with gr.Tab("Examples", id="gallery_tab"):
                                    gallery = gr.Gallery(value=PERSPECTIVE_GALLERY, columns=3, rows=4, show_label=False)
                    
                    with gr.Column(scale=1, elem_classes="mobile-stack"):
                        gr.Markdown("### 2. Check Status")
                        with gr.Group(elem_classes="unified-input-container"):
                            job_dropdown = gr.Dropdown(label="Current or Target Role", choices=RESTRICTED_JOBS, value=RESTRICTED_JOBS[0])
                            submit_btn = gr.Button("Check Eligibility", variant="primary", size="lg")
                            status_output = gr.HTML('<div class="result-box">Waiting for selection...</div>')
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🛡️ Privacy")
                        gr.Markdown("Images are processed in transient memory and deleted immediately.")

            with gr.Tab("Logic", id="how_it_works"):
                gr.Markdown("## How it Works")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ⚖️ 1. Legislative Match\nDetected items are matched against the **Restricted Jobs** list.")
                        gr.Markdown("### 🛡️ 2. AI Detection\nWe use CLIP to scan for religious headwear, jewelry, and face coverings.")
                    with gr.Column():
                        gr.Markdown("### 🔒 3. Privacy First\n**No data is stored.** Your image is processed locally and never touches a database.")

            with gr.Tab("Impact", id="impact"):
                gr.Markdown("## The Human Cost")
                gr.HTML('<div class="video-wrapper"><iframe src="https://www.youtube.com/embed/urcZnCopNAc" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>')

            with gr.Tab("Laws", id="laws"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### The Legislation")
                        gr.Markdown("""
                        **Bill 21:** Prohibits symbols for authority figures.
                        **Bill 94:** Expands ban to all school staff.
                        
                        ### Resources
                        - [Bill 21 Text](https://www.legisquebec.gouv.qc.ca/en/document/cs/L-0.3)
                        - [Bill 94 Text](https://www.assnat.qc.ca/en/travaux-parlementaires/projets-loi/projet-loi-94-43-1.html)
                        - [Charter of Rights](https://www.canada.ca/en/canadian-heritage/services/how-rights-protected/guide-canadian-charter-rights-freedoms.html)
                        """)
                    with gr.Column():
                        gr.Markdown("### Key Terms")
                        gr.HTML("""
                            <div class="glossary-box"><strong>Notwithstanding Clause:</strong> Override of Charter rights.</div>
                            <div class="glossary-box"><strong>Laicity:</strong> State religious neutrality.</div>
                        """)
                        gr.HTML(f'<img src="{get_base64_img("banned-by-21-at-work.png")}" style="width: 100%; border-radius: 12px; margin-top: 15px;">')

        gr.Markdown("---")
        gr.HTML(f"""
            <div style="text-align: center; margin-top: 40px;">
                <h2>Defend Civil Liberties: Take Action</h2>
                {generate_donation_cards()}
            </div>
        """)
        
        gr.Markdown("---")
        gr.Markdown("*Disclaimer: This tool is for informational and advocacy purposes. Created by Bilal Shirazi (bilalshirazi.com)*")

    # --- 5. Event Handlers ---
    start_btn.click(fn=lambda: gr.Tabs(selected="checker"), outputs=tabs, api_name=False)
    
    def on_select(evt: gr.SelectData):
        return evt.value['image']['path'], gr.Tabs(selected="upload_tab")

    gallery.select(fn=on_select, outputs=[image_input, input_sources], api_name=False)
    submit_btn.click(fn=get_eligibility, inputs=[image_input, job_dropdown], outputs=status_output, api_name=False)

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
