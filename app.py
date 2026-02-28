import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import os
import glob

# --- Configuration & Data ---

# Load CLIP model and processor
model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# Labels for zero-shot detection
DETECTION_LABELS = [
    "a person wearing a hijab",
    "a person wearing a kippah or taqiya",
    "a person wearing a Sikh turban",
    "a person wearing a niqab or burqa (face covered)",
    "a person with their face covered by a cloth or mask",
    "a person wearing a religious cross or crucifix necklace",
    "a person wearing a Sikh kirpan ceremonial dagger",
    "a person wearing a Sikh kara bracelet",
    "a person wearing a Catholic nun's habit and veil",
    "a person wearing a Buddhist monk's saffron robe",
    "a person wearing a clerical collar",
    "a person with a religious tilak, tilakah, or bindi marking on their forehead",
    "a person wearing Jewish tzitzit or tzatzit tassels at their waist",
    "a person wearing a baseball cap",
    "a person wearing a winter beanie or hat",
    "a person wearing large headphones",
    "a person with no headwear or religious symbols"
]

# Restricted Jobs
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

DONATION_DATA = [
    {
        "organization": "National Council of Canadian Muslims (NCCM)",
        "logo": "assets/nccm-logo.png",
        "description": "The NCCM is challenging Quebec's secularism laws to defend the rights of public sector employees and is providing legal aid and advocacy against Islamophobia.",
        "address": "PO Box 77062, Markham ON, L3P 0C8",
        "email": "info@nccm.ca",
        "actionButtonText": "Donate to NCCM",
        "actionButtonLink": "https://www.nccm.ca/donate/"
    },
    {
        "organization": "Canadian Civil Liberties Association (CCLA)",
        "logo": "assets/CCLA-logo.png",
        "description": "The CCLA acts as a vigilant watchdog for rights and freedoms in Canada. Donations ensure they can continue their work challenging Bill 21 and Bill 94 at the Supreme Court.",
        "address": "124 Merton St, Suite #400, Toronto, ON M4S 2Z2",
        "email": "donations@ccla.org",
        "phone": "(416) 646-1407",
        "charityNumber": "Charitable Registration: 75480 2288 RR0001",
        "actionButtonText": "Donate to CCLA",
        "actionButtonLink": "https://ccla.org/donate/"
    }
]

# Get example images
EXAMPLE_IMAGES = sorted(glob.glob("data/test_images/*.png"))
EXAMPLES = [[img] for img in EXAMPLE_IMAGES]

# --- Logic Functions ---

def analyze_image(image):
    if image is None:
        return None, 0.0
    
    inputs = processor(text=DETECTION_LABELS, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = outputs.logits_per_image.softmax(dim=1)
    max_idx = probs.argmax().item()
    confidence = probs[0][max_idx].item()
    
    return DETECTION_LABELS[max_idx], confidence

def get_eligibility(image, job):
    label, confidence = analyze_image(image)
    if label is None:
        return "Please upload an image."

    religious_items = [
        "a person wearing a hijab", 
        "a person wearing a kippah or taqiya", 
        "a person wearing a Sikh turban", 
        "a person wearing a religious cross or crucifix necklace",
        "a person wearing a Sikh kirpan ceremonial dagger",
        "a person wearing a Sikh kara bracelet",
        "a person wearing a Catholic nun's habit and veil",
        "a person wearing a Buddhist monk's saffron robe",
        "a person wearing a clerical collar",
        "a person with a religious tilak, tilakah, or bindi marking on their forehead",
        "a person wearing Jewish tzitzit or tzatzit tassels at their waist"
    ]
    face_coverings = [
        "a person wearing a niqab or burqa (face covered)", 
        "a person with their face covered by a cloth or mask"
    ]
    
    religious_symbol_detected = label in religious_items
    face_covered = label in face_coverings
    
    status = "Eligible ✅"
    reason = f"Detected: {label} ({confidence:.1%})"
    color = "green"
    
    if religious_symbol_detected and job in RESTRICTED_JOBS:
        status = "Ineligible ❌"
        reason = f"Detected: {label}. Under current legislation (Bill 21/Bill 94), religious symbols are banned for this role in Quebec's public sector."
        color = "red"
        
    if face_covered:
        status = "Ineligible ❌"
        reason = f"Detected: {label}. Faces must be uncovered to receive or provide public services in Quebec's education network (Bill 94)."
        color = "red"

    bg_color = '#fee2e2' if color == 'red' else '#dcfce7'
    border_color = '#ef4444' if color == 'red' else '#22c55e'
    text_color = '#991b1b' if color == 'red' else '#166534'

    result_html = f"""
    <div style="padding: 24px; border-radius: 12px; border: 3px solid {border_color}; background-color: {bg_color}; color: {text_color} !important; font-family: sans-serif; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); margin-bottom: 20px;">
        <h2 style="margin: 0 0 12px 0; font-size: 1.6em; font-weight: 800; display: flex; align-items: center; gap: 10px; color: {text_color} !important;">
            {status}
        </h2>
        <p style="margin: 0; font-size: 1.05em; line-height: 1.5; font-weight: 600; color: {text_color} !important;">
            {reason}
        </p>
    </div>
    """
    
    return result_html

# --- UI Layout ---

custom_css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.stat-box { background: #ffffff !important; padding: 24px; border-radius: 12px; text-align: center; border: 2px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); margin-bottom: 16px; }
.stat-box h2 { font-size: 2.5em !important; font-weight: 900 !important; color: #ef4444 !important; }
.stat-box p { color: #000000 !important; font-weight: 700 !important; font-size: 1.1em !important; }
.banner-img img { width: 100% !important; height: auto !important; max-height: 350px !important; object-fit: contain !important; border-radius: 12px !important; margin-bottom: 20px; }
.at-work-banner img { width: auto !important; height: auto !important; max-height: 80px !important; object-fit: contain !important; border-radius: 8px !important; margin: 0 auto 20px auto !important; display: block; }
.logo-img img { height: 50px !important; width: auto !important; object-fit: contain !important; margin-bottom: 15px !important; }
.cta-header { color: #ffffff !important; font-weight: 800 !important; text-align: center; font-size: 1.5em !important; margin-bottom: 10px !important; }
.cta-subtext { color: #cbd5e1 !important; text-align: center; margin-bottom: 20px; font-weight: 500; font-size: 0.9em !important; }
.gr-samples .gallery { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)) !important; gap: 10px !important; }
.gr-samples .gallery img { height: 80px !important; border-radius: 8px !important; }
.glossary-box { background: #f1f5f9; padding: 20px; border-radius: 8px; border-left: 5px solid #2563eb; margin-bottom: 15px; }
.glossary-box strong { color: #1e293b; }
"""

with gr.Blocks(title="Banned by 21") as demo:
    gr.Markdown("# Banned by 21: Quebec Employment Eligibility")
    
    with gr.Tabs() as tabs:
        # --- TAB 1: HOME ---
        with gr.Tab("Home", id="home"):
            with gr.Row():
                with gr.Column(scale=2):
                    if os.path.exists("assets/banned-by-21.png"):
                        gr.Image("assets/banned-by-21.png", show_label=False, container=False, elem_classes="banner-img")
                    
                    gr.Markdown("""
                    ## The Fight for Civil Liberties in Quebec
                    In 2019, Quebec passed **Bill 21**, followed by the expansion of **Bill 94** in 2025. These laws prohibit public sector employees—teachers, police, and school staff—from wearing religious symbols while exercising their functions.
                    
                    **"Banned by 21"** is an interactive tool designed to help you navigate these restrictions and understand the legal reality for thousands of workers in the province.
                    """)
                    start_btn = gr.Button("Enter Eligibility Checker", variant="primary")
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="stat-box">
                        <h2 style="margin: 0;">71%</h2>
                        <p>of Muslim women surveyed are considering leaving Quebec due to Bill 21.</p>
                    </div>
                    <div class="stat-box">
                        <h2 style="margin: 0;">88%</h2>
                        <p>felt Quebec has become a less welcoming place to live since 2019.</p>
                    </div>
                    """)

        # --- TAB 2: ELIGIBILITY CHECKER ---
        with gr.Tab("Eligibility Checker", id="checker"):
            with gr.Row():
                # LEFT COLUMN: INPUT & SAMPLES
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Photo Input")
                    image_input = gr.Image(label="Upload Photo or Use Webcam", type="pil", height=400, sources=["upload", "webcam"])
                    
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[image_input],
                        label="Try an Example Image",
                        examples_per_page=12
                    )
                
                # RIGHT COLUMN: CONFIG & RESULTS (ALWAYS VISIBLE CTA)
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Job & Eligibility")
                    job_dropdown = gr.Dropdown(label="Select Your Profession", choices=RESTRICTED_JOBS, value=RESTRICTED_JOBS[0])
                    submit_btn = gr.Button("Check My Status", variant="primary")
                    
                    status_output = gr.HTML(label="Eligibility Status")
                    
                    # CTA SECTION - PERMANENTLY VISIBLE
                    gr.Markdown("## Defend Civil Liberties: Take Action", elem_classes="cta-header")
                    gr.Markdown("The NCCM and CCLA have partnered to take the civil liberties battle to the Supreme Court of Canada.", elem_classes="cta-subtext")
                    
                    with gr.Row():
                        with gr.Column():
                            if os.path.exists("assets/nccm-logo.png"):
                                gr.Image("assets/nccm-logo.png", show_label=False, container=False, elem_classes="logo-img")
                            gr.Markdown(f"**NCCM**\n\n{DONATION_DATA[0]['description']}")
                            gr.Button("Donate to NCCM", link=DONATION_DATA[0]['actionButtonLink'])
                            
                        with gr.Column():
                            if os.path.exists("assets/CCLA-logo.png"):
                                gr.Image("assets/CCLA-logo.png", show_label=False, container=False, elem_classes="logo-img")
                            gr.Markdown(f"**CCLA**\n\n{DONATION_DATA[1]['description']}")
                            gr.Button("Donate to CCLA", link=DONATION_DATA[1]['actionButtonLink'])

        # --- TAB 3: HOW IT WORKS ---
        with gr.Tab("How it Works", id="how_it_works"):
            gr.Markdown("""
            ## How the Determination is Made
            Understanding the logic behind the "Banned by 21" eligibility checker.
            
            ### 1. The Visual Scan (AI Detection)
            The app uses **Image Recognition** technology (CLIP) to scan your photo for patterns associated with religious attire or face coverings. 
            *   **What we look for:** Symbols like the Hijab, Kippah, Turban, Crucifixes, and ceremonial items like the Kirpan.
            *   **Face Uncovering:** We also check if the face is covered by a niqab, burqa, or mask, which is restricted under Bill 94.
            
            ### 2. The Job Check
            The app compares your selected profession against the list of **"Covered Positions"** in the legislation. 
            *   **Bill 21** targets positions of authority (Teachers, Police, Lawyers).
            *   **Bill 94** expanded this to nearly **all school staff**, including daycare workers and support personnel.
            
            ### 3. The Final Verdict
            The logic is simple but strict:
            *   **INELIGIBLE ❌:** If a religious symbol is detected AND your job is on the restricted list.
            *   **INELIGIBLE ❌:** If your face is covered while working in the education network.
            *   **ELIGIBLE ✅:** If no religious symbols are detected, or if your job is not covered by current legislation.
            
            ### 🛡️ Your Privacy
            **No data is ever stored.** The processing happens in your browser's temporary memory. Once you close this tab or upload a new photo, the previous data is permanently deleted.
            """)

        # --- TAB 4: ABOUT THE LAWS ---
        with gr.Tab("About the Laws", id="laws"):
            if os.path.exists("assets/banned-by-21-at-work.png"):
                gr.Image("assets/banned-by-21-at-work.png", show_label=False, container=False, elem_classes="at-work-banner")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### The Legislation
                    **Bill 21 (2019):** *An Act respecting the laicity of the State*. Prohibits "visible religious symbols" for public sector workers in positions of authority.
                    
                    **Bill 94 (2025):** *An Act to reinforce laicity in the education network*. Expands the ban to all school staff and mandates uncovered faces on school premises.
                    
                    ### Legal Resources
                    *   [Bill 21: Official Text](https://www.legisquebec.gouv.qc.ca/en/document/cs/L-0.3)
                    *   [Bill 94: Official Text](https://www.assnat.qc.ca/en/travaux-parlementaires/projets-loi/projet-loi-94-43-1.html)
                    *   [Canadian Charter of Rights and Freedoms](https://www.canada.ca/en/canadian-heritage/services/how-rights-protected/guide-canadian-charter-rights-freedoms.html)
                    """)
                with gr.Column():
                    gr.Markdown("### Key Terms")
                    gr.HTML("""
                    <div class="glossary-box">
                        <strong>Laicity (Laïcité):</strong> The separation of State and religious affairs, ensuring State neutrality toward all religions.
                    </div>
                    <div class="glossary-box">
                        <strong>Notwithstanding Clause:</strong> Section 33 of the Charter, which allows a government to pass laws that bypass certain fundamental rights for up to 5 years.
                    </div>
                    <div class="glossary-box">
                        <strong>SCC Appeal:</strong> The Supreme Court of Canada granted leave in Jan 2025 to hear the constitutional challenge against Bill 21.
                    </div>
                    """)

    # Event Wiring
    start_btn.click(fn=lambda: gr.Tabs(selected="checker"), outputs=tabs)
    
    submit_btn.click(
        fn=get_eligibility,
        inputs=[image_input, job_dropdown],
        outputs=[status_output]
    )

    gr.Markdown("---")
    gr.Markdown("*Disclaimer: This tool is for informational and advocacy purposes. Images are processed locally in memory and not stored on any server.*")
    gr.Markdown("*Created by Bilal Shirazi (bilalshirazi.com)*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", allowed_paths=["assets", "data/test_images"], theme=gr.themes.Soft())
