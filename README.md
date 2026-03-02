---
title: Banned By 21
emoji: ⚖️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: false
license: mit
---

# Banned by 21: Quebec Employment Eligibility

An interactive application and advocacy platform to help users understand if their attire affects their eligibility to work in specific public sector roles in Quebec under **Bill 21** and **Bill 94**.

## 🚀 Features
- **Multi-Tab Interface:** Educational landing page, interactive eligibility checker, and legislative guide.
- **AI-Powered Detection:** Uses CLIP (Zero-shot Classification) to identify religious symbols and face coverings.
- **Take Action:** Direct integration with NCCM and CCLA donation platforms to support the legal battle at the Supreme Court.

## 📁 Project Structure
- `app.py`: The main Gradio application logic.
- `docs/`: Technical specifications, legislative knowledge base, and research.
- `data/`: Local test image collection for model calibration.
- `requirements.txt`: Python dependencies.

## ⚖️ Legal Disclaimer
This tool is for advocacy and informational purposes only. It reflects a hardline interpretation of current Quebec legislation to highlight the risks faced by public sector employees. No images are stored or saved to any server.

## 🛠 Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Prerequisites
- Python 3.10+
- `uv` installed on your system.

### Setup & Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd banned-by-21
    ```

2.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

### Running the Application
To launch the Gradio app locally using `uv`:
```bash
uv run app.py
```
The application will be available at `http://127.0.0.1:7860`.

## 📸 Test Data
You can test the application using the sample images located in `data/test_images/`. These images were generated to calibrate the CLIP model for diverse religious symbols and face coverings.

---
*Created by Bilal Shirazi (bilalshirazi.com)*
