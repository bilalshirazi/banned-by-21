# Banned by 21: Quebec Public Sector Employment Eligibility App

**Platform Phase 1:** Web Application
**Platform Phase 2:** iOS Application

## 1. Project Overview
An interactive application to help users understand if their attire affects their eligibility to work in specific public sector roles in Quebec. The app evaluates images and job titles against the stipulations of Bill 21 (An Act respecting the laicity of the State) and Bill 94 (An Act to, in particular, reinforce laicity in the education network). 

## 2. User Flow & Core Features
*   **Image Input:** Users upload or take a picture.
*   **Computer Vision Module:** Analyzes the image to detect religious symbols or face coverings.
*   **Job Selection Dropdown:** Users select their profession from a restricted jobs list.
*   **Result UI:** A Green Check (Eligible) or Red X (Ineligible) overlay. For example, personnel cannot exercise functions if they refuse to remove a hijab, facing the ultimatum to remove it or lose their job.
*   **Call to Action (CTA) Page:** A dedicated exit page directing users to donate to the National Council of Canadian Muslims (NCCM) and the Canadian Civil Liberties Association (CCLA).

## 3. Data Dictionary: Restricted Professions
The job selector must include roles strictly regulated by the legislation:
*   Principals, vice-principals, and teachers of public educational institutions.
*   School service centre personnel, including support staff, daycare workers, supervisors, and special education technicians.
*   Any person who provides services to students or regularly provides services on school premises.
*   Peace officers exercising functions mainly in Quebec.
*   Administrative justices of the peace, clerks, sheriffs, and bankruptcy registrars.
*   Government-employed lawyers, notaries, and criminal/penal prosecuting attorneys.
*   Members or commissioners of provincial boards and tribunals (e.g., Human Rights Tribunal, Administrative Labour Tribunal).
*   President and Vice-Presidents of the National Assembly.

## 4. Logic Engine Rules
*   **Rule 1 (Religious Symbols):** If `religious_symbol_detected == true` AND `job_selected` is in the Restricted List, THEN `status = Red X`.
*   **Rule 2 (Face Coverings):** If `face_covered == true` AND no health, handicap, or task-related exemption applies, THEN `status = Red X`.

## 5. Implementation Details: Hugging Face Space (Gradio)
The Phase 1 Web Application is implemented as a **Hugging Face Space** using the **Gradio** SDK.

### A. Technical Stack
- **Framework:** Gradio (`gradio`)
- **Computer Vision:** `openai/clip-vit-base-patch32` for zero-shot image classification.
- **Logic:** Python-based logic engine implementing Bill 21 and Bill 94 rules.

### B. Core Implementation Files
- `app.py`: Main application logic and UI.
- `requirements.txt`: Dependencies including `gradio`, `transformers`, `torch`.
- `README.md`: Metadata and description for the Hugging Face Space.

### C. Detection Logic (CLIP Labels)
The zero-shot detection uses the following prompts:
- "a person wearing a hijab"
- "a person wearing a kippah"
- "a person wearing a turban"
- "a person wearing a niqab or burqa (face covered)"
- "a person with their face covered by a cloth or mask"
- "a person not wearing any religious symbols"

### D. Donation Cards (JSON Data)
Map over the following JSON object to generate `<DonationCard />` components.

```json
[
  {
    "organization": "National Council of Canadian Muslims (NCCM)",
    "description": "The NCCM is challenging Bill 21 to defend the rights of public sector employees who are banned from wearing religious symbols, and is providing legal aid and advocacy against Islamophobia.",
    "address": "PO Box 77062, Markham ON, L3P 0C8",
    "email": "info@nccm.ca",
    "actionButtonText": "Donate to NCCM",
    "actionButtonLink": "https://www.nccm.ca/donate/"
  },
  {
    "organization": "Canadian Civil Liberties Association (CCLA)",
    "description": "The CCLA acts as a vigilant watchdog for rights and freedoms in Canada. Donations ensure they can continue their work challenging Bill 21 at the Supreme Court.",
    "address": "124 Merton St, Suite #400, Toronto, ON M4S 2Z2",
    "email": "donations@ccla.org",
    "phone": "(416) 646-1407",
    "charityNumber": "Charitable Registration: 75480 2288 RR0001",
    "actionButtonText": "Donate to CCLA",
    "actionButtonLink": "https://ccla.org/donate/"
  }
]
```

### E. CSS / Styling Directives
*   Use a responsive CSS Grid: `grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));`
*   Include standard `<a href="mailto:...">` tags for the contact emails.
