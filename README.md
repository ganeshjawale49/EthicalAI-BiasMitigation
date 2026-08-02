# Mitigation of Bias and Improve Fairness in Machine Learning using Large Language Models Towards Ethical AI Systems

A complete, production-ready, full-stack web application developed for final-year engineering demonstration. The system enables automated algorithmic bias detection, fairness metric calculation (Disparate Impact, Demographic Parity, Equalized Odds), Gemini LLM plain-English explainability, and Machine Learning bias mitigation (Reweighing & Threshold Calibration).

---

## Key Features

- **User Authentication**: Secure signup and login with hashed password credentials stored in SQLite.
- **Dataset Upload & Schema Profiling**: Upload custom CSV datasets or load the built-in 1,000-row benchmark credit scoring dataset.
- **Automated Preprocessing**: Label encoding, missing value median imputation, one-hot encoding, feature scaling, and train/test splits.
- **Classification Engine**: Multi-algorithm support (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting).
- **Fairness Metrics Suite**:
  - Disparate Impact (DI) Ratio (&ge; 0.80 EEOC rule standard)
  - Demographic Parity Difference (DPD)
  - Equalized Odds Difference (EOD)
  - Equal Opportunity Difference (EOD_opp)
  - Confusion Matrix Visual Grid
- **Gemini LLM Explainability**: Automatically generates human-understandable plain English audit summaries, root cause analysis, and mitigation recommendations.
- **Bias Mitigation Engine**:
  - Pre-processing Reweighing (Kamiran & Calders sample weighting)
  - Post-processing Subgroup Threshold Calibration
- **Interactive Visual Dashboard**: Real-time Chart.js charts for selection rate disparity and before-vs-after fairness improvement.
- **Project Deliverables**: Downloadable PDF/HTML reports and a 15-Slide PowerPoint (`.pptx`) presentation deck.

---

## Tech Stack

- **Backend**: Python 3.12, Flask, SQLite3
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Fairlearn
- **LLM Integration**: Google Gemini API REST Integration / OpenAI API (with intelligent local audit engine fallback)
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (ES6+), Bootstrap 5, FontAwesome 6, Chart.js
- **Presentation Deck**: Python-PPTX

---

## Directory Structure

```
EthicalAI-BiasMitigation/
├── app.py                      # Main Flask application entry point
├── config.py                   # System configuration & API keys
├── requirements.txt            # Dependency specification
├── schema.sql                  # Database schema script
├── database.db                 # SQLite database file (auto-generated)
├── README.md                   # Project overview & instructions
│
├── models/                     # Core Business & ML Logic
│   ├── auth.py                 # User authentication & database operations
│   ├── dataset_manager.py      # CSV dataset management & preprocessing
│   ├── ml_engine.py            # Model training & standard metrics calculation
│   ├── bias_detector.py        # Fairness metrics (Disparate Impact, Demographic Parity)
│   ├── bias_mitigator.py       # Pre-processing Reweighing & Post-processing Threshold Calibration
│   ├── llm_explainer.py        # Gemini LLM audit generator & intelligent fallback
│   └── ppt_generator.py        # 15-slide PowerPoint deck generator (.pptx)
│
├── static/                     # Web static assets
│   ├── css/
│   │   └── style.css           # Glassmorphism & Bootstrap extension styling
│   ├── js/
│   │   ├── main.js             # Client UI handlers
│   │   └── charts.js           # Chart.js visualization logic
│   └── uploads/                # File uploads & benchmark CSV datasets
│       ├── sample_credit_bias.csv
│       └── Ethical_AI_Bias_Mitigation_Presentation.pptx
│
├── templates/                  # Jinja2 HTML View Templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── preprocess.html
│   ├── train.html
│   ├── bias_detection.html
│   ├── llm_audit.html
│   ├── mitigation.html
│   ├── evaluation.html
│   └── reports.html
│
└── docs/                       # Project Documentation Deliverables
    ├── FINAL_PROJECT_REPORT.md
    ├── PROJECT_DOCUMENTATION.md
    ├── PPT_PRESENTATION_OUTLINE.md
    ├── FUTURE_SCOPE.md
    └── INSTALLATION_GUIDE.md
```

---

## Quick Start & Installation

1. **Clone or Extract Workspace**:
   Navigate to the project root directory:
   ```bash
   cd EthicalAI-BiasMitigation
   ```

2. **Install Dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Run Flask Server**:
   ```bash
   python app.py
   ```

4. **Access Web Application**:
   Open browser at: `http://127.0.0.1:5000`

---

## Final-Year Engineering Demonstration Steps

1. **Login**: Register a user or login with `admin` / `admin123`.
2. **Upload / Load Dataset**: Go to Datasets and click **Load Benchmark Credit Dataset**.
3. **Configure Sensitive Attribute**: Select `Gender` as sensitive attribute, `Male` as Privileged, `Female` as Unprivileged.
4. **Train Model**: Choose `Random Forest Classifier` and click **Train & Evaluate**.
5. **Inspect Bias Metrics**: Notice Disparate Impact is ~`0.76` (Flagged as **BIASED** under EEOC 80% Rule).
6. **Trigger LLM Audit**: Click **Explain via Gemini LLM** to view plain English diagnostic explanation.
7. **Apply Mitigation**: Click **Proceed to Bias Mitigation**, choose **Post-Processing Threshold Calibration**, and click **Apply Mitigation**.
8. **Verify Fairness Improvement**: Observe Disparate Impact improves to ~`0.91` (**FAIR** state) with < 1% impact on accuracy!
9. **Download Deliverables**: Navigate to **Download Reports & PPT** to download the 15-Slide Presentation (`.pptx`) and print the audit report.
