# Step-by-Step Installation & Setup Manual

## Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12 installed.
- **Git** or File Explorer.
- Web Browser (Google Chrome, Microsoft Edge, or Firefox).

---

## Step 1: Clone or Navigate to Workspace
Open PowerShell or Terminal and navigate to the project directory:
```bash
cd c:\project\EthicalAI-BiasMitigation
```

---

## Step 2: Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate
# On Linux/macOS:
source venv/bin/activate
```

---

## Step 3: Install Required Dependencies
Install all required libraries specified in `requirements.txt`:
```bash
python -m pip install -r requirements.txt
```

---

## Step 4: Configure Environment Variables (Optional)
If you wish to use your live Google Gemini API key for real-time LLM generation:
```powershell
# Windows PowerShell:
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# Linux / macOS:
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```
*Note: If no API key is set, the system seamlessly uses its built-in Intelligent Local Audit Engine so the live demonstration will never fail.*

---

## Step 5: Launch the Application
Run the Flask application:
```bash
python app.py
```

Expected Output:
```
Starting Ethical AI Bias Mitigation Web Server...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

## Step 6: Access Web Interface
Open your web browser and navigate to:
`http://127.0.0.1:5000`
