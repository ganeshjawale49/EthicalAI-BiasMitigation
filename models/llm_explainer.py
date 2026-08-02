"""
LLM Bias Explanation & Audit Module.
Integrates Google Gemini / OpenAI APIs with a local intelligent fallback engine 
to produce human-understandable bias analysis, root-cause diagnostics, and actionable recommendations.
"""
import os
import json
import requests
from config import Config
from models.auth import get_db_connection

def generate_local_bias_explanation(model_name, sensitive_col, priv_group, unpriv_group, perf_metrics, fairness_metrics):
    """
    Generates a high-quality, structured plain-English audit report locally 
    based on mathematical fairness metrics when an LLM API key is unavailable.
    """
    di = fairness_metrics['disparate_impact']
    dpd = fairness_metrics['demographic_parity_diff']
    eod = fairness_metrics['equalized_odds_diff']
    eod_opp = fairness_metrics['equal_opportunity_diff']
    acc = perf_metrics['accuracy']
    
    g_metrics = fairness_metrics['group_metrics']
    priv_sr = g_metrics['priv_selection_rate']
    unpriv_sr = g_metrics['unpriv_selection_rate']
    
    is_biased = di < 0.8 or dpd > 0.15
    
    summary = (
        f"The audit of the {model_name} model reveals significant algorithmic disparity against the unprivileged group ('{unpriv_group}') "
        f"when compared to the privileged group ('{priv_group}'). The Disparate Impact ratio is {di:.2f}, which falls below the ethical "
        f"threshold of 0.80 (the EEOC 80% Rule standard)." if is_biased else
        f"The {model_name} model demonstrates satisfactory algorithmic fairness across demographic subgroups ('{priv_group}' vs '{unpriv_group}'). "
        f"The Disparate Impact ratio is {di:.2f}, indicating minimal bias under standard regulatory frameworks."
    )
    
    explanation = (
        f"1. **Demographic Selection Rate Disparity**: The model grants positive predictions (e.g. loan approval) to **{priv_sr * 100:.1f}%** "
        f"of '{priv_group}' applicants, compared to only **{unpriv_sr * 100:.1f}%** of '{unpriv_group}' applicants.\n"
        f"2. **Demographic Parity Difference**: A selection gap of **{dpd * 100:.1f}%** exists between the two demographic subgroups.\n"
        f"3. **Equalized Odds & Opportunity**: The True Positive Rate difference is **{eod_opp * 100:.1f}%**, indicating that qualified members "
        f"of the unprivileged group ('{unpriv_group}') face lower odds of receiving favorable outcomes even when equally qualified.\n"
        f"4. **Model Performance Context**: Overall accuracy stands at **{acc * 100:.1f}%**, confirming that high accuracy alone does not guarantee algorithmic fairness."
    )
    
    root_cause = (
        f"The primary root cause of this bias is historical representation imbalance and systemic proxy variables in the training data. "
        f"The sensitive attribute '{sensitive_col}' correlates with target outcomes due to historical biases present in the dataset. "
        f"Non-sensitive features (such as income or credit score) may also act as statistical proxies, allowing the classifier to infer demographic "
        f"groups indirectly."
    )
    
    recommendations = [
        f"Apply **Reweighing** pre-processing to adjust sample weights during training, balancing subgroup representation without discarding data.",
        f"Calibrate subgroup decision thresholds via **Post-processing Threshold Optimization** (e.g., setting unprivileged threshold lower to equalize approval rates).",
        f"Perform **Feature Neutralization** by removing or regularizing proxy features that heavily correlate with '{sensitive_col}'.",
        f"Establish continuous auditing and monitor Disparate Impact across future model deployments."
    ]
    
    return {
        'summary': summary,
        'explanation': explanation,
        'root_cause': root_cause,
        'recommendations': recommendations,
        'source': 'Local Intelligent Audit Engine'
    }

def call_gemini_api(api_key, prompt):
    """Calls Google Gemini API via REST endpoint."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        if response.status_code == 200:
            res_data = response.json()
            text = res_data['candidates'][0]['content']['parts'][0]['text']
            return text
    except Exception as e:
        print(f"Gemini API Call Error: {e}")
    return None

def analyze_bias_with_llm(model_name, sensitive_col, priv_group, unpriv_group, perf_metrics, fairness_metrics, api_key_override=None):
    """
    Main entry point for generating LLM Bias Audit.
    Attempts API call to Gemini/OpenAI if key is present; otherwise uses local engine.
    """
    api_key = api_key_override or Config.GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY', '')
    
    if api_key:
        prompt = f"""
You are an expert AI Ethics and Machine Learning Audit System. Analyze the following model fairness evaluation metrics and provide a structured bias explanation report in simple English.

Model Name: {model_name}
Sensitive Attribute: {sensitive_col}
Privileged Group: {priv_group}
Unprivileged Group: {unpriv_group}
Accuracy: {perf_metrics['accuracy']}
Disparate Impact: {fairness_metrics['disparate_impact']} (80% rule threshold = 0.80)
Demographic Parity Difference: {fairness_metrics['demographic_parity_diff']}
Equalized Odds Difference: {fairness_metrics['equalized_odds_diff']}
Group Selection Rates: Privileged={fairness_metrics['group_metrics']['priv_selection_rate']}, Unprivileged={fairness_metrics['group_metrics']['unpriv_selection_rate']}

Please return your response as a valid JSON object with the following keys:
- "summary": A 2-sentence executive summary of the bias findings.
- "explanation": A 4-bullet point explanation in simple English describing why predictions are biased or fair.
- "root_cause": An explanation of potential data imbalances or proxy variables causing bias.
- "recommendations": A list of 4 concrete actionable steps to mitigate this bias.
        """
        raw_response = call_gemini_api(api_key, prompt)
        if raw_response:
            try:
                # Clean markdown json code blocks if present
                clean_json = raw_response.strip().replace('```json', '').replace('```', '').strip()
                parsed = json.loads(clean_json)
                parsed['source'] = 'Google Gemini LLM'
                return parsed
            except Exception:
                pass
                
    # Fallback to local intelligent engine
    return generate_local_bias_explanation(
        model_name, sensitive_col, priv_group, unpriv_group, perf_metrics, fairness_metrics
    )

def save_llm_audit(model_run_id, audit_result):
    """Saves LLM audit result into SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO llm_audits (model_run_id, summary_text, explanation_text, root_cause, recommendations_json)
           VALUES (?, ?, ?, ?, ?)""",
        (
            model_run_id,
            audit_result['summary'],
            audit_result['explanation'],
            audit_result['root_cause'],
            json.dumps(audit_result['recommendations'])
        )
    )
    conn.commit()
    conn.close()
    return True
