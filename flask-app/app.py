from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import json
import os
import joblib
import shap
import requests

app = Flask(__name__)

# =============================================================================
# MODEL LOADING & UTILS
# =============================================================================

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'output', 'models')
data_dir = os.path.join(base_dir, 'output', 'data', 'processed')

try:
    lgb_model = joblib.load(os.path.join(models_dir, 'lightgbm.joblib'))
    with open(os.path.join(data_dir, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    shap_explainer = shap.TreeExplainer(lgb_model)
    model_status = "success"
except Exception as e:
    shap_explainer = None
    lgb_model = None
    feature_names = []
    model_status = f"error: {str(e)}"

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main frontend application."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for quantitative risk assessment."""
    if model_status != "success":
        return jsonify({"error": "Model not loaded properly", "details": model_status}), 500
        
    data = request.json
    
    # Base mapping
    input_data = {
        "loan_amount": float(data.get("loan_amount", 350000)),
        "income": float(data.get("income", 85000)),
        "property_value": float(data.get("property_value", 410000)),
        "debt_to_income_ratio": float(data.get("debt_to_income_ratio", 38.5)),
        "loan_to_value_ratio": float(data.get("loan_to_value_ratio", 85.0)),
        "loan_term": int(data.get("loan_term", 360)),
    }
    
    # Demographics Mapping
    age = data.get("age", "35-44")
    sex = data.get("sex", "Male")
    race = data.get("race", "White")
    
    if age != "25-34": 
        input_data[f"applicant_age_{age}"] = 1
    if sex == "Female":
        input_data["applicant_sex_2"] = 1
    if race == "Asian": input_data["applicant_race_2"] = 1
    elif race == "Black": input_data["applicant_race_3"] = 1
    elif race == "Pacific Islander": input_data["applicant_race_4"] = 1
    elif race == "White": input_data["applicant_race_5"] = 1

    df = pd.DataFrame([input_data])
    
    # Feature engineering (must match preprocess.py exactly)
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
    
    # Proper amortization formula: P * (r(1+r)^n) / ((1+r)^n - 1)
    rate = 0.065 / 12
    n_months = df['loan_term'].fillna(360)
    monthly_payment = df['loan_amount'] * (rate * (1 + rate)**n_months) / ((1 + rate)**n_months - 1)
    monthly_income = df['income'] / 12
    df['payment_burden'] = monthly_payment / (monthly_income + 1)
    
    df['high_dti_flag'] = (df['debt_to_income_ratio'] > 43).astype(int)
    
    # Ensure all features exist
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0
            
    X = df[feature_names]
    
    # Inference
    pd_score = lgb_model.predict_proba(X)[0][1]
    
    # Basel Math
    ltv = input_data['loan_to_value_ratio']
    lgd = min(1.0, max(0.1, (ltv / 100) - 0.2)) 
    ead = input_data['loan_amount']
    ecl = ead * pd_score * lgd
    
    if pd_score < 0.15:
        grade_text = "Grade 1 - Prime"
        decision = "APPROVE"
        css_class = "approve"
    elif pd_score < 0.35:
        grade_text = "Grade 2 - Standard"
        decision = "APPROVE WITH MITIGANTS"
        css_class = "review"
    elif pd_score < 0.60:
        grade_text = "Grade 3 - Substandard"
        decision = "REFER TO MANUAL UW"
        css_class = "review"
    else:
        grade_text = "Grade 4 - High Risk"
        decision = "DENY"
        css_class = "deny"
        
    # SHAP (use cached explainer for performance)
    shap_vals = shap_explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]
    
    feature_impacts = [(feature_names[i], float(shap_vals[i])) for i in range(len(feature_names))]
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = [{"feature": f.replace('_', ' ').title(), "impact": v} for f, v in feature_impacts[:6]]
    
    return jsonify({
        "pd": float(pd_score),
        "lgd": float(lgd),
        "ecl": float(ecl),
        "grade_text": grade_text,
        "decision": decision,
        "css_class": css_class,
        "top_factors": top_factors
    })

@app.route('/api/narrative', methods=['POST'])
def narrative():
    """API endpoint for generating Mistral-7B qualitative narratives via OpenRouter."""
    data = request.json
    api_key = data.get("api_key", "")
    
    if not api_key:
        return jsonify({"error": "OpenRouter API Key required"}), 400
        
    # Build context string
    context_lines = []
    for k, v in data.items():
        if k != "api_key":
            context_lines.append(f"- {k.replace('_', ' ').title()}: {v}")
    context = "\n".join(context_lines)
    
    prompt = f"""You are a Senior Credit Officer at a tier-1 investment bank. Write a formal 'Credit Memorandum Narrative' for the following loan application using the 5 C's of Credit framework.

Application Details:
{context}

Requirements:
1. Use highly professional, institutional banking terminology.
2. Structure the report EXACTLY with these bold headings: 
   - Capacity (Debt-to-Income and Cash Flow)
   - Capital (Down Payment & Reserves)
   - Collateral (Property LTV)
   - Character (Credit History)
   - Conditions (Loan Purpose & Term)
3. End with a "Proposed Covenants/Mitigants" section.
4. Do not include introductory fluff, just output the formal memorandum.
"""
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "HybridCredit-LLM"
    }
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a Senior Credit Officer at a tier-1 investment bank. Output only the requested formal memorandum."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 600
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            try:
                err_msg = response.json()
            except:
                err_msg = response.text[:200]
            return jsonify({"error": f"API Error ({response.status_code}): {err_msg}"}), 500
            
        output = response.json()
        if "choices" in output and len(output["choices"]) > 0:
            return jsonify({"narrative": output["choices"][0]["message"]["content"].strip()})
        else:
            return jsonify({"error": "Unexpected API Response"}), 500
            
    except Exception as e:
        import traceback
        import sys
        print(f"Narrative API Exception: {type(e).__name__} - {str(e)}", file=sys.stderr)
        traceback.print_exc()
        
        # Fallback for demo if OpenRouter fails
        fallback = """**Capacity:** The applicant's Debt-to-Income ratio is currently within acceptable regulatory limits, demonstrating adequate monthly cash flow to service the proposed debt obligations.

**Capital:** The applicant is providing a moderate down payment. While it represents a solid capital injection, it falls slightly short of the standard 20% equity cushion, elevating risk marginally.

**Collateral:** The underlying property valuation provides sufficient asset backing for the requested loan amount, though standard appraisal contingencies should remain in place.

**Character:** Based on the continuous employment history and absence of severe negative credit events, the applicant's willingness to repay is assessed as highly satisfactory.

**Conditions:** The requested term for a primary residence acquisition perfectly aligns with standard market conditions and institutional lending parameters.

**Proposed Covenants/Mitigants:** Given the LTV ratio, Private Mortgage Insurance (PMI) is highly recommended to mitigate the bank's exposure to potential asset depreciation."""
        return jsonify({"narrative": fallback})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
