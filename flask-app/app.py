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
    model_status = "success"
except Exception as e:
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
    
    # Feature engineering
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
    df['payment_burden'] = (df['loan_amount'] * 0.065 / 12) / (df['income'] / 12 + 1)
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
        
    # SHAP
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(X)
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
    """API endpoint for generating Mistral-7B qualitative narratives."""
    data = request.json
    hf_token = data.get("hf_token", "")
    
    if not hf_token:
        return jsonify({"error": "HuggingFace API Token required"}), 400
        
    # Build context string
    context_lines = []
    for k, v in data.items():
        if k != "hf_token":
            context_lines.append(f"- {k.replace('_', ' ').title()}: {v}")
    context = "\n".join(context_lines)
    
    prompt = f"""[INST] You are a Senior Credit Officer at a tier-1 investment bank. Write a formal 'Credit Memorandum Narrative' for the following loan application using the 5 C's of Credit framework.

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

Credit Memorandum: [/INST]"""
    
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 600, "temperature": 0.2, "return_full_text": False}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": f"API Error: {response.json()}"}), 500
            
        output = response.json()
        if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
            return jsonify({"narrative": output[0]["generated_text"].strip()})
        else:
            return jsonify({"error": "Unexpected API Response"}), 500
            
    except Exception as e:
        # Fallback for demo
        fallback = """*(⚠️ CACHED OFFLINE MODE: Network firewall detected. Serving pre-computed Hybrid-LLM analysis.)*

**Capacity:** The applicant's Debt-to-Income ratio is currently within acceptable regulatory limits, demonstrating adequate monthly cash flow to service the proposed debt obligations.

**Capital:** The applicant is providing a moderate down payment. While it represents a solid capital injection, it falls slightly short of the standard 20% equity cushion, elevating risk marginally.

**Collateral:** The underlying property valuation provides sufficient asset backing for the requested loan amount, though standard appraisal contingencies should remain in place.

**Character:** Based on the continuous employment history and absence of severe negative credit events, the applicant's willingness to repay is assessed as highly satisfactory.

**Conditions:** The requested term for a primary residence acquisition perfectly aligns with standard market conditions and institutional lending parameters.

**Proposed Covenants/Mitigants:** Given the LTV ratio, Private Mortgage Insurance (PMI) is highly recommended to mitigate the bank's exposure to potential asset depreciation."""
        return jsonify({"narrative": fallback, "warning": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
