"""
===============================================================================
KAGGLE NOTEBOOK 7: Explainability (XAI) & Fairness Audit
===============================================================================
Run this SEVENTH (LAST) on Kaggle (CPU only - no GPU needed)

What this notebook does:
1. SHAP analysis on LightGBM model
2. Faithfulness audit (LLM rationales vs SHAP)
3. Counterfactual explanations (DiCE-ML)
4. Fairness audit across demographic groups (ECOA compliance)
5. Generates all XAI figures for the paper
6. Creates compliance summary

PREREQUISITE: Notebooks 4, 5, 6
KAGGLE SETUP: Accelerator: None, Runtime: ~20-30 minutes
===============================================================================
"""

# =============================================================================
# CELL 1: Imports & Setup
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300})

WORKING_DIR = Path("./output")
PROCESSED_DIR = WORKING_DIR / "data" / "processed"
FIGURES_DIR = WORKING_DIR / "figures"
RESULTS_DIR = WORKING_DIR / "results"
MODELS_DIR = WORKING_DIR / "models"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KAGGLE NOTEBOOK 7: Explainability (XAI) & Fairness Audit")
print("=" * 70)

# =============================================================================
# CELL 2: Load Data & Models
# =============================================================================

print("\nLoading data and models...")

# Load processed data
X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

with open(PROCESSED_DIR / "feature_names.json", 'r') as f:
    feature_names = json.load(f)

# Load LightGBM model
import joblib
lgb_model = joblib.load(MODELS_DIR / "lightgbm.joblib")

# Load test dataframe with demographics
test_df = pd.read_csv(PROCESSED_DIR / "test_df.csv")

# Load LLM rationales
try:
    with open(PROCESSED_DIR / "llm_rationales.json", 'r') as f:
        llm_rationales = json.load(f)
except:
    llm_rationales = []

print(f"Test set: {X_test.shape}")
print(f"Features: {len(feature_names)}")
print(f"LLM rationales: {len(llm_rationales)}")

# =============================================================================
# CELL 3: SHAP Analysis
# =============================================================================

print("\n" + "=" * 70)
print("LAYER 1: SHAP FEATURE ATTRIBUTION")
print("=" * 70)

# Compute SHAP values
print("Computing SHAP values (sample of 3000)...")
explainer = shap.TreeExplainer(lgb_model)
shap_sample_size = min(3000, len(X_test))
shap_values = explainer.shap_values(X_test[:shap_sample_size])

if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Positive class

print(f"SHAP values shape: {shap_values.shape}")

# SHAP Summary Plot (beeswarm)
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test[:shap_sample_size],
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title('SHAP Feature Importance - Credit Default Prediction', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_shap_summary.pdf", bbox_inches='tight')
plt.show()

# SHAP Bar Plot (mean absolute)
print("\nGenerating SHAP bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test[:shap_sample_size],
    feature_names=feature_names,
    plot_type='bar',
    max_display=15,
    show=False
)
plt.title('Mean |SHAP| Feature Importance', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig9_shap_bar.pdf", bbox_inches='tight')
plt.show()

# Get SHAP feature ranking
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_ranking = np.argsort(-mean_abs_shap)
top_shap_features = [feature_names[i] for i in shap_ranking[:10]]
print(f"\nTop 10 SHAP features: {top_shap_features}")

# =============================================================================
# CELL 4: Faithfulness Audit
# =============================================================================

print("\n" + "=" * 70)
print("LAYER 2: FAITHFULNESS AUDIT (LLM vs SHAP)")
print("=" * 70)

# Feature keyword mapping for parsing LLM rationales
feature_keywords = {
    'loan_to_income_ratio': ['loan-to-income', 'loan to income', 'lti'],
    'debt_to_income_ratio': ['debt-to-income', 'dti', 'debt ratio'],
    'loan_amount': ['loan amount', 'amount'],
    'income': ['income', 'salary', 'earnings'],
    'loan_to_value_ratio': ['loan-to-value', 'ltv', 'value ratio'],
    'property_value': ['property value', 'home value', 'collateral'],
    'loan_term': ['term', 'duration', 'years'],
    'high_dti_flag': ['high dti', 'dti flag'],
    'safe_ltv_flag': ['safe ltv', 'ltv flag'],
    'payment_burden': ['payment burden', 'monthly payment'],
    'dti_ltv_interaction': ['dti ltv', 'interaction', 'combined risk'],
}

faithfulness_scores = []
n_audit = min(200, len(llm_rationales), len(shap_values))

print(f"Auditing {n_audit} samples...")

for idx in range(n_audit):
    rationale = str(llm_rationales[idx]).lower()
    
    # Parse mentioned features
    llm_mentioned = []
    for feat, keywords in feature_keywords.items():
        if feat in feature_names:
            if any(kw in rationale for kw in keywords):
                llm_mentioned.append(feat)
    
    # SHAP top-5 for this sample
    sample_shap = shap_values[idx]
    sample_shap_rank = [feature_names[i] for i in np.argsort(-np.abs(sample_shap))]
    top_5_shap = set(sample_shap_rank[:5])
    
    # Overlap score
    overlap = len(set(llm_mentioned) & top_5_shap)
    faithfulness_scores.append(overlap / 5.0)

mean_faithfulness = np.mean(faithfulness_scores)

print(f"\nMean Faithfulness Score: {mean_faithfulness:.3f}")
print(f"Interpretation:", end=" ")
if mean_faithfulness >= 0.7:
    print("High alignment - LLM rationales strongly agree with SHAP")
elif mean_faithfulness >= 0.5:
    print("Moderate alignment - LLM captures main factors but misses nuances")
elif mean_faithfulness >= 0.3:
    print("Low alignment - LLM explanations partially disconnected from SHAP")
else:
    print("Poor alignment - LLM rationales poorly reflect actual model behavior")

# Distribution plot
plt.figure(figsize=(10, 6))
plt.hist(faithfulness_scores, bins=20, color='#5B9BD5', edgecolor='black', alpha=0.7)
plt.axvline(mean_faithfulness, color='red', ls='--', lw=2, label=f'Mean: {mean_faithfulness:.3f}')
plt.axvline(0.5, color='orange', ls=':', lw=2, label='Moderate (0.5)')
plt.axvline(0.7, color='green', ls=':', lw=2, label='High (0.7)')
plt.xlabel('Faithfulness Score (Overlap / 5)')
plt.ylabel('Number of Samples')
plt.title('Distribution of SHAP-LLM Faithfulness Scores')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig10_faithfulness_dist.pdf", bbox_inches='tight')
plt.show()

# Save faithfulness results
faith_results = {
    'mean_faithfulness': round(float(mean_faithfulness), 4),
    'n_samples': n_audit,
    'distribution': {
        'mean': float(np.mean(faithfulness_scores)),
        'std': float(np.std(faithfulness_scores)),
        'median': float(np.median(faithfulness_scores)),
        'q25': float(np.percentile(faithfulness_scores, 25)),
        'q75': float(np.percentile(faithfulness_scores, 75)),
    }
}

with open(RESULTS_DIR / "faithfulness_audit.json", 'w') as f:
    json.dump(faith_results, f, indent=2)

# =============================================================================
# CELL 5: Counterfactual Explanations
# =============================================================================

print("\n" + "=" * 70)
print("LAYER 3: COUNTERFACTUAL EXPLANATIONS")
print("=" * 70)

try:
    import dice_ml
    
    # Prepare data for DiCE
    # Create a training dataframe with features and target
    train_sample_size = min(5000, len(X_test))
    
    # Use test data as proxy (in practice, use training data)
    X_test_df = pd.DataFrame(X_test[:train_sample_size], columns=feature_names)
    X_test_df['default'] = y_test[:train_sample_size]
    
    # Identify continuous features
    continuous_features = []
    for col in feature_names:
        if X_test_df[col].nunique() > 10:
            continuous_features.append(col)
    
    # Setup DiCE
    d = dice_ml.Data(
        dataframe=X_test_df,
        continuous_features=continuous_features[:10],  # Limit for speed
        outcome_name='default'
    )
    
    m = dice_ml.Model(model=lgb_model, backend='sklearn')
    exp = dice_ml.Dice(d, m, method='random')
    
    # Generate counterfactual for one denied applicant
    denied_idx = np.where(y_test == 1)[0][0]
    query = X_test_df.iloc[denied_idx:denied_idx+1].drop(columns=['default'])
    
    print(f"\nGenerating counterfactual for denied applicant (index {denied_idx})...")
    
    cf = exp.generate_counterfactuals(
        query,
        total_CFs=3,
        desired_class=0,  # Want approval (class 0)
        verbose=False
    )
    
    # Visualize
    cf.visualize_as_dataframe(show_only_changes=True)
    plt.savefig(FIGURES_DIR / "fig11_counterfactual.pdf", bbox_inches='tight')
    plt.show()
    
    print("\nCounterfactual explanation generated!")
    print("Shows: 'What would need to change for approval?'")
    
except Exception as e:
    print(f"\nCounterfactual generation skipped: {e}")
    print("(Install dice-ml for counterfactual explanations)")

# =============================================================================
# CELL 6: Fairness Audit
# =============================================================================

print("\n" + "=" * 70)
print("FAIRNESS AUDIT (ECOA COMPLIANCE)")
print("=" * 70)

# Get predictions
y_proba = lgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Protected attributes to audit
protected_attrs = ['applicant_sex', 'applicant_ethnicity', 'applicant_age']

all_fairness_results = {}

for attr in protected_attrs:
    if attr not in test_df.columns:
        print(f"  Skipping {attr} - not in data")
        continue
    
    print(f"\n--- Auditing: {attr} ---")
    
    groups = test_df[attr].dropna().unique()
    group_metrics = []
    
    for group in groups:
        mask = test_df[attr] == group
        n = mask.sum()
        
        if n < 30:
            continue
        
        y_g = y_test[mask]
        pred_g = y_pred[mask]
        proba_g = y_proba[mask]
        
        approval_rate = 1 - pred_g.mean()  # Approval = not denied
        
        tp = ((y_g == 1) & (pred_g == 1)).sum()
        fp = ((y_g == 0) & (pred_g == 1)).sum()
        tn = ((y_g == 0) & (pred_g == 0)).sum()
        fn = ((y_g == 1) & (pred_g == 0)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        group_metrics.append({
            'group': str(group),
            'n': int(n),
            'approval_rate': round(float(approval_rate), 4),
            'tpr': round(float(tpr), 4),
            'fpr': round(float(fpr), 4),
        })
    
    # Calculate disparate impact
    approval_rates = [g['approval_rate'] for g in group_metrics]
    if len(approval_rates) >= 2 and max(approval_rates) > 0:
        di_ratios = [ar / max(approval_rates) for ar in approval_rates]
        for g, di in zip(group_metrics, di_ratios):
            g['di_ratio'] = round(float(di), 4)
            g['passes_di'] = di >= 0.80
    
    all_fairness_results[attr] = group_metrics
    
    # Print table
    print(f"\n{attr} metrics:")
    for g in group_metrics:
        status = "PASS" if g.get('passes_di', True) else "FAIL"
        print(f"  {g['group']:20s} | N={g['n']:6,} | Approval: {g['approval_rate']:.2%} | "
              f"DI: {g.get('di_ratio', 'N/A')} | {status}")

# =============================================================================
# CELL 7: Fairness Dashboard
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE: Fairness Dashboard")
print("=" * 70)

n_attrs = len(all_fairness_results)
if n_attrs > 0:
    fig, axes = plt.subplots(n_attrs, 3, figsize=(18, 5*n_attrs))
    if n_attrs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (attr_name, metrics) in enumerate(all_fairness_results.items()):
        groups = [g['group'] for g in metrics]
        
        # Approval rates
        approval_rates = [g['approval_rate'] for g in metrics]
        axes[idx, 0].bar(groups, approval_rates, color='#4472C4', edgecolor='black')
        axes[idx, 0].set_title(f'{attr_name}\nApproval Rates')
        axes[idx, 0].set_ylabel('Approval Rate')
        axes[idx, 0].tick_params(axis='x', rotation=45)
        axes[idx, 0].axhline(np.mean(approval_rates), color='red', ls='--')
        
        # TPR/FPR
        x = np.arange(len(groups))
        width = 0.35
        tprs = [g['tpr'] for g in metrics]
        fprs = [g['fpr'] for g in metrics]
        axes[idx, 1].bar(x - width/2, tprs, width, label='TPR', color='#70AD47')
        axes[idx, 1].bar(x + width/2, fprs, width, label='FPR', color='#ED7D31')
        axes[idx, 1].set_title(f'{attr_name}\nTPR vs FPR')
        axes[idx, 1].set_xticks(x)
        axes[idx, 1].set_xticklabels(groups, rotation=45)
        axes[idx, 1].legend()
        
        # DI Ratios
        if all('di_ratio' in g for g in metrics):
            di_ratios = [g['di_ratio'] for g in metrics]
            colors = ['green' if d >= 0.80 else 'red' for d in di_ratios]
            axes[idx, 2].bar(groups, di_ratios, color=colors, edgecolor='black')
            axes[idx, 2].axhline(0.80, color='red', ls='--', lw=2, label='ECOA threshold')
            axes[idx, 2].set_title(f'{attr_name}\nDisparate Impact Ratios')
            axes[idx, 2].tick_params(axis='x', rotation=45)
            axes[idx, 2].legend()
    
    plt.suptitle('Fairness Audit Dashboard', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig12_fairness_dashboard.pdf", bbox_inches='tight')
    plt.show()

# =============================================================================
# CELL 8: Compliance Summary
# =============================================================================

print("\n" + "=" * 70)
print("ECOA COMPLIANCE SUMMARY")
print("=" * 70)

all_di_ratios = []
for attr, metrics in all_fairness_results.items():
    for g in metrics:
        if 'di_ratio' in g:
            all_di_ratios.append(g['di_ratio'])

passing_di = sum(1 for d in all_di_ratios if d >= 0.80)
total_di = len(all_di_ratios)

compliance_summary = {
    'status': 'COMPLIANT' if passing_di == total_di else 'REQUIRES MITIGATION',
    'disparate_impact_tests': f'{passing_di}/{total_di} passing',
    'min_di_ratio': round(min(all_di_ratios), 4) if all_di_ratios else 'N/A',
    'shap_faithfulness': round(float(mean_faithfulness), 4),
    'recommendations': [
        'Monitor approval rates across demographic groups in production',
        'Document all fairness testing for regulatory examination',
        'Apply threshold optimization per group if disparities exceed 20%',
    ]
}

print(f"\nOverall Status: {compliance_summary['status']}")
print(f"DI Tests Passing: {compliance_summary['disparate_impact_tests']}")
print(f"Minimum DI Ratio: {compliance_summary['min_di_ratio']}")
print(f"SHAP Faithfulness: {compliance_summary['shap_faithfulness']}")

with open(RESULTS_DIR / "compliance_summary.json", 'w') as f:
    json.dump(compliance_summary, f, indent=2)

# =============================================================================
# CELL 9: Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("NOTEBOOK 7 COMPLETE - ALL EXPERIMENTS FINISHED!")
print("=" * 70)

print(f"""
XAI & Fairness Results:
  SHAP Top Feature: {top_shap_features[0] if top_shap_features else 'N/A'}
  Faithfulness Score: {mean_faithfulness:.3f}
  ECOA Compliance: {compliance_summary['status']}
  DI Tests: {compliance_summary['disparate_impact_tests']}

All figures saved to: {FIGURES_DIR}/
All results saved to: {RESULTS_DIR}/

=======================================================================
PROJECT COMPLETE!
=======================================================================

You now have:
  - 7 completed Kaggle notebooks
  - Trained models (LightGBM, Mistral-7B QLoRA)
  - Hybrid fusion model
  - SHAP explanations
  - Faithfulness audit
  - Fairness audit
  - All publication-quality figures

Next steps:
  1. Download results from Kaggle
  2. Deploy API on AWS (see aws-api/ folder)
  3. Run Streamlit app (see streamlit-app/ folder)
  4. Write final paper (see research-paper/ folder)
""")
