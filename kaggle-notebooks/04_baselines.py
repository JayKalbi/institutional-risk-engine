"""
===============================================================================
KAGGLE NOTEBOOK 4: Classical ML Baselines
===============================================================================
Run this FOURTH on Kaggle (CPU only - no GPU needed)

What this notebook does:
1. Trains 3 classical ML models:
   - Logistic Regression (regulatory baseline)
   - XGBoost
   - LightGBM (primary classical baseline)
2. Evaluates all models with comprehensive metrics
3. Generates ROC, PR, calibration, and confusion matrix plots
4. Extracts feature importance for SHAP analysis
5. Saves models and results for fusion (Notebook 6)

PREREQUISITE: Notebook 3 (preprocessing)
KAGGLE SETUP: Accelerator: None, Runtime: ~30-45 minutes
===============================================================================
"""

# =============================================================================
# CELL 1: Imports & Setup
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, precision_score, recall_score,
                             brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve, classification_report)
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

WORKING_DIR = Path("./output")
PROCESSED_DIR = WORKING_DIR / "data" / "processed"
FIGURES_DIR = WORKING_DIR / "figures"
RESULTS_DIR = WORKING_DIR / "results"
MODELS_DIR = WORKING_DIR / "models"

for d in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KAGGLE NOTEBOOK 4: Classical ML Baselines")
print("=" * 70)

# =============================================================================
# CELL 2: Load Processed Data
# =============================================================================

print("\nLoading processed data...")
X_train = np.load(PROCESSED_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy")
X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

with open(PROCESSED_DIR / "feature_names.json", 'r') as f:
    feature_names = json.load(f)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Features: {len(feature_names)}")
print(f"Denial rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")

# =============================================================================
# CELL 3: Evaluation Function
# =============================================================================

def evaluate_model(y_true, y_proba, model_name):
    """Comprehensive model evaluation."""
    y_pred = (y_proba >= 0.5).astype(int)
    
    results = {
        'model': model_name,
        'auc_roc': round(float(roc_auc_score(y_true, y_proba)), 4),
        'pr_auc': round(float(average_precision_score(y_true, y_proba)), 4),
        'ks_statistic': round(float(ks_2samp(y_proba[y_true==1], y_proba[y_true==0])[0]), 4),
        'brier_score': round(float(brier_score_loss(y_true, y_proba)), 4),
        'f1': round(float(f1_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_default': round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
    }
    
    print(f"\n{'='*50}")
    print(f"Results: {model_name}")
    print(f"{'='*50}")
    for k, v in results.items():
        if k != 'model':
            print(f"  {k:20s}: {v}")
    
    return results

# =============================================================================
# CELL 4: Model 1 - Logistic Regression
# =============================================================================

print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', 
                        random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)

lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_results = evaluate_model(y_test, lr_proba, "Logistic Regression")

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print("\nTop 10 features (by coefficient magnitude):")
print(coef_df.head(10).to_string())

# Save
joblib.dump(lr, MODELS_DIR / "logistic_regression.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
np.save(PROCESSED_DIR / "lr_test_proba.npy", lr_proba)

# =============================================================================
# CELL 5: Model 2 - XGBoost
# =============================================================================

print("\n" + "=" * 70)
print("MODEL 2: XGBOOST")
print("=" * 70)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_results = evaluate_model(y_test, xgb_proba, "XGBoost")

# Feature importance
imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 features (by importance):")
print(imp_df.head(10).to_string())

# Save
joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")
np.save(PROCESSED_DIR / "xgb_test_proba.npy", xgb_proba)

# Also save validation probabilities for fusion
xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
np.save(PROCESSED_DIR / "xgb_val_proba.npy", xgb_val_proba)

# =============================================================================
# CELL 6: Model 3 - LightGBM (Primary Baseline)
# =============================================================================

print("\n" + "=" * 70)
print("MODEL 3: LIGHTGBM (Primary Classical Baseline)")
print("=" * 70)

lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    max_depth=8,
    num_leaves=63,
    learning_rate=0.03,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7,
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.log_evaluation(period=200), lgb.early_stopping(50)]
)

lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_results = evaluate_model(y_test, lgb_proba, "LightGBM")

# Feature importance
lgb_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 features (by importance):")
print(lgb_imp.head(10).to_string())

# Save
joblib.dump(lgb_model, MODELS_DIR / "lightgbm.joblib")
np.save(PROCESSED_DIR / "lgb_test_proba.npy", lgb_proba)

# Save validation and train probabilities for fusion
lgb_val_proba = lgb_model.predict_proba(X_val)[:, 1]
lgb_train_proba = lgb_model.predict_proba(X_train)[:, 1]
np.save(PROCESSED_DIR / "lgb_val_proba.npy", lgb_val_proba)
np.save(PROCESSED_DIR / "lgb_train_proba.npy", lgb_train_proba)

# =============================================================================
# CELL 7: Results Comparison Table
# =============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)

all_results = [lr_results, xgb_results, lgb_results]
results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['model_object', 'scaler', 'meta_learner', 'top_features']}
    for r in all_results
])

print("\n" + results_df.to_string(index=False))

# Save
results_df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
results_df.to_json(RESULTS_DIR / "baseline_results.json", orient='records', indent=2)

# Save predictions for fusion
predictions_dict = {
    'Logistic Regression': lr_proba,
    'XGBoost': xgb_proba,
    'LightGBM': lgb_proba
}

# =============================================================================
# CELL 8: ROC Curves
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE: ROC Curves")
print("=" * 70)

plt.figure(figsize=(8, 7))

colors = ['#888888', '#4472C4', '#ED7D31']
for (name, proba), color in zip(predictions_dict.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, lw=2)

plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random (AUC=0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Classical ML Baselines', fontsize=13)
plt.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_baselines_roc.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig_baselines_roc.png", bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 9: Precision-Recall Curves
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE: Precision-Recall Curves")
print("=" * 70)

plt.figure(figsize=(8, 7))

baseline = y_test.mean()
for (name, proba), color in zip(predictions_dict.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    plt.plot(recall, precision, label=f'{name} (AP={pr_auc:.3f})', color=color, lw=2)

plt.axhline(baseline, color='k', ls=':', lw=1, label=f'Baseline ({baseline:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=13)
plt.legend(fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_baselines_pr.pdf", bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 10: Calibration Curves
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE: Calibration Curves")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for (name, proba), color in zip(predictions_dict.items(), colors):
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    ax1.plot(prob_pred, prob_true, 'o-', label=name, color=color, markersize=6)
    
    brier = brier_score_loss(y_test, proba)
    ax2.bar(name, brier, color=color, alpha=0.7)

ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
ax1.set_xlabel('Mean Predicted Probability')
ax1.set_ylabel('Fraction of Positives')
ax1.set_title('(a) Reliability Diagram')
ax1.legend()

ax2.set_ylabel('Brier Score (lower=better)')
ax2.set_title('(b) Brier Score Comparison')
ax2.tick_params(axis='x', rotation=45)

plt.suptitle('Model Calibration', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_baselines_calibration.pdf", bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 11: Confusion Matrices
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE: Confusion Matrices")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, ((name, proba), color, ax) in enumerate(zip(predictions_dict.items(), colors, axes)):
    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Approved', 'Denied'],
               yticklabels=['Approved', 'Denied'])
    ax.set_title(name, fontsize=11)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_baselines_confusion.pdf", bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 12: Save Final Results
# =============================================================================

print("\n" + "=" * 70)
print("SAVING ALL RESULTS")
print("=" * 70)

# Save comprehensive results
final_results = {
    'models': {r['model']: {k: v for k, v in r.items() if k != 'model'} for r in all_results},
    'best_model': 'LightGBM',
    'best_auc_roc': lgb_results['auc_roc'],
    'n_features': len(feature_names),
    'feature_names': feature_names
}

with open(RESULTS_DIR / "baseline_full_results.json", 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\nSaved results to {RESULTS_DIR}/")
print(f"Saved models to {MODELS_DIR}/")
print(f"Saved predictions to {PROCESSED_DIR}/")

print("\n" + "=" * 70)
print("NOTEBOOK 4 COMPLETE!")
print("=" * 70)
print(f"\nBest baseline: LightGBM (AUC-ROC: {lgb_results['auc_roc']:.4f})")
print(f"\nNext: Run Notebook 5 (Mistral-7B QLoRA Fine-Tuning) - GPU REQUIRED")
