"""
Model training and inference utilities for CreditRisk-LLM.
Includes: LightGBM, XGBoost, Logistic Regression, and Hybrid Fusion.
Run on Kaggle (Notebooks 4 & 6).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, precision_score, recall_score,
                             brier_score_loss, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve)
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Any, Optional
import json
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, 
                   y_pred: Optional[np.ndarray] = None,
                   model_name: str = "Model") -> Dict[str, float]:
    """
    Comprehensive model evaluation with all key credit risk metrics.
    
    Metrics:
    - AUC-ROC: Primary discrimination metric
    - PR-AUC: Critical for imbalanced datasets
    - KS Statistic: Industry standard for scorecards
    - Brier Score: Probability calibration
    - F1, Precision, Recall: Classification performance
    
    Returns:
        Dictionary of metric names to values
    """
    if y_pred is None:
        y_pred = (y_proba >= 0.5).astype(int)
    
    # Primary metrics
    auc_roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # KS statistic (Kolmogorov-Smirnov)
    ks_stat, _ = ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0])
    
    # Calibration
    brier = brier_score_loss(y_true, y_proba)
    
    # Classification metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # F1 for default class specifically
    f1_default = f1_score(y_true, y_pred, pos_label=1)
    
    results = {
        'model': model_name,
        'auc_roc': round(float(auc_roc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'ks_statistic': round(float(ks_stat), 4),
        'brier_score': round(float(brier), 4),
        'f1': round(float(f1), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_default': round(float(f1_default), 4),
    }
    
    print(f"\n{'='*50}")
    print(f"Results: {model_name}")
    print(f"{'='*50}")
    for k, v in results.items():
        if k != 'model':
            print(f"  {k:20s}: {v}")
    
    return results


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               feature_names: Optional[List[str]] = None) -> Dict:
    """
    Train Logistic Regression (regulatory baseline).
    Standardized features + balanced class weights.
    """
    print("\n" + "="*60)
    print("Training: Logistic Regression")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    results = evaluate_model(y_test, y_proba, model_name="Logistic Regression")
    
    # Feature importance (coefficients)
    if feature_names:
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        results['top_features'] = coef_df.head(10).to_dict('records')
        print("\nTop 10 features by coefficient:")
        print(coef_df.head(10).to_string())
    
    results['model_object'] = model
    results['scaler'] = scaler
    
    return results


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  feature_names: Optional[List[str]] = None) -> Dict:
    """
    Train XGBoost classifier with early stopping.
    """
    print("\n" + "="*60)
    print("Training: XGBoost")
    print("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=7,  # ~88/12 class ratio
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    y_proba = model.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_proba, model_name="XGBoost")
    
    # Feature importance
    if feature_names:
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        results['top_features'] = imp_df.head(10).to_dict('records')
        print("\nTop 10 features by importance:")
        print(imp_df.head(10).to_string())
    
    results['model_object'] = model
    
    return results


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: Optional[List[str]] = None) -> Dict:
    """
    Train LightGBM classifier (primary classical ML baseline).
    Optimized hyperparameters for credit risk.
    """
    print("\n" + "="*60)
    print("Training: LightGBM")
    print("="*60)
    
    model = lgb.LGBMClassifier(
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
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=200), lgb.early_stopping(50)]
    )
    
    y_proba = model.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_proba, model_name="LightGBM")
    
    # Feature importance
    if feature_names:
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        results['top_features'] = imp_df.head(10).to_dict('records')
        print("\nTop 10 features by importance:")
        print(imp_df.head(10).to_string())
    
    results['model_object'] = model
    
    return results


def train_hybrid_fusion(lgb_train_proba: np.ndarray,
                        lgb_val_proba: np.ndarray,
                        lgb_test_proba: np.ndarray,
                        llm_train_proba: np.ndarray,
                        llm_val_proba: np.ndarray,
                        llm_test_proba: np.ndarray,
                        y_train: np.ndarray,
                        y_val: np.ndarray,
                        y_test: np.ndarray) -> Dict:
    """
    Train late-fusion meta-learner combining LightGBM + LLM predictions.
    
    Uses Logistic Regression as meta-learner fitted on validation set
    to prevent data leakage.
    
    Returns:
        Dictionary with hybrid results and fusion weights
    """
    print("\n" + "="*60)
    print("Training: Hybrid Fusion (LightGBM + LLM)")
    print("="*60)
    
    # Build meta-feature matrices
    meta_train = np.column_stack([lgb_train_proba, llm_train_proba])
    meta_val = np.column_stack([lgb_val_proba, llm_val_proba])
    meta_test = np.column_stack([lgb_test_proba, llm_test_proba])
    
    # Train meta-learner on VALIDATION set (prevent leakage)
    meta_learner = LogisticRegression(C=1.0, max_iter=1000)
    meta_learner.fit(meta_val, y_val)
    
    # Predict
    hybrid_proba = meta_learner.predict_proba(meta_test)[:, 1]
    
    # Fusion weights
    w_lgb = meta_learner.coef_[0][0]
    w_llm = meta_learner.coef_[0][1]
    
    print(f"\nFusion Weights:")
    print(f"  LightGBM: {w_lgb:.4f}")
    print(f"  LLM:      {w_llm:.4f}")
    print(f"  Ratio (LLM/LGB): {abs(w_llm/w_lgb)*100:.1f}%")
    
    results = evaluate_model(y_test, hybrid_proba, model_name="HybridCredit-LLM")
    results['fusion_weights'] = {
        'lightgbm': float(w_lgb),
        'llm': float(w_llm)
    }
    results['meta_learner'] = meta_learner
    
    return results


def compare_all_models(results_list: List[Dict], save_path: str = "results/model_comparison.json") -> pd.DataFrame:
    """
    Create comparison table of all models.
    
    Returns:
        DataFrame with all models and metrics
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    
    rows = []
    for r in results_list:
        row = {k: v for k, v in r.items() if k not in ['model_object', 'scaler', 'meta_learner', 'top_features']}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    metric_cols = ['auc_roc', 'pr_auc', 'ks_statistic', 'brier_score', 'f1', 'f1_default']
    df = df[['model'] + [c for c in metric_cols if c in df.columns]]
    
    print("\n" + df.to_string(index=False))
    
    # Save
    df.to_json(save_path, orient='records', indent=2)
    print(f"\nSaved comparison to {save_path}")
    
    return df


def calibrate_probabilities(model, X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray) -> np.ndarray:
    """
    Calibrate predicted probabilities using isotonic regression.
    Fit on validation set, apply to test set.
    """
    from sklearn.isotonic import IsotonicRegression
    
    # Get uncalibrated probabilities
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Fit isotonic regression
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(val_proba, y_val)
    
    # Calibrate test predictions
    calibrated = calibrator.predict(test_proba)
    
    return calibrated


def get_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find optimal classification threshold using Youden's J statistic.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")
    print(f"TPR at optimal: {tpr[optimal_idx]:.3f}")
    print(f"FPR at optimal: {fpr[optimal_idx]:.3f}")
    
    return optimal_threshold
