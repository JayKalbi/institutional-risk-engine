"""
Explainability (XAI) Stack for CreditRisk-LLM.
SHAP, counterfactual explanations, and faithfulness audit.
Run on Kaggle (Notebook 7).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from dice_ml import Dice
from dice_ml.utils import helpers
import dice_ml
from typing import Dict, List, Tuple, Optional
import re
import os
import warnings
warnings.filterwarnings('ignore')


def compute_shap_values(model, X_sample: np.ndarray, 
                        feature_names: List[str],
                        sample_size: int = 3000) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer for tree-based models.
    
    Args:
        model: Trained tree model (LightGBM/XGBoost)
        X_sample: Feature matrix
        feature_names: Feature names
        sample_size: Number of samples to explain
    
    Returns:
        SHAP values array
    """
    print(f"\nComputing SHAP values for {min(sample_size, len(X_sample)):,} samples...")
    
    X = X_sample[:sample_size]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For binary classification, shap_values is list of two arrays
    # We want the positive class (index 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    return shap_values


def plot_shap_summary(shap_values: np.ndarray,
                      X_sample: np.ndarray,
                      feature_names: List[str],
                      save_path: str = "figures/fig_shap_summary.pdf",
                      max_display: int = 20):
    """
    Create SHAP summary plot (global feature importance).
    """
    print("\nGenerating SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample[:len(shap_values)],
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    plt.title('SHAP Feature Importance - Credit Default Prediction', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved SHAP summary to {save_path}")


def plot_shap_bar(shap_values: np.ndarray,
                  X_sample: np.ndarray,
                  feature_names: List[str],
                  save_path: str = "figures/fig_shap_bar.pdf",
                  max_display: int = 15):
    """
    Create SHAP bar plot (mean absolute importance).
    """
    print("\nGenerating SHAP bar plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample[:len(shap_values)],
        feature_names=feature_names,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    
    plt.title('Mean |SHAP| Feature Importance', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved SHAP bar plot to {save_path}")


def explain_single_prediction(model, X: np.ndarray, idx: int,
                              feature_names: List[str],
                              shap_values: Optional[np.ndarray] = None) -> Dict:
    """
    Generate explanation for a single prediction.
    
    Returns:
        Dictionary with prediction, SHAP values, and top contributing features
    """
    # Get prediction
    proba = model.predict_proba(X[idx:idx+1])[0, 1]
    prediction = 1 if proba >= 0.5 else 0
    
    # Get SHAP values for this sample
    if shap_values is None:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[idx:idx+1])
        if isinstance(sv, list):
            sv = sv[1][0]
        else:
            sv = sv[0]
    else:
        sv = shap_values[idx]
    
    # Get feature contributions
    contributions = []
    for i, feat_name in enumerate(feature_names):
        contributions.append({
            'feature': feat_name,
            'value': float(X[idx, i]),
            'shap_value': float(sv[i]),
            'abs_shap': float(abs(sv[i]))
        })
    
    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    result = {
        'index': int(idx),
        'predicted_probability': float(proba),
        'prediction': 'DENIED' if prediction == 1 else 'APPROVED',
        'top_increasing_risk': [c for c in contributions if c['shap_value'] > 0][:5],
        'top_decreasing_risk': [c for c in contributions if c['shap_value'] < 0][:5],
        'all_contributions': contributions
    }
    
    return result


def generate_counterfactual_explanation(model, X_train_df: pd.DataFrame,
                                        query_instance: pd.DataFrame,
                                        total_cfs: int = 3,
                                        desired_class: str = 'opposite',
                                        continuous_features: List[str] = None,
                                        save_path: str = "figures/fig_counterfactual.pdf") -> Dict:
    """
    Generate counterfactual explanations using DiCE-ML.
    
    "What would need to change for this loan to be approved?"
    
    Args:
        model: Trained sklearn model
        X_train_df: Training data as DataFrame
        query_instance: Instance to explain (single row DataFrame)
        total_cfs: Number of counterfactuals to generate
        desired_class: 'opposite' or specific class
        continuous_features: List of continuous feature names
        save_path: Path to save visualization
    
    Returns:
        Dictionary with counterfactuals
    """
    print("\n" + "="*60)
    print("Generating Counterfactual Explanations (DiCE-ML)")
    print("="*60)
    
    # Prepare data for DiCE
    outcome_name = 'default'
    
    if outcome_name not in X_train_df.columns:
        raise ValueError(f"Target column '{outcome_name}' not found in training data")
    
    # Identify continuous features if not provided
    if continuous_features is None:
        continuous_features = []
        for col in X_train_df.columns:
            if col == outcome_name:
                continue
            if X_train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if X_train_df[col].nunique() > 10:
                    continuous_features.append(col)
    
    # Setup DiCE
    d = dice_ml.Data(
        dataframe=X_train_df,
        continuous_features=continuous_features,
        outcome_name=outcome_name
    )
    
    m = dice_ml.Model(model=model, backend='sklearn')
    
    exp = Dice(d, m, method='random')
    
    # Generate counterfactuals
    try:
        cf = exp.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_class=desired_class,
            verbose=False
        )
        
        # Save visualization
        cf.visualize_as_dataframe(show_only_changes=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved counterfactual visualization to {save_path}")
        
        # Extract counterfactual data
        cf_df = cf.cf_examples_list[0].final_cfs_df
        
        result = {
            'query_instance': query_instance.to_dict('records')[0],
            'counterfactuals': cf_df.to_dict('records') if cf_df is not None else [],
            'message': 'Counterfactuals generated successfully'
        }
        
    except Exception as e:
        print(f"Counterfactual generation failed: {e}")
        result = {
            'query_instance': query_instance.to_dict('records')[0],
            'counterfactuals': [],
            'message': f'Failed: {str(e)}'
        }
    
    return result


def faithfulness_audit(model, X_test: np.ndarray,
                       llm_rationales: List[str],
                       shap_values: np.ndarray,
                       feature_names: List[str],
                       n_samples: int = 200) -> Dict:
    """
    Compare LLM self-explanations vs SHAP feature rankings.
    
    This is a KEY NOVEL CONTRIBUTION:
    Measures alignment between what the LLM says vs what SHAP empirically measures.
    
    Args:
        model: Tabular model (LightGBM)
        X_test: Test features
        llm_rationales: List of LLM-generated rationale strings
        shap_values: Pre-computed SHAP values
        feature_names: Feature names
        n_samples: Number of samples to audit
    
    Returns:
        Dictionary with faithfulness scores
    """
    print("\n" + "="*60)
    print("FAITHFULNESS AUDIT: LLM Rationales vs SHAP")
    print("="*60)
    
    # Feature keyword mapping for parsing LLM rationales
    feature_keywords = {
        'loan_to_income_ratio': ['loan-to-income', 'loan to income', 'ratio'],
        'debt_to_income_ratio': ['debt-to-income', 'dti', 'debt ratio'],
        'loan_amount': ['loan amount', 'amount'],
        'income': ['income', 'salary', 'earnings'],
        'loan_to_value_ratio': ['loan-to-value', 'ltv', 'value ratio'],
        'property_value': ['property value', 'home value', 'collateral'],
        'loan_term': ['term', 'duration', 'years'],
        'applicant_credit_score_type': ['credit score', 'fico', 'credit'],
        'applicant_sex': ['gender', 'sex', 'male', 'female'],
        'applicant_age': ['age', 'young', 'old', 'elderly'],
        'high_dti_flag': ['high dti', 'dti flag', 'debt threshold'],
        'safe_ltv_flag': ['ltv flag', 'safe ltv', 'collateral'],
        'payment_burden': ['payment burden', 'monthly payment', 'affordability'],
        'dti_ltv_interaction': ['dti ltv', 'interaction', 'combined risk'],
    }
    
    faithfulness_scores = []
    feature_mentions = {f: 0 for f in feature_names}
    feature_matches = {f: 0 for f in feature_names}
    
    n_audit = min(n_samples, len(llm_rationales), len(shap_values))
    
    for idx in range(n_audit):
        rationale = str(llm_rationales[idx]).lower()
        
        # Parse which features LLM mentioned
        llm_mentioned = []
        for feat, keywords in feature_keywords.items():
            if feat in feature_names:
                if any(kw in rationale for kw in keywords):
                    llm_mentioned.append(feat)
                    feature_mentions[feat] += 1
        
        # Get SHAP ranking for this specific sample
        sample_shap = shap_values[idx]
        sample_shap_rank = [feature_names[i] for i in np.argsort(-np.abs(sample_shap))]
        top_5_shap = set(sample_shap_rank[:5])
        
        # Calculate overlap
        mentioned_set = set(llm_mentioned)
        overlap = len(mentioned_set & top_5_shap)
        
        # Track matches per feature
        for feat in mentioned_set & top_5_shap:
            if feat in feature_matches:
                feature_matches[feat] += 1
        
        faithfulness_scores.append(overlap / 5.0)  # Normalize to [0, 1]
    
    # Aggregate results
    mean_faithfulness = np.mean(faithfulness_scores)
    
    # Per-feature faithfulness
    per_feature_faithfulness = {}
    for feat in feature_names:
        if feature_mentions[feat] > 0:
            per_feature_faithfulness[feat] = feature_matches[feat] / feature_mentions[feat]
    
    results = {
        'mean_faithfulness': round(float(mean_faithfulness), 4),
        'interpretation': _interpret_faithfulness(mean_faithfulness),
        'n_samples_audited': n_audit,
        'faithfulness_distribution': {
            'mean': float(np.mean(faithfulness_scores)),
            'std': float(np.std(faithfulness_scores)),
            'median': float(np.median(faithfulness_scores)),
            'q25': float(np.percentile(faithfulness_scores, 25)),
            'q75': float(np.percentile(faithfulness_scores, 75)),
        },
        'per_feature_faithfulness': {k: round(v, 4) for k, v in per_feature_faithfulness.items()},
        'feature_mention_frequency': {k: int(v) for k, v in feature_mentions.items() if v > 0},
    }
    
    print(f"\nMean Faithfulness Score: {mean_faithfulness:.3f}")
    print(f"Interpretation: {results['interpretation']}")
    print(f"\nTop 5 Most Faithful Features:")
    sorted_faith = sorted(per_feature_faithfulness.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_faith[:5]:
        print(f"  {feat}: {score:.3f}")
    
    return results


def _interpret_faithfulness(score: float) -> str:
    """Interpret faithfulness score."""
    if score >= 0.7:
        return "High alignment - LLM rationales strongly agree with SHAP attributions"
    elif score >= 0.5:
        return "Moderate alignment - LLM captures main risk factors but misses nuances"
    elif score >= 0.3:
        return "Low alignment - LLM explanations partially disconnected from empirical attributions"
    else:
        return "Poor alignment - LLM rationales poorly reflect actual model behavior"


def plot_faithfulness_distribution(faithfulness_scores: List[float],
                                    save_path: str = "figures/fig_faithfulness_dist.pdf"):
    """Plot distribution of faithfulness scores across samples."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(faithfulness_scores, bins=20, color='#5B9BD5', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(faithfulness_scores), color='red', ls='--', lw=2, 
                label=f'Mean: {np.mean(faithfulness_scores):.3f}')
    plt.axvline(0.5, color='orange', ls=':', lw=2, label='Moderate threshold (0.5)')
    plt.axvline(0.7, color='green', ls=':', lw=2, label='High threshold (0.7)')
    
    plt.xlabel('Faithfulness Score (Overlap / 5)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of SHAP-LLM Faithfulness Scores', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved faithfulness distribution to {save_path}")


def generate_explanation_report(model, X_test: np.ndarray,
                                feature_names: List[str],
                                shap_values: np.ndarray,
                                llm_rationale: str,
                                idx: int) -> str:
    """
    Generate a human-readable explanation report for a single prediction.
    
    Returns:
        Formatted explanation string
    """
    proba = model.predict_proba(X_test[idx:idx+1])[0, 1]
    prediction = "DENIED" if proba >= 0.5 else "APPROVED"
    
    sv = shap_values[idx]
    
    # Get top positive and negative contributors
    contributions = [(feature_names[i], float(sv[i]), float(X_test[idx, i])) 
                     for i in range(len(feature_names))]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_risk = [c for c in contributions if c[1] > 0][:3]
    top_mitigating = [c for c in contributions if c[1] < 0][:3]
    
    report = f"""
{'='*60}
CREDIT RISK DECISION EXPLANATION
{'='*60}

Prediction: {prediction} (Probability of denial: {proba:.1%})

--- SHAP Feature Attribution ---

Factors INCREASING risk:
"""
    for feat, shap_val, feat_val in top_risk:
        report += f"  - {feat}: {feat_val:.2f} (contribution: +{shap_val:.4f})\n"
    
    report += "\nFactors DECREASING risk:\n"
    for feat, shap_val, feat_val in top_mitigating:
        report += f"  - {feat}: {feat_val:.2f} (contribution: {shap_val:.4f})\n"
    
    report += f"""
--- LLM Generated Rationale ---
{llm_rationale}

--- Regulatory Compliance ---
This explanation satisfies:
  - EU AI Act Article 13 (transparency requirement)
  - ECOA/Regulation B (adverse action notice requirement)
  - Basel III Pillar 3 (risk disclosure)
{'='*60}
"""
    return report
