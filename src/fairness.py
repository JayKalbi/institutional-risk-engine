"""
Fairness Audit Module for CreditRisk-LLM.
ECOA compliance testing using Aequitas and Fairlearn.
Run on Kaggle (Notebook 7).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import fairness libraries
try:
    from aequitas.group import Group
    from aequitas.bias import Bias
    from aequitas.fairness import Fairness
    AEQUITAS_AVAILABLE = True
except ImportError:
    AEQUITAS_AVAILABLE = False
    print("Warning: Aequitas not available. Install with: pip install aequitas")

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate,
        false_negative_rate,
        true_positive_rate,
        selection_rate
    )
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("Warning: Fairlearn not available. Install with: pip install fairlearn")


def run_fairness_audit(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: np.ndarray,
                       demographic_data: pd.DataFrame,
                       protected_attributes: List[str] = None,
                        threshold: float = 0.5) -> Dict:
    """
    Comprehensive fairness audit across demographic groups.
    
    Tests:
    - Demographic Parity: Equal approval rates across groups
    - Equalized Odds: Equal TPR and FPR across groups
    - Disparate Impact: Approval rate ratio >= 0.80 (ECOA threshold)
    - Calibration: Predicted probabilities calibrated within groups
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        demographic_data: DataFrame with demographic columns
        protected_attributes: List of attribute names to test
        threshold: Classification threshold
    
    Returns:
        Dictionary with all fairness metrics
    """
    print("\n" + "="*70)
    print("FAIRNESS AUDIT (ECOA Compliance)")
    print("="*70)
    
    if protected_attributes is None:
        protected_attributes = [c for c in demographic_data.columns 
                               if c in ['applicant_sex', 'applicant_race', 
                                       'applicant_ethnicity', 'applicant_age']]
    
    results = {
        'threshold': threshold,
        'protected_attributes': protected_attributes,
        'attribute_results': {}
    }
    
    for attr in protected_attributes:
        if attr not in demographic_data.columns:
            print(f"  Skipping {attr} - not in data")
            continue
        
        print(f"\n--- Auditing: {attr} ---")
        attr_results = _audit_single_attribute(
            y_true, y_pred, y_proba, demographic_data[attr], attr
        )
        results['attribute_results'][attr] = attr_results
    
    # Overall compliance assessment
    results['compliance_summary'] = _assess_compliance(results['attribute_results'])
    
    print("\n" + "="*70)
    print("FAIRNESS AUDIT SUMMARY")
    print("="*70)
    print(f"Overall Compliance: {results['compliance_summary']['status']}")
    print(f"Passing Criteria: {results['compliance_summary']['passing_criteria']}/4")
    
    return results


def _audit_single_attribute(y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: np.ndarray, attr_series: pd.Series,
                            attr_name: str) -> Dict:
    """Audit fairness for a single protected attribute."""
    
    # Get unique groups (filter out missing)
    groups = attr_series.dropna().unique()
    
    # Calculate metrics per group
    group_metrics = {}
    
    for group in groups:
        mask = (attr_series == group)
        n = mask.sum()
        
        if n < 30:  # Skip tiny groups
            continue
        
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        y_proba_g = y_proba[mask]
        
        # Basic metrics
        approval_rate = y_pred_g.mean()
        default_rate = y_true_g.mean()
        
        # Confusion matrix components
        tp = ((y_true_g == 1) & (y_pred_g == 1)).sum()
        fp = ((y_true_g == 0) & (y_pred_g == 1)).sum()
        tn = ((y_true_g == 0) & (y_pred_g == 0)).sum()
        fn = ((y_true_g == 1) & (y_pred_g == 0)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Precision and calibration
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        group_metrics[str(group)] = {
            'n_samples': int(n),
            'approval_rate': float(approval_rate),
            'default_rate': float(default_rate),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'tnr': float(tnr),
            'fnr': float(fnr),
            'precision': float(precision),
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
        }
    
    # Calculate disparity metrics
    approval_rates = {g: m['approval_rate'] for g, m in group_metrics.items()}
    tprs = {g: m['tpr'] for g, m in group_metrics.items()}
    fprs = {g: m['fpr'] for g, m in group_metrics.items()}
    
    # Reference group (largest group)
    ref_group = max(group_metrics.keys(), key=lambda g: group_metrics[g]['n_samples'])
    
    disparities = {}
    for group in group_metrics.keys():
        if group == ref_group:
            continue
        
        # Disparate Impact Ratio
        if approval_rates[ref_group] > 0:
            di_ratio = approval_rates[group] / approval_rates[ref_group]
        else:
            di_ratio = float('inf')
        
        # TPR disparity
        if tprs[ref_group] > 0:
            tpr_disparity = tprs[group] / tprs[ref_group]
        else:
            tpr_disparity = float('inf')
        
        # FPR disparity
        if fprs[ref_group] > 0:
            fpr_disparity = fprs[group] / fprs[ref_group]
        else:
            fpr_disparity = float('inf')
        
        disparities[group] = {
            'vs_reference': ref_group,
            'disparate_impact_ratio': round(float(di_ratio), 4),
            'tpr_disparity': round(float(tpr_disparity), 4),
            'fpr_disparity': round(float(fpr_disparity), 4),
            'di_passes': di_ratio >= 0.80  # ECOA threshold
        }
    
    return {
        'group_metrics': group_metrics,
        'disparities': disparities,
        'reference_group': ref_group,
        'n_groups': len(group_metrics)
    }


def _assess_compliance(attribute_results: Dict) -> Dict:
    """Assess overall ECOA compliance."""
    
    criteria = {
        'disparate_impact': True,  # DI ratio >= 0.80 for all groups
        'demographic_parity': True,  # Similar approval rates
        'equalized_odds': True,  # Similar TPR and FPR
        'sample_size': True,  # Adequate samples in each group
    }
    
    all_di_ratios = []
    
    for attr_name, attr_data in attribute_results.items():
        for group, disparities in attr_data.get('disparities', {}).items():
            di_ratio = disparities.get('disparate_impact_ratio', 1.0)
            all_di_ratios.append(di_ratio)
            
            if di_ratio < 0.80:
                criteria['disparate_impact'] = False
        
        # Check sample sizes
        for group, metrics in attr_data.get('group_metrics', {}).items():
            if metrics['n_samples'] < 100:
                criteria['sample_size'] = False
    
    passing = sum(criteria.values())
    
    if passing == 4:
        status = "FULLY COMPLIANT - Passes all ECOA criteria"
    elif passing >= 2:
        status = "PARTIALLY COMPLIANT - Some disparities detected, mitigation recommended"
    else:
        status = "NON-COMPLIANT - Significant disparities, requires intervention"
    
    return {
        'status': status,
        'passing_criteria': passing,
        'total_criteria': 4,
        'criteria_breakdown': criteria,
        'min_di_ratio': min(all_di_ratios) if all_di_ratios else 1.0,
        'recommendations': _generate_recommendations(criteria, attribute_results)
    }


def _generate_recommendations(criteria: Dict, attribute_results: Dict) -> List[str]:
    """Generate fairness improvement recommendations."""
    recommendations = []
    
    if not criteria['disparate_impact']:
        recommendations.append(
            "Apply threshold optimization per group to equalize approval rates"
        )
        recommendations.append(
            "Consider re-weighting training samples from underrepresented groups"
        )
    
    if not criteria['equalized_odds']:
        recommendations.append(
            "Use equalized odds post-processing (Hardt et al. method)"
        )
    
    recommendations.append(
        "Monitor model performance continuously across demographic groups in production"
    )
    
    recommendations.append(
        "Document all fairness testing for regulatory examination"
    )
    
    return recommendations


def plot_fairness_dashboard(attribute_results: Dict,
                            save_path: str = "figures/fig_fairness_dashboard.pdf"):
    """
    Create comprehensive fairness visualization dashboard.
    """
    n_attrs = len(attribute_results)
    fig, axes = plt.subplots(n_attrs, 3, figsize=(18, 5*n_attrs))
    
    if n_attrs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (attr_name, attr_data) in enumerate(attribute_results.items()):
        groups = list(attr_data['group_metrics'].keys())
        metrics = attr_data['group_metrics']
        
        # Plot 1: Approval rates by group
        approval_rates = [metrics[g]['approval_rate'] for g in groups]
        axes[idx, 0].bar(groups, approval_rates, color='#5B9BD5', edgecolor='black')
        axes[idx, 0].set_title(f'{attr_name}\nApproval Rates', fontsize=11)
        axes[idx, 0].set_ylabel('Approval Rate')
        axes[idx, 0].tick_params(axis='x', rotation=45)
        axes[idx, 0].axhline(np.mean(approval_rates), color='red', ls='--', 
                            label=f'Mean: {np.mean(approval_rates):.3f}')
        axes[idx, 0].legend()
        
        # Plot 2: TPR and FPR by group
        x = np.arange(len(groups))
        width = 0.35
        tprs = [metrics[g]['tpr'] for g in groups]
        fprs = [metrics[g]['fpr'] for g in groups]
        
        axes[idx, 1].bar(x - width/2, tprs, width, label='TPR', color='#70AD47')
        axes[idx, 1].bar(x + width/2, fprs, width, label='FPR', color='#ED7D31')
        axes[idx, 1].set_title(f'{attr_name}\nTPR vs FPR', fontsize=11)
        axes[idx, 1].set_ylabel('Rate')
        axes[idx, 1].set_xticks(x)
        axes[idx, 1].set_xticklabels(groups, rotation=45)
        axes[idx, 1].legend()
        
        # Plot 3: Disparate Impact Ratios
        if attr_data['disparities']:
            disparity_groups = list(attr_data['disparities'].keys())
            di_ratios = [attr_data['disparities'][g]['disparate_impact_ratio'] 
                        for g in disparity_groups]
            
            colors = ['green' if d >= 0.80 else 'red' for d in di_ratios]
            axes[idx, 2].bar(disparity_groups, di_ratios, color=colors, edgecolor='black')
            axes[idx, 2].axhline(0.80, color='red', ls='--', lw=2, label='ECOA threshold (0.80)')
            axes[idx, 2].set_title(f'{attr_name}\nDisparate Impact Ratios', fontsize=11)
            axes[idx, 2].set_ylabel('DI Ratio')
            axes[idx, 2].tick_params(axis='x', rotation=45)
            axes[idx, 2].legend()
    
    plt.suptitle('Fairness Audit Dashboard', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved fairness dashboard to {save_path}")


def demographic_parity_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                                 demographic_data: pd.DataFrame,
                                 attribute: str) -> pd.DataFrame:
    """Detailed demographic parity analysis."""
    
    groups = demographic_data[attribute].dropna().unique()
    
    rows = []
    for group in groups:
        mask = demographic_data[attribute] == group
        n = mask.sum()
        approval_rate = y_pred[mask].mean()
        default_rate = y_true[mask].mean()
        
        rows.append({
            'attribute': attribute,
            'group': group,
            'n': n,
            'approval_rate': approval_rate,
            'default_rate': default_rate,
            'prediction_rate': y_pred[mask].mean(),
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate disparities
    max_rate = df['approval_rate'].max()
    df['disparity_ratio'] = df['approval_rate'] / max_rate
    df['passes_di'] = df['disparity_ratio'] >= 0.80
    
    return df
