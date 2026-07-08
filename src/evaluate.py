"""
Evaluation utilities: ROC/PR curves, calibration plots, model comparison visualizations.
Run on Kaggle (Notebooks 4 & 6).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, 
                             confusion_matrix, classification_report,
                             brier_score_loss, calibration_curve)
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_roc_curves(results_dict: Dict[str, np.ndarray], 
                    y_true: np.ndarray,
                    save_path: str = "figures/roc_curves.pdf"):
    """
    Plot ROC curves for all models on the same axes.
    
    Args:
        results_dict: {model_name: y_proba_array}
        y_true: True labels
        save_path: Where to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = ['#888888', '#4472C4', '#ED7D31', '#5B9BD5', '#70AD47', '#C00000']
    
    for (name, y_proba), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapz(tpr, fpr)
        
        lw = 2.5 if 'Hybrid' in name else 1.5
        ls = '-' if 'Hybrid' in name else '--'
        
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                color=color, lw=lw, ls=ls)
    
    ax.plot([0, 1], [0, 1], 'k:', lw=1, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Credit Default Prediction', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved ROC curves to {save_path}")


def plot_pr_curves(results_dict: Dict[str, np.ndarray],
                   y_true: np.ndarray,
                   save_path: str = "figures/pr_curves.pdf"):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = ['#888888', '#4472C4', '#ED7D31', '#5B9BD5', '#70AD47', '#C00000']
    baseline = y_true.mean()
    
    for (name, y_proba), color in zip(results_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = np.trapz(precision, recall)
        
        lw = 2.5 if 'Hybrid' in name else 1.5
        ls = '-' if 'Hybrid' in name else '--'
        
        ax.plot(recall, precision, label=f'{name} (AP={pr_auc:.3f})',
                color=color, lw=lw, ls=ls)
    
    ax.axhline(baseline, color='k', ls=':', lw=1, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved PR curves to {save_path}")


def plot_calibration_curves(results_dict: Dict[str, np.ndarray],
                            y_true: np.ndarray,
                            save_path: str = "figures/calibration_curves.pdf"):
    """Plot calibration curves (reliability diagrams) for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#888888', '#4472C4', '#ED7D31', '#5B9BD5', '#70AD47', '#C00000']
    
    for (name, y_proba), color in zip(results_dict.items(), colors):
        # Reliability diagram
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        ax1.plot(prob_pred, prob_true, 'o-', label=name, color=color, markersize=6)
        
        # Brier score
        brier = brier_score_loss(y_true, y_proba)
        ax2.bar(name[:15], brier, color=color, alpha=0.7)
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('(a) Reliability Diagram', fontsize=13)
    ax1.legend(fontsize=8, loc='upper left')
    
    ax2.set_ylabel('Brier Score (lower=better)', fontsize=12)
    ax2.set_title('(b) Brier Score Comparison', fontsize=13)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved calibration curves to {save_path}")


def plot_confusion_matrices(results_dict: Dict[str, np.ndarray],
                            y_true: np.ndarray,
                            save_path: str = "figures/confusion_matrices.pdf",
                            threshold: float = 0.5):
    """Plot confusion matrices for all models."""
    n_models = len(results_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, ((name, y_proba), ax) in enumerate(zip(results_dict.items(), axes)):
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Approved', 'Denied'],
                   yticklabels=['Approved', 'Denied'])
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrices to {save_path}")


def plot_ks_curves(results_dict: Dict[str, np.ndarray],
                   y_true: np.ndarray,
                   save_path: str = "figures/ks_curves.pdf"):
    """Plot KS (Kolmogorov-Smirnov) statistic curves."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = ['#888888', '#4472C4', '#ED7D31', '#5B9BD5', '#70AD47', '#C00000']
    
    for (name, y_proba), color in zip(results_dict.items(), colors):
        # Sort by probability
        sorted_indices = np.argsort(y_proba)
        y_sorted = y_true[sorted_indices]
        proba_sorted = y_proba[sorted_indices]
        
        # Cumulative distributions
        n = len(y_true)
        cum_defaults = np.cumsum(y_sorted) / y_sorted.sum()
        cum_non_defaults = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
        
        ks_stat = np.max(np.abs(cum_defaults - cum_non_defaults))
        
        ax.plot(proba_sorted, cum_defaults, label=f'{name} - Defaulters', 
                color=color, lw=1.5)
        ax.plot(proba_sorted, cum_non_defaults, label=f'{name} - Non-defaulters',
                color=color, lw=1.5, ls='--')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Cumulative Distribution', fontsize=12)
    ax.set_title('KS Curves', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved KS curves to {save_path}")


def generate_all_figures(results_dict: Dict[str, np.ndarray],
                         y_test: np.ndarray,
                         output_dir: str = "figures"):
    """Generate all evaluation figures at once."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING ALL EVALUATION FIGURES")
    print("="*60)
    
    plot_roc_curves(results_dict, y_test, f"{output_dir}/fig_roc_curves.pdf")
    plot_pr_curves(results_dict, y_test, f"{output_dir}/fig_pr_curves.pdf")
    plot_calibration_curves(results_dict, y_test, f"{output_dir}/fig_calibration.pdf")
    plot_confusion_matrices(results_dict, y_test, f"{output_dir}/fig_confusion_matrices.pdf")
    plot_ks_curves(results_dict, y_test, f"{output_dir}/fig_ks_curves.pdf")
    
    print(f"\nAll figures saved to {output_dir}/")


def create_results_table(results_list: List[Dict]) -> pd.DataFrame:
    """Create publication-ready results table."""
    rows = []
    for r in results_list:
        row = {
            'Model': r.get('model', 'Unknown'),
            'AUC-ROC': f"{r.get('auc_roc', 0):.4f}",
            'PR-AUC': f"{r.get('pr_auc', 0):.4f}",
            'KS': f"{r.get('ks_statistic', 0):.4f}",
            'Brier': f"{r.get('brier_score', 0):.4f}",
            'F1': f"{r.get('f1', 0):.4f}",
            'F1-Default': f"{r.get('f1_default', 0):.4f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Highlight best in each column
    print("\n" + "="*60)
    print("PUBLICATION RESULTS TABLE")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save as CSV and LaTeX
    df.to_csv("results/table2_main_results.csv", index=False)
    print("\nSaved to results/table2_main_results.csv")
    
    # Generate LaTeX
    latex = df.to_latex(index=False, float_format="%.4f")
    with open("results/table2_main_results.tex", 'w') as f:
        f.write(latex)
    print("Saved LaTeX to results/table2_main_results.tex")
    
    return df
