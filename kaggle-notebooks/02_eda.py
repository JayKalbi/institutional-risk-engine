"""
===============================================================================
KAGGLE NOTEBOOK 2: Exploratory Data Analysis (EDA)
===============================================================================
Run this SECOND on Kaggle (CPU only - no GPU needed)

What this notebook does:
1. Loads the HMDA binary dataset from Notebook 1
2. Generates comprehensive EDA visualizations
3. Creates Table 1 (dataset statistics) for the research paper
4. Analyzes target distribution, feature correlations, missing data
5. Analyzes demographics for fairness audit preparation

PREREQUISITE: Run Notebook 1 first to generate hmda_binary.parquet
KAGGLE SETUP: Accelerator: None, Runtime: ~10-15 minutes
===============================================================================
"""

# =============================================================================
# CELL 1: Imports & Setup
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
})

# Paths
WORKING_DIR = Path("./output")
PROCESSED_DIR = WORKING_DIR / "data" / "processed"
FIGURES_DIR = WORKING_DIR / "figures"
RESULTS_DIR = WORKING_DIR / "results"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KAGGLE NOTEBOOK 2: Exploratory Data Analysis (EDA)")
print("=" * 70)

# =============================================================================
# CELL 2: Load Data
# =============================================================================

print("\nLoading data from Notebook 1...")
df = pd.read_parquet(PROCESSED_DIR / "hmda_binary.parquet")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Denial rate: {df['default'].mean():.2%}")

# =============================================================================
# CELL 3: Target Variable Distribution
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE 1: Target Variable Distribution")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall distribution
action_labels = {1: 'Originated', 3: 'Denied'}
df['action_label'] = df['action_taken'].map(action_labels)
value_counts = df['action_label'].value_counts()

colors = ['#70AD47', '#C00000']
axes[0].bar(value_counts.index, value_counts.values, color=colors, edgecolor='black')
axes[0].set_title('(a) Loan Application Outcomes', fontsize=12)
axes[0].set_ylabel('Count')
for i, (idx, val) in enumerate(value_counts.items()):
    axes[0].text(i, val + len(df)*0.01, f'{val:,}\n({val/len(df)*100:.1f}%)', 
                ha='center', fontsize=10)

# Default rate (denial rate)
default_rate = df['default'].mean()
axes[1].pie([1-default_rate, default_rate], 
           labels=['Originated', 'Denied'],
           colors=colors, autopct='%1.1f%%', startangle=90,
           explode=(0, 0.05))
axes[1].set_title('(b) Denial Rate', fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_target_distribution.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig1_target_distribution.png", bbox_inches='tight')
plt.show()
print(f"Saved to {FIGURES_DIR}/fig1_target_distribution.pdf")

# =============================================================================
# CELL 4: Numeric Features Distribution by Target
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE 2: Numeric Feature Distributions by Outcome")
print("=" * 70)

numeric_cols = ['loan_amount', 'income', 'loan_to_value_ratio', 
                'debt_to_income_ratio', 'property_value']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    if col not in df.columns:
        continue
    
    # Use log scale for monetary values
    data_originated = df[df['default'] == 0][col].dropna()
    data_denied = df[df['default'] == 1][col].dropna()
    
    axes[idx].hist(data_originated, bins=50, alpha=0.6, label='Originated', 
                  color='#70AD47', density=True)
    axes[idx].hist(data_denied, bins=50, alpha=0.6, label='Denied', 
                  color='#C00000', density=True)
    axes[idx].set_title(f'{col.replace("_", " ").title()}', fontsize=11)
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    
    # Add median lines
    axes[idx].axvline(data_originated.median(), color='#70AD47', ls='--', lw=1.5)
    axes[idx].axvline(data_denied.median(), color='#C00000', ls='--', lw=1.5)

# Remove empty subplot
axes[5].axis('off')

plt.suptitle('Feature Distributions by Loan Outcome', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_feature_distributions.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig2_feature_distributions.png", bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 5: Correlation Matrix
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE 3: Feature Correlation Matrix")
print("=" * 70)

# Select numeric features for correlation
numeric_features = [c for c in ['loan_amount', 'income', 'loan_to_value_ratio',
                                'debt_to_income_ratio', 'property_value', 'loan_term']
                   if c in df.columns]

corr_matrix = df[numeric_features + ['default']].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, vmin=-1, vmax=1, square=True,
           linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_correlation_matrix.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig3_correlation_matrix.png", bbox_inches='tight')
plt.show()
print(f"Saved to {FIGURES_DIR}/fig3_correlation_matrix.pdf")

# Print strongest correlations with default
print("\nCorrelations with default (denial):")
default_corr = corr_matrix['default'].drop('default').sort_values(key=abs, ascending=False)
for feat, corr in default_corr.items():
    print(f"  {feat:30s}: {corr:6.3f}")

# =============================================================================
# CELL 6: Denial Rate by Demographics
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE 4: Denial Rates by Demographics (Fairness Preview)")
print("=" * 70)

demo_cols = ['applicant_sex', 'applicant_ethnicity', 'applicant_age']
demo_labels = {
    'applicant_sex': {1: 'Male', 2: 'Female', 3: 'Joint', 4: 'Sex Not Available'},
    'applicant_ethnicity': {1: 'Hispanic/Latino', 2: 'Not Hispanic', 
                           3: 'Info not provided', 4: 'Not applicable'},
    'applicant_age': {}  # Already string
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(demo_cols):
    if col not in df.columns:
        axes[idx].text(0.5, 0.5, f'{col}\nnot available', ha='center', va='center')
        continue
    
    # Calculate denial rate by group
    denial_by_group = df.groupby(col)['default'].agg(['mean', 'count']).reset_index()
    denial_by_group = denial_by_group[denial_by_group['count'] >= 100]  # Filter small groups
    denial_by_group = denial_by_group.sort_values('mean', ascending=False)
    
    # Map labels
    if col in demo_labels and demo_labels[col]:
        denial_by_group[col] = denial_by_group[col].map(demo_labels[col])
    
    bars = axes[idx].bar(range(len(denial_by_group)), denial_by_group['mean'], 
                        color='#4472C4', edgecolor='black')
    axes[idx].set_xticks(range(len(denial_by_group)))
    axes[idx].set_xticklabels(denial_by_group[col], rotation=45, ha='right', fontsize=8)
    axes[idx].set_title(f'{col.replace("_", " ").title()}', fontsize=11)
    axes[idx].set_ylabel('Denial Rate')
    axes[idx].axhline(df['default'].mean(), color='red', ls='--', 
                     label=f'Overall ({df["default"].mean():.1%})')
    axes[idx].legend(fontsize=8)
    
    # Add value labels
    for bar, rate in zip(bars, denial_by_group['mean']):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{rate:.1%}', ha='center', fontsize=8)

plt.suptitle('Denial Rates by Demographic Groups', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_denial_by_demographics.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig4_denial_by_demographics.png", bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 7: Missing Data Analysis
# =============================================================================

print("\n" + "=" * 70)
print("FIGURE 5: Missing Data Analysis")
print("=" * 70)

missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]

if len(missing) > 0:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='barh', color='#ED7D31', edgecolor='black')
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Data by Column', fontsize=13)
    plt.axvline(40, color='red', ls='--', lw=1.5, label='40% threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_missing_data.pdf", bbox_inches='tight')
    plt.show()
    print(f"\nColumns with >40% missing: {(missing > 40).sum()}")
else:
    print("No missing values found!")

# =============================================================================
# CELL 8: Generate Table 1 (Dataset Statistics for Paper)
# =============================================================================

print("\n" + "=" * 70)
print("TABLE 1: Dataset Statistics (for Research Paper)")
print("=" * 70)

table1 = pd.DataFrame({
    'Dataset': ['HMDA 2022 (Primary)'],
    'Records': [f"{len(df):,}"],
    'Features': [f"{df.shape[1]}"],
    'Denial Rate': [f"{df['default'].mean():.2%}"],
    'Numeric Features': [f"{len(numeric_features)}"],
    'Categorical Features': [f"{len([c for c in df.columns if c not in numeric_features + ['default', 'action_taken', 'action_label']])}"],
    'Demographics': ['Sex, Race, Ethnicity, Age'],
    'Use Case': ['Primary training & fairness audit']
})

print("\n" + table1.to_string(index=False))

# Save
table1.to_csv(RESULTS_DIR / "table1_dataset_stats.csv", index=False)
latex_table = table1.style.hide(axis="index").to_latex()
with open(RESULTS_DIR / "table1_dataset_stats.tex", 'w') as f:
    f.write(latex_table)
print(f"\nSaved to {RESULTS_DIR}/table1_dataset_stats.csv and .tex")

# =============================================================================
# CELL 9: Key Statistics Summary
# =============================================================================

print("\n" + "=" * 70)
print("KEY STATISTICS SUMMARY")
print("=" * 70)

print(f"\nDataset Size: {len(df):,} applications")
print(f"Denial Rate: {df['default'].mean():.2%}")
print(f"Features: {df.shape[1]} total")

if 'loan_amount' in df.columns:
    print(f"\nLoan Amount: ${df['loan_amount'].median():,.0f} median")
if 'income' in df.columns:
    print(f"Income: ${df['income'].median():,.0f} median")
if 'debt_to_income_ratio' in df.columns:
    print(f"DTI: {df['debt_to_income_ratio'].median():.1f}% median")
if 'loan_to_value_ratio' in df.columns:
    print(f"LTV: {df['loan_to_value_ratio'].median():.1f}% median")

# Denial rate by key segments
print("\n--- Denial Rates by Key Segments ---")
if 'debt_to_income_ratio' in df.columns:
    high_dti = df['debt_to_income_ratio'] > 43
    print(f"High DTI (>43%): {df[high_dti]['default'].mean():.2%} ({high_dti.sum():,} apps)")
    print(f"Low DTI (<=43%): {df[~high_dti]['default'].mean():.2%} ({(~high_dti).sum():,} apps)")

if 'loan_to_value_ratio' in df.columns:
    high_ltv = df['loan_to_value_ratio'] > 80
    print(f"High LTV (>80%): {df[high_ltv]['default'].mean():.2%} ({high_ltv.sum():,} apps)")
    print(f"Low LTV (<=80%): {df[~high_ltv]['default'].mean():.2%} ({(~high_ltv).sum():,} apps)")

print("\n" + "=" * 70)
print("NOTEBOOK 2 COMPLETE!")
print("=" * 70)
print(f"\nFigures saved to: {FIGURES_DIR}/")
print(f"Next: Run Notebook 3 (Tabular Preprocessing)")
