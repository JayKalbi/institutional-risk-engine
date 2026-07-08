"""
===============================================================================
KAGGLE NOTEBOOK 3: Tabular Preprocessing & Feature Engineering
===============================================================================
Run this THIRD on Kaggle (CPU only - no GPU needed)

What this notebook does:
1. Cleans HMDA data (filters, handles leakage)
2. Engineers 8 novel risk features
3. Encodes categoricals, imputes missing values
4. Applies SMOTE for class imbalance
5. Creates train/val/test split
6. Saves processed arrays for model training

PREREQUISITE: Notebook 1 (data download)
KAGGLE SETUP: Accelerator: None, Runtime: ~10-15 minutes
===============================================================================
"""

# =============================================================================
# CELL 1: Imports & Setup
# =============================================================================

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

WORKING_DIR = Path("./output")
PROCESSED_DIR = WORKING_DIR / "data" / "processed"
RESULTS_DIR = WORKING_DIR / "results"
SRC_DIR = WORKING_DIR / "src"

# Create directories
for d in [PROCESSED_DIR, RESULTS_DIR, SRC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KAGGLE NOTEBOOK 3: Tabular Preprocessing & Feature Engineering")
print("=" * 70)

# =============================================================================
# CELL 2: Load Raw Data
# =============================================================================

print("\nLoading HMDA data...")
df = pd.read_parquet(PROCESSED_DIR / "hmda_binary.parquet")
print(f"Raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Denial rate: {df['default'].mean():.2%}")

# =============================================================================
# CELL 3: Data Cleaning
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: DATA CLEANING")
print("=" * 70)

# Keep only originated (1) and denied (3)
df = df[df['action_taken'].isin([1, 3])].copy()
df['default'] = (df['action_taken'] == 3).astype(int)
print(f"After filtering to originated/denied: {len(df):,} rows")

# Drop leakage columns (post-decision information)
leakage_cols = [
    'action_taken_date', 'denial_reason_1', 'denial_reason_2',
    'denial_reason_3', 'denial_reason_4', 'aus_1', 'aus_2', 'aus_3',
    'aus_4', 'aus_5', 'interest_rate', 'rate_spread', 'hoepa_status',
    'preapproval', 'loan_status'
]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')
print(f"Dropped {len([c for c in leakage_cols if c in df.columns])} leakage columns")

# Drop rows with missing critical features
critical_cols = ['loan_amount', 'income']
df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
print(f"After dropping missing critical fields: {len(df):,} rows")

# Remove income outliers (top/bottom 0.5%)
income_low = df['income'].quantile(0.005)
income_high = df['income'].quantile(0.995)
df = df[(df['income'] >= income_low) & (df['income'] <= income_high)]
print(f"After income outlier removal: {len(df):,} rows")

# Filter reasonable loan amounts
df = df[(df['loan_amount'] >= 10000) & (df['loan_amount'] <= 2000000)]
print(f"After loan amount filtering: {len(df):,} rows")

print(f"\nClean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Denial rate: {df['default'].mean():.2%}")

# =============================================================================
# CELL 4: Feature Engineering (8 Novel Features)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING (8 Novel Features)")
print("=" * 70)

# Feature 1: Loan-to-income ratio
df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
print(f"[1] loan_to_income_ratio: mean={df['loan_to_income_ratio'].mean():.3f}")

# Feature 2: Estimated monthly payment burden
rate = 0.065 / 12  # 6.5% annual rate
n_months = df['loan_term'].fillna(360)
monthly_payment = df['loan_amount'] * (rate * (1 + rate)**n_months) / ((1 + rate)**n_months - 1)
monthly_income = df['income'] / 12
df['payment_burden'] = monthly_payment / (monthly_income + 1)
print(f"[2] payment_burden: mean={df['payment_burden'].mean():.3f}")

# Feature 3: High DTI flag (>43% = Qualified Mortgage threshold)
df['high_dti_flag'] = (df['debt_to_income_ratio'].fillna(0) > 43).astype(int)
print(f"[3] high_dti_flag: {df['high_dti_flag'].mean():.2%} flagged")

# Feature 4: Safe LTV flag (<80% avoids PMI, lower risk)
df['safe_ltv_flag'] = (df['loan_to_value_ratio'].fillna(100) < 80).astype(int)
print(f"[4] safe_ltv_flag: {df['safe_ltv_flag'].mean():.2%} safe")

# Feature 5: Income category
income_median = df['income'].median()
income_q75 = df['income'].quantile(0.75)
df['income_category'] = pd.cut(
    df['income'],
    bins=[0, income_median, income_q75, float('inf')],
    labels=['low', 'medium', 'high']
).astype(str)
print(f"[5] income_category: {df['income_category'].value_counts().to_dict()}")

# Feature 6: Loan size category
df['loan_size_category'] = pd.cut(
    df['loan_amount'],
    bins=[0, 250000, 500000, float('inf')],
    labels=['small', 'medium', 'jumbo']
).astype(str)
print(f"[6] loan_size_category: {df['loan_size_category'].value_counts().to_dict()}")

# Feature 7: Has credit score
df['has_credit_score'] = df['applicant_credit_score_type'].notna().astype(int)
print(f"[7] has_credit_score: {df['has_credit_score'].mean():.2%} have score")

# Feature 8: DTI x LTV interaction (multiplicative risk)
df['dti_ltv_interaction'] = (
    df['debt_to_income_ratio'].fillna(43) * 
    df['loan_to_value_ratio'].fillna(80) / 100
)
print(f"[8] dti_ltv_interaction: mean={df['dti_ltv_interaction'].mean():.1f}")

# =============================================================================
# CELL 5: Feature Selection
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: FEATURE SELECTION")
print("=" * 70)

# Numeric features
numeric_features = [c for c in [
    'loan_amount', 'income', 'loan_to_value_ratio',
    'debt_to_income_ratio', 'property_value', 'loan_term',
    'loan_to_income_ratio', 'payment_burden',
    'high_dti_flag', 'safe_ltv_flag',
    'has_credit_score', 'dti_ltv_interaction'
] if c in df.columns]

# Categorical features
categorical_features = [c for c in [
    'loan_type', 'loan_purpose', 'lien_status', 'occupancy_type',
    'applicant_credit_score_type', 'applicant_sex',
    'applicant_race', 'applicant_ethnicity', 'applicant_age',
    'income_category', 'loan_size_category'
] if c in df.columns]

# Demographic columns (for fairness audit, not modeling)
demographic_cols = [c for c in [
    'applicant_sex', 'applicant_race', 
    'applicant_ethnicity', 'applicant_age'
] if c in df.columns]

feature_cols = numeric_features + categorical_features
all_cols = feature_cols + ['default'] + demographic_cols
all_cols = [c for c in all_cols if c in df.columns]
all_cols = list(dict.fromkeys(all_cols))  # Remove duplicates to prevent get_dummies errors

df_model = df[all_cols].copy()

print(f"Selected {len(numeric_features)} numeric features: {numeric_features}")
print(f"Selected {len(categorical_features)} categorical features: {categorical_features}")
print(f"Demographic columns (for audit): {demographic_cols}")
print(f"Modeling dataset: {df_model.shape}")

# =============================================================================
# CELL 6: Train/Val/Test Split
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: TRAIN/VAL/TEST SPLIT (Stratified)")
print("=" * 70)

from sklearn.model_selection import train_test_split

train, temp = train_test_split(df_model, test_size=0.30, random_state=42, 
                                stratify=df_model['default'])
val, test = train_test_split(temp, test_size=0.50, random_state=42,
                              stratify=temp['default'])

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

print(f"Train: {len(train):,} ({len(train)/len(df_model)*100:.1f}%) - Denial: {train['default'].mean():.2%}")
print(f"Val:   {len(val):,} ({len(val)/len(df_model)*100:.1f}%) - Denial: {val['default'].mean():.2%}")
print(f"Test:  {len(test):,} ({len(test)/len(df_model)*100:.1f}%) - Denial: {test['default'].mean():.2%}")

# =============================================================================
# CELL 7: Encoding & Imputation
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: ENCODING & IMPUTATION")
print("=" * 70)

from sklearn.impute import SimpleImputer

# One-hot encode categoricals
train_enc = pd.get_dummies(train, columns=categorical_features, drop_first=True)
val_enc = pd.get_dummies(val, columns=categorical_features, drop_first=True)
test_enc = pd.get_dummies(test, columns=categorical_features, drop_first=True)

# Align columns
val_enc = val_enc.reindex(columns=train_enc.columns, fill_value=0)
test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

# Get feature names (exclude target and demographics)
exclude_cols = ['default'] + demographic_cols
feature_names = [c for c in train_enc.columns if c not in exclude_cols]

# Impute with median
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(train_enc[feature_names])
X_val = imputer.transform(val_enc[feature_names])
X_test = imputer.transform(test_enc[feature_names])

y_train = train_enc['default'].values
y_val = val_enc['default'].values
y_test = test_enc['default'].values

print(f"Imputed missing values with median")
print(f"Feature matrix: {X_train.shape}")
print(f"Total features: {len(feature_names)}")

# =============================================================================
# CELL 8: SMOTE for Class Imbalance
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: SMOTE RESAMPLING")
print("=" * 70)

from imblearn.combine import SMOTETomek

print(f"Before SMOTE: {X_train.shape}, denial rate: {y_train.mean():.2%}")

smote = SMOTETomek(random_state=42, sampling_strategy='auto')
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {X_train_res.shape}, denial rate: {y_train_res.mean():.2%}")

# =============================================================================
# CELL 9: Save Processed Data
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: SAVING PROCESSED DATA")
print("=" * 70)

# Save numpy arrays
np.save(PROCESSED_DIR / "X_train.npy", X_train_res)
np.save(PROCESSED_DIR / "y_train.npy", y_train_res)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

# Save feature names
with open(PROCESSED_DIR / "feature_names.json", 'w') as f:
    json.dump(feature_names, f)

# Save test dataframe (with demographics)
test.to_csv(PROCESSED_DIR / "test_df.csv", index=False)

# Save imputer
import joblib
joblib.dump(imputer, PROCESSED_DIR / "imputer.joblib")

print(f"\nSaved all processed data to {PROCESSED_DIR}/")

# Save metadata
metadata = {
    'n_train': int(len(y_train_res)),
    'n_val': int(len(y_val)),
    'n_test': int(len(y_test)),
    'n_features': int(len(feature_names)),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'feature_names': feature_names,
    'denial_rate_train': float(y_train_res.mean()),
    'denial_rate_val': float(y_val.mean()),
    'denial_rate_test': float(y_test.mean()),
    'smote_sampling_strategy': 'auto',
}

with open(RESULTS_DIR / "preprocessing_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to {RESULTS_DIR}/preprocessing_metadata.json")

print("\n" + "=" * 70)
print("NOTEBOOK 3 COMPLETE!")
print("=" * 70)
print(f"\nProcessed arrays saved:")
print(f"  X_train: {X_train_res.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"\nNext: Run Notebook 4 (Classical ML Baselines)")
