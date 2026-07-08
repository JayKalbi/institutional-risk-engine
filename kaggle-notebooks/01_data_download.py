"""
===============================================================================
KAGGLE NOTEBOOK 1: HMDA Data Download & Setup
===============================================================================
Run this FIRST on Kaggle (CPU only - no GPU needed)

What this notebook does:
1. Downloads the HMDA (Home Mortgage Disclosure Act) dataset
2. Verifies the download and shows basic info
3. Saves data to /kaggle/working/ for subsequent notebooks

HMDA Dataset Info:
- Source: Consumer Financial Protection Bureau (CFPB)
- Contains: Mortgage application data with demographics
- Size: ~20M+ records annually (we use a subset)
- Key features: Loan amount, income, DTI, LTV, demographics, action taken
- Target: action_taken (1=originated, 3=denied)

KAGGLE SETUP:
- Create new notebook on kaggle.com
- Set Accelerator: None (CPU)
- Runtime: ~5-10 minutes
===============================================================================
"""

# =============================================================================
# CELL 1: Install Requirements & Imports
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Auto-detect if running on Kaggle or locally
if Path('/kaggle/working').exists():
    WORKING_DIR = Path("/kaggle/working")
    KAGGLE_INPUT = Path("/kaggle/input/datasets/jaykalbi/2025-hmda-nationwide-mortgage-lending-dataset/HMDA_year_2025.csv")
    print(f"🌍 Detected Kaggle environment. Working dir: {WORKING_DIR}")
else:
    WORKING_DIR = Path("./output")
    KAGGLE_INPUT = Path("./input/HMDA_year_2025.csv")
    print(f"💻 Detected Local environment. Working dir: {WORKING_DIR}")

DATA_DIR = WORKING_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = WORKING_DIR / "results"

# Create directories
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KAGGLE NOTEBOOK 1: HMDA Data Download & Setup")
print("=" * 70)
print(f"Working directory: {WORKING_DIR}")
print(f"Data will be saved to: {RAW_DIR}")

# =============================================================================
# CELL 2: Download HMDA Dataset
# =============================================================================

"""
OPTION A: Download from Kaggle Dataset (RECOMMENDED)

The HMDA dataset is available as a Kaggle dataset.
Add this dataset to your notebook:
1. Click "Add Input" in the right panel
2. Search for "HMDA" or "Home Mortgage Disclosure Act"
3. Select the CFPB HMDA dataset (latest year)

After adding, the data will be at: /kaggle/input/hmda-...
"""

# (Legacy code removed - logic handled in download_hmda_data)

"""
OPTION B: Download via URL (if not available as Kaggle dataset)

We'll download from the CFPB HMDA Platform.
Note: The full dataset is very large. We'll use a sampled subset.
"""

# Download HMDA LAR (Loan Application Register) data
# Using a recent year (2022) for analysis
HMDA_URL = "https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data/2022/2022_public_lar_csv.zip"

# Alternative: Use smaller test file or sample
# For this project, we'll create a function that can handle multiple sources

def download_hmda_data(source="kaggle", sample_size=None):
    """
    Download HMDA data from Kaggle or fallback to sample.
    """
    print(f"\nLoading HMDA data (source={source})...")
    
    if source == "kaggle":
        if not KAGGLE_INPUT.exists():
            raise FileNotFoundError(f"Kaggle file not found at: {KAGGLE_INPUT}")
        
        print(f"Reading data from {KAGGLE_INPUT}...")
        df = pd.read_csv(KAGGLE_INPUT, nrows=sample_size, low_memory=False)
        print(f"Loaded from Kaggle input: {KAGGLE_INPUT.name}")
    
    elif source == "sample":
        # For demonstration, create a representative sample structure
        # In practice, replace this with actual data download
        print("Creating sample dataset for demonstration...")
        print("NOTE: Replace with actual HMDA data download for production use")
        
        np.random.seed(42)
        n = 500000  # 500K sample
        
        df = pd.DataFrame({
            'loan_amount': np.random.lognormal(12.2, 0.5, n).astype(int),
            'income': np.random.lognormal(11.2, 0.8, n).astype(int),
            'loan_to_value_ratio': np.random.normal(80, 15, n).clip(10, 150),
            'debt_to_income_ratio': np.random.normal(36, 12, n).clip(0, 60),
            'property_value': np.random.lognormal(12.5, 0.6, n).astype(int),
            'loan_term': np.random.choice([180, 240, 360], n, p=[0.05, 0.15, 0.8]),
            'loan_type': np.random.choice([1, 2, 3, 4], n, p=[0.85, 0.10, 0.03, 0.02]),
            'loan_purpose': np.random.choice([1, 2, 31, 32, 4, 5], n, p=[0.45, 0.35, 0.08, 0.07, 0.03, 0.02]),
            'lien_status': np.random.choice([1, 2], n, p=[0.75, 0.25]),
            'occupancy_type': np.random.choice([1, 2, 3], n, p=[0.88, 0.10, 0.02]),
            'applicant_credit_score_type': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 1111], n),
            'applicant_sex': np.random.choice([1, 2, 3, 4, 6], n, p=[0.45, 0.48, 0.03, 0.02, 0.02]),
            'applicant_race': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 41, 42, 43, 44, 45, 46, 47, 48], n),
            'applicant_ethnicity': np.random.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.78, 0.05, 0.03, 0.02]),
            'applicant_age': np.random.choice(['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74'], 
                                              n, p=[0.08, 0.22, 0.28, 0.22, 0.12, 0.06, 0.02]),
            'action_taken': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n, p=[0.55, 0.05, 0.15, 0.15, 0.05, 0.03, 0.01, 0.01]),
        })
        
        # Make target more realistic (denial correlated with risk factors)
        risk_score = (
            (df['debt_to_income_ratio'] > 43).astype(float) * 0.3 +
            (df['loan_to_value_ratio'] > 90).astype(float) * 0.25 +
            (df['loan_amount'] > df['loan_amount'].quantile(0.8)).astype(float) * 0.15 +
            (df['income'] < df['income'].quantile(0.2)).astype(float) * 0.3
        )
        
        # Adjust action_taken based on risk
        deny_prob = risk_score.clip(0, 0.7)
        random_draw = np.random.random(n)
        
        # Replace some originated (1) with denied (3) based on risk
        high_risk_mask = (deny_prob > 0.4) & (random_draw < deny_prob) & (df['action_taken'] == 1)
        df.loc[high_risk_mask, 'action_taken'] = 3
        
        print(f"Created realistic sample with {df['action_taken'].value_counts().to_dict()}")
    
    else:
        raise ValueError(f"Unknown source: {source}")
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {sample_size:,} records")
    
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# =============================================================================
# CELL 3: Load Data
# =============================================================================

# Try loading from Kaggle input first, fall back to sample
try:
    df = download_hmda_data(source="kaggle")
except Exception as e:
    print(f"Kaggle input not available: {e}")
    print("Using sample data for demonstration...")
    df = download_hmda_data(source="sample", sample_size=500000)

# Save raw data
df.to_parquet(RAW_DIR / "hmda_raw.parquet", index=False)
print(f"\nSaved raw data to {RAW_DIR}/hmda_raw.parquet")

# =============================================================================
# CELL 4: Data Verification & Basic Info
# =============================================================================

print("\n" + "=" * 70)
print("DATA VERIFICATION")
print("=" * 70)

print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicated rows: {df.duplicated().sum():,}")

# =============================================================================
# CELL 5: Target Variable Analysis
# =============================================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE ANALYSIS")
print("=" * 70)

# HMDA action_taken codes:
action_codes = {
    1: 'Loan originated',
    2: 'Application approved but not accepted',
    3: 'Application denied',
    4: 'Application withdrawn by applicant',
    5: 'File closed for incompleteness',
    6: 'Purchased loan',
    7: 'Preapproval request denied',
    8: 'Preapproval request approved but not accepted'
}

print("\nAction Taken Distribution:")
action_counts = df['action_taken'].value_counts().sort_index()
for code, count in action_counts.items():
    desc = action_codes.get(code, 'Unknown')
    pct = count / len(df) * 100
    print(f"  {code}: {desc:45s} | {count:>10,} ({pct:5.1f}%)")

# Focus on originated (1) vs denied (3)
print("\n--- Binary Target (Originated vs Denied) ---")
df_binary = df[df['action_taken'].isin([1, 3])].copy()
df_binary['default'] = (df_binary['action_taken'] == 3).astype(int)
print(f"Records with binary outcome: {len(df_binary):,}")
print(f"Denial rate: {df_binary['default'].mean():.2%}")

# Save metadata
metadata = {
    'dataset': 'HMDA',
    'total_records': int(len(df)),
    'binary_records': int(len(df_binary)),
    'denial_rate': float(df_binary['default'].mean()),
    'columns': list(df.columns),
    'action_taken_distribution': {str(k): int(v) for k, v in action_counts.items()},
    'data_year': 2022,
    'download_date': pd.Timestamp.now().strftime('%Y-%m-%d')
}

with open(RESULTS_DIR / "download_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved metadata to {RESULTS_DIR}/download_metadata.json")

# =============================================================================
# CELL 6: Quick Profile of Key Features
# =============================================================================

print("\n" + "=" * 70)
print("QUICK FEATURE PROFILE")
print("=" * 70)

# Numeric features
numeric_cols = ['loan_amount', 'income', 'loan_to_value_ratio', 
                'debt_to_income_ratio', 'property_value', 'loan_term']

print("\nNumeric Features Summary:")
for col in numeric_cols:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        print(f"\n{col}:")
        print(f"  Mean: {series.mean():,.2f}")
        print(f"  Median: {series.median():,.2f}")
        print(f"  Min: {series.min():,.2f}")
        print(f"  Max: {series.max():,.2f}")
        print(f"  Missing: {series.isnull().sum():,} ({series.isnull().mean():.1%})")

# Categorical features
cat_cols = ['loan_type', 'loan_purpose', 'lien_status', 'occupancy_type',
            'applicant_sex', 'applicant_ethnicity']

print("\nCategorical Features - Top Values:")
for col in cat_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head(5).to_string())

# =============================================================================
# CELL 7: Save Cleaned Binary Dataset
# =============================================================================

# Save the binary-focused dataset for next notebook
df_binary.to_parquet(PROCESSED_DIR / "hmda_binary.parquet", index=False)
print(f"\nSaved binary dataset to {PROCESSED_DIR}/hmda_binary.parquet")

print("\n" + "=" * 70)
print("NOTEBOOK 1 COMPLETE!")
print("=" * 70)
print(f"\nNext: Run Notebook 2 (EDA) on this dataset")
print(f"File: {PROCESSED_DIR}/hmda_binary.parquet")
print(f"Shape: {df_binary.shape}")
