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

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths for Kaggle
WORKING_DIR = Path("/kaggle/working")
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

"""
Download from Kaggle Dataset (RECOMMENDED)

The HMDA dataset is available as a Kaggle dataset.
Add this dataset to your notebook:
1. Click "Add Input" in the right panel
2. Search for "HMDA" or "Home Mortgage Disclosure Act"
3. Select the CFPB HMDA dataset (2025)

After adding, the data will be at: /kaggle/input/hmda-...
"""

# Try to find HMDA data in Kaggle input
KAGGLE_INPUT = Path("/kaggle/input")

print("--- Scanning Kaggle Input Directory ---")
if KAGGLE_INPUT.exists():
    # os.walk forces Python to look deep inside all nested folders
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        for file in files:
            if file.endswith(('.csv', '.parquet')):
                full_path = Path(root) / file
                print(f"Found File: {file}")
                print(f"Absolute Path: {full_path}\n")
else:
    print("⚠️ /kaggle/input directory does not exist!")

# Defining main working project directory
PROJECT_DIR = Path("/kaggle/working")

# Create a 'data' or 'raw' subdirectory structure
RAW_DIR = PROJECT_DIR / "data_raw"

# Safely generate the physical folders on disk if they don't exist yet
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Output directory initialized at: {RAW_DIR}")

# 1. DEFINE THE FUNCTION FIRST
def download_hmda_data(source="kaggle", sample_size=500000):
    """
    Loads HMDA data from Kaggle input or falls back to a sample.
    """
    # Note: Check this path. Your directory list cell showed empty output for 'hmda'
    kaggle_path = Path("/kaggle/input/datasets/jaykalbi/2025-hmda-nationwide-mortgage-lending-dataset/HMDA_year_2025.csv")
    
    if source == "kaggle":
        if not kaggle_path.exists():
            raise FileNotFoundError(f"Kaggle file not found at: {kaggle_path}")
        print(f"Reading data from {kaggle_path}...")
        return pd.read_csv(kaggle_path, nrows=sample_size, low_memory=False)
        
    elif source == "sample":
        print(f"Generating random sample dataframe of size {sample_size}...")
        mock_data = {
            'action_taken': np.random.choice([1, 2, 3], size=sample_size),
            'loan_amount': np.random.randint(50000, 500000, size=sample_size),
            'applicant_income': np.random.randint(30000, 200000, size=sample_size)
        }
        return pd.DataFrame(mock_data)
    else:
        raise ValueError("Invalid source. Choose 'kaggle' or 'sample'.")

# 2. RUN THE EXECUTION CODE SECOND
try:
    df = download_hmda_data(source="kaggle")
except Exception as e:
    print(f"Kaggle input not available: {e}")
    print("Using sample data for demonstration...")
    df = download_hmda_data(source="sample", sample_size=500000)

# Save raw data (Ensure RAW_DIR is defined above this cell)
df.to_parquet(RAW_DIR / "hmda_raw.parquet", index=False)
print(f"\nSaved raw data to {RAW_DIR}/hmda_raw.parquet")

print("\n" + "=" * 70)
print("DATA VERIFICATION")
print("=" * 70)

print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicated rows: {df.duplicated().sum():,}")

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

print("\n" + "=" * 70)
print("QUICK FEATURE PROFILE")
print("=" * 70)

# Numeric features
numeric_cols = ['loan_amount', 'income', 'loan_to_value_ratio', 
                'debt_to_income_ratio', 'property_value', 'loan_term']

print("\nNumeric Features Summary:")
for col in numeric_cols:
    if col in df.columns:
        # --- FIX: Convert column to numeric, forcing strings/text to NaN ---
        series = pd.to_numeric(df[col], errors='coerce')
        
        print(f"\n{col}:")
        print(f"  Mean: {series.mean():,.2f}")
        print(f"  Median: {series.median():,.2f}")
        print(f"  Min: {series.min():,.2f}")
        print(f"  Max: {series.max():,.2f}")
        print(f"  Missing/Exempt: {series.isnull().sum():,} ({series.isnull().mean():.1%})")

# Categorical features
cat_cols = ['loan_type', 'loan_purpose', 'lien_status', 'occupancy_type',
            'applicant_sex', 'applicant_ethnicity']

print("\nCategorical Features - Top Values:")
for col in cat_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head(5).to_string())


# Save the binary-focused dataset for next notebook
df_binary.to_parquet(PROCESSED_DIR / "hmda_binary.parquet", index=False)
print(f"\nSaved binary dataset to {PROCESSED_DIR}/hmda_binary.parquet")

print("\n" + "=" * 70)
print("Dataset Download COMPLETE!")
print("=" * 70)
print(f"\nNext: Run Notebook 2 (EDA) on this dataset")
print(f"File: {PROCESSED_DIR}/hmda_binary.parquet")
print(f"Shape: {df_binary.shape}")



