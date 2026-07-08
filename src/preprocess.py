"""
HMDA Dataset Preprocessing Pipeline
Run this on Kaggle (Notebook 3)

This module handles all tabular preprocessing for the CreditRisk-LLM project:
- Data cleaning and filtering
- Feature engineering
- Missing value imputation
- Encoding categorical variables
- SMOTE for class imbalance
- Train/validation/test temporal split
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - HMDA Dataset Specific Features
# =============================================================================

# Core numeric features from HMDA dataset
NUMERIC_FEATURES = [
    'loan_amount',
    'income', 
    'loan_to_value_ratio',
    'debt_to_income_ratio',
    'property_value',
    'loan_term',
]

# Categorical features
CATEGORICAL_FEATURES = [
    'loan_type',
    'loan_purpose', 
    'lien_status',
    'occupancy_type',
    'applicant_credit_score_type',
    'applicant_sex',
    'applicant_race',
    'applicant_ethnicity',
    'applicant_age',
]

# Target variable
TARGET_COL = 'action_taken'

# Leakage columns (post-application outcomes, not available at decision time)
LEAKAGE_COLS = [
    'action_taken_date',
    'denial_reason_1',
    'denial_reason_2', 
    'denial_reason_3',
    'denial_reason_4',
    'aus_1',
    'aus_2',
    'aus_3',
    'aus_4',
    'aus_5',
    'interest_rate',
    'rate_spread',
    'hoepa_status',
    'total loan costs',
    'total_points_and_fees',
    'origination_charges',
    'discount_points',
    'lender_credits',
    'loan_status',
    'preapproval',
]

# Demographic columns for fairness audit
DEMOGRAPHIC_COLS = [
    'applicant_sex',
    'applicant_race', 
    'applicant_ethnicity',
    'applicant_age',
]


def load_hmda_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load HMDA dataset from CSV/parquet.
    
    Args:
        filepath: Path to HMDA dataset file
        nrows: Number of rows to load (for testing)
    
    Returns:
        Raw DataFrame
    """
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False, nrows=nrows)
    
    print(f"Loaded HMDA data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean_hmda_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HMDA dataset: filter valid records, create binary target.
    
    HMDA action_taken codes:
    1 = Loan originated
    2 = Application approved but not accepted  
    3 = Application denied
    4 = Application withdrawn by applicant
    5 = File closed for incompleteness
    6 = Purchased loan
    7 = Preapproval request denied
    8 = Preapproval request approved but not accepted
    
    We focus on: 1 (originated/success) vs 3 (denied/failure)
    """
    print("=" * 60)
    print("STEP 1: Data Cleaning")
    print("=" * 60)
    
    # Keep only originated (1) and denied (3) applications
    df = df[df[TARGET_COL].isin([1, 3])].copy()
    print(f"After filtering to originated/denied: {len(df):,} rows")
    
    # Create binary target: 1 = denied (default/high-risk), 0 = originated (low-risk)
    df['default'] = (df[TARGET_COL] == 3).astype(int)
    print(f"Default (denial) rate: {df['default'].mean():.2%}")
    
    # Remove rows with missing target
    df = df.dropna(subset=['default'])
    
    # Drop obvious leakage columns
    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=leakage_present, errors='ignore')
    print(f"Dropped {len(leakage_present)} leakage columns")
    
    # Drop rows with missing key features (>30% missing in critical fields)
    critical_cols = ['loan_amount', 'income']
    df = df.dropna(subset=critical_cols)
    print(f"After dropping missing critical fields: {len(df):,} rows")
    
    # Filter outlier income (top/bottom 0.5%)
    income_low = df['income'].quantile(0.005)
    income_high = df['income'].quantile(0.995)
    df = df[(df['income'] >= income_low) & (df['income'] <= income_high)]
    print(f"After income outlier removal: {len(df):,} rows")
    
    # Filter reasonable loan amounts
    df = df[(df['loan_amount'] >= 10000) & (df['loan_amount'] <= 2000000)]
    print(f"After loan amount filtering: {len(df):,} rows")
    
    print(f"\nFinal clean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Default rate: {df['default'].mean():.2%}")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for credit risk modeling.
    
    Key engineered features:
    - loan_to_income_ratio: Loan amount relative to annual income
    - payment_burden: Estimated monthly payment as % of monthly income
    - income_per_dependent: Income adjusted for family size
    - loan_value_ratio_safe: Binary flag for LTV < 80%
    - high_dti_flag: Binary flag for DTI > 43% (QM threshold)
    - income_category: Binned income levels
    """
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    
    df = df.copy()
    
    # 1. Loan-to-income ratio (higher = more risk)
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
    print(f"Created: loan_to_income_ratio")
    
    # 2. Estimated monthly payment burden
    # Approximate monthly payment using standard mortgage formula
    # P * (r(1+r)^n) / ((1+r)^n - 1) where r = rate/12, n = term
    # Using average rate of 6.5% for approximation
    rate = 0.065 / 12
    if 'loan_term' in df.columns:
        n_months = df['loan_term'].fillna(360)
    else:
        n_months = 360  # Default 30-year
    
    monthly_payment = df['loan_amount'] * (rate * (1 + rate)**n_months) / ((1 + rate)**n_months - 1)
    monthly_income = df['income'] / 12
    df['payment_burden'] = monthly_payment / (monthly_income + 1)
    print(f"Created: payment_burden")
    
    # 3. High DTI flag (>43% is Qualified Mortgage threshold)
    if 'debt_to_income_ratio' in df.columns:
        df['high_dti_flag'] = (df['debt_to_income_ratio'].fillna(0) > 43).astype(int)
        print(f"Created: high_dti_flag")
    
    # 4. Safe LTV flag (<80% avoids PMI, lower risk)
    if 'loan_to_value_ratio' in df.columns:
        df['safe_ltv_flag'] = (df['loan_to_value_ratio'].fillna(100) < 80).astype(int)
        print(f"Created: safe_ltv_flag")
    
    # 5. Income category
    income_median = df['income'].median()
    income_q75 = df['income'].quantile(0.75)
    df['income_category'] = pd.cut(
        df['income'],
        bins=[0, income_median, income_q75, float('inf')],
        labels=['low', 'medium', 'high']
    )
    print(f"Created: income_category")
    
    # 6. Loan amount category
    df['loan_size_category'] = pd.cut(
        df['loan_amount'],
        bins=[0, 250000, 500000, float('inf')],
        labels=['small', 'medium', 'jumbo']
    )
    print(f"Created: loan_size_category")
    
    # 7. Credit score proxy category (if available)
    if 'applicant_credit_score_type' in df.columns:
        df['has_credit_score'] = df['applicant_credit_score_type'].notna().astype(int)
        print(f"Created: has_credit_score")
    
    # 8. Interaction: DTI x LTV (multiplicative risk factor)
    if 'debt_to_income_ratio' in df.columns and 'loan_to_value_ratio' in df.columns:
        df['dti_ltv_interaction'] = (
            df['debt_to_income_ratio'].fillna(43) * 
            df['loan_to_value_ratio'].fillna(80) / 100
        )
        print(f"Created: dti_ltv_interaction")
    
    print(f"\nFeature engineering complete. Shape: {df.shape}")
    return df


def prepare_tabular_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Select and prepare tabular features for modeling.
    
    Returns:
        DataFrame with selected features, numeric columns, categorical columns
    """
    print("\n" + "=" * 60)
    print("STEP 3: Feature Selection")
    print("=" * 60)
    
    # All potential features
    available_numeric = [c for c in NUMERIC_FEATURES + [
        'loan_to_income_ratio', 'payment_burden', 
        'high_dti_flag', 'safe_ltv_flag',
        'dti_ltv_interaction', 'has_credit_score'
    ] if c in df.columns]
    
    available_categorical = [c for c in CATEGORICAL_FEATURES + [
        'income_category', 'loan_size_category'
    ] if c in df.columns]
    
    feature_cols = available_numeric + available_categorical
    
    # Keep only features that exist in dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Keep features + target
    keep_cols = feature_cols + ['default'] + DEMOGRAPHIC_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    df_model = df[keep_cols].copy()
    
    print(f"Selected {len(available_numeric)} numeric features: {available_numeric}")
    print(f"Selected {len(available_categorical)} categorical features: {available_categorical}")
    print(f"Final modeling dataset: {df_model.shape}")
    
    return df_model, available_numeric, available_categorical


def temporal_split(df: pd.DataFrame, 
                   date_col: Optional[str] = None,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (if date available) or randomly with fixed seed.
    
    Args:
        df: Input dataframe
        date_col: Column with date information (if available)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        train, val, test dataframes
    """
    print("\n" + "=" * 60)
    print("STEP 4: Train/Validation/Test Split")
    print("=" * 60)
    
    if date_col and date_col in df.columns:
        # Temporal split
        df = df.sort_values(date_col)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        print(f"Using TEMPORAL split on {date_col}")
    else:
        # Random split with fixed seed
        from sklearn.model_selection import train_test_split
        train, temp = train_test_split(df, test_size=(1-train_ratio), random_state=42, stratify=df['default'])
        val, test = train_test_split(temp, test_size=(1-train_ratio-val_ratio)/(1-train_ratio), 
                                      random_state=42, stratify=temp['default'])
        print("Using RANDOM stratified split (no date column)")
    
    print(f"Train: {len(train):,} ({len(train)/len(df)*100:.1f}%) - Default rate: {train['default'].mean():.2%}")
    print(f"Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%) - Default rate: {val['default'].mean():.2%}")
    print(f"Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%) - Default rate: {test['default'].mean():.2%}")
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def encode_and_impute(train: pd.DataFrame, 
                      val: pd.DataFrame, 
                      test: pd.DataFrame,
                      numeric_cols: List[str],
                      categorical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Encode categoricals and impute missing values.
    Fit on train only, transform all sets.
    
    Returns:
        X_train, X_val, X_test, feature_names
    """
    print("\n" + "=" * 60)
    print("STEP 5: Encoding & Imputation")
    print("=" * 60)
    
    train = train.copy()
    val = val.copy()
    test = test.copy()
    
    # One-hot encode categoricals
    all_cat_cols = [c for c in categorical_cols if c in train.columns]
    if all_cat_cols:
        train = pd.get_dummies(train, columns=all_cat_cols, drop_first=True)
        val = pd.get_dummies(val, columns=all_cat_cols, drop_first=True)
        test = pd.get_dummies(test, columns=all_cat_cols, drop_first=True)
        
        # Align columns
        val = val.reindex(columns=train.columns, fill_value=0)
        test = test.reindex(columns=train.columns, fill_value=0)
        print(f"One-hot encoded {len(all_cat_cols)} categorical columns")
    
    # Get final feature columns (exclude target and demographics)
    exclude = ['default'] + [c for c in DEMOGRAPHIC_COLS if c in train.columns]
    feature_names = [c for c in train.columns if c not in exclude]
    
    # Impute numerics with median (fit on train only)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(train[feature_names])
    X_val = imputer.transform(val[feature_names])
    X_test = imputer.transform(test[feature_names])
    
    print(f"Imputed missing values with median strategy")
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Total features: {len(feature_names)}")
    
    return X_train, X_val, X_test, feature_names


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, 
                sampling_strategy: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE + Tomek links for class imbalance.
    ONLY apply to training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Ratio of minority to majority after resampling
        random_state: Random seed
    
    Returns:
        Resampled X_train, y_train
    """
    print("\n" + "=" * 60)
    print("STEP 6: SMOTE Resampling")
    print("=" * 60)
    
    print(f"Before SMOTE: {X_train.shape}, default rate: {y_train.mean():.2%}")
    
    smote = SMOTETomek(random_state=random_state, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {X_res.shape}, default rate: {y_res.mean():.2%}")
    
    return X_res, y_res


def full_preprocessing_pipeline(df: pd.DataFrame, 
                                apply_smote_flag: bool = True) -> Dict:
    """
    Run the complete preprocessing pipeline.
    
    Returns:
        Dictionary with all preprocessed data and metadata
    """
    print("\n" + "=" * 70)
    print("FULL PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Clean
    df_clean = clean_hmda_data(df)
    
    # Step 2: Engineer features
    df_engineered = engineer_features(df_clean)
    
    # Step 3: Select features
    df_model, numeric_cols, categorical_cols = prepare_tabular_features(df_engineered)
    
    # Step 4: Split
    train, val, test = temporal_split(df_model)
    
    # Step 5: Encode & impute
    X_train, X_val, X_test, feature_names = encode_and_impute(
        train, val, test, numeric_cols, categorical_cols
    )
    
    y_train = train['default'].values
    y_val = val['default'].values
    y_test = test['default'].values
    
    # Step 6: SMOTE
    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # Extract demographic info for fairness audit
    demo_test = test[[c for c in DEMOGRAPHIC_COLS if c in test.columns]].copy() if any(c in test.columns for c in DEMOGRAPHIC_COLS) else None
    
    result = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_names,
        'train_df': train, 'val_df': val, 'test_df': test,
        'demographic_test': demo_test,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    
    return result


def save_processed_data(result: Dict, output_dir: str = "data/processed"):
    """Save all processed datasets to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(f"{output_dir}/X_train.npy", result['X_train'])
    np.save(f"{output_dir}/y_train.npy", result['y_train'])
    np.save(f"{output_dir}/X_val.npy", result['X_val'])
    np.save(f"{output_dir}/y_val.npy", result['y_val'])
    np.save(f"{output_dir}/X_test.npy", result['X_test'])
    np.save(f"{output_dir}/y_test.npy", result['y_test'])
    
    # Save feature names
    import json
    with open(f"{output_dir}/feature_names.json", 'w') as f:
        json.dump(result['feature_names'], f)
    
    # Save dataframes
    result['test_df'].to_csv(f"{output_dir}/test_df.csv", index=False)
    if result['demographic_test'] is not None:
        result['demographic_test'].to_csv(f"{output_dir}/demographic_test.csv", index=False)
    
    print(f"\nSaved all processed data to {output_dir}/")
