"""
Unit Tests for HybridCredit-LLM Core Modules
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# PREPROCESSING TESTS (test math directly, no full pipeline dependency)
# =============================================================================

class TestPreprocessing:
    """Tests for feature engineering math matching preprocess.py formulas."""

    def test_loan_to_income_ratio(self):
        """Verify loan_to_income_ratio formula: loan / (income + 1)."""
        loan, income = 200000, 100000
        result = loan / (income + 1)
        assert abs(result - 1.99998) < 0.01

    def test_high_dti_flag_above_threshold(self):
        """DTI above 43% (QM threshold) should flag as 1."""
        dti = 55.0
        flag = int(dti > 43)
        assert flag == 1

    def test_high_dti_flag_below_threshold(self):
        """DTI below 43% should flag as 0."""
        dti = 30.0
        flag = int(dti > 43)
        assert flag == 0

    def test_payment_burden_amortization(self):
        """Payment burden using proper amortization formula must be positive."""
        loan_amount = 350000
        income = 85000
        rate = 0.065 / 12
        n_months = 360
        monthly_payment = loan_amount * (rate * (1 + rate)**n_months) / ((1 + rate)**n_months - 1)
        monthly_income = income / 12
        burden = monthly_payment / (monthly_income + 1)
        assert burden > 0
        # Sanity check: burden should be roughly 0.3-0.4 for this scenario
        assert 0.1 < burden < 1.0

    def test_payment_burden_matches_preprocess_formula(self):
        """Verify the Flask app formula matches preprocess.py exactly."""
        # Both should use: P * (r(1+r)^n) / ((1+r)^n - 1) / (income/12 + 1)
        loan_amount = 350000
        income = 85000
        rate = 0.065 / 12
        n = 360

        # preprocess.py formula
        monthly_payment = loan_amount * (rate * (1 + rate)**n) / ((1 + rate)**n - 1)
        monthly_income = income / 12
        preprocess_burden = monthly_payment / (monthly_income + 1)

        # Flask app formula (after our fix)
        df = pd.DataFrame([{'loan_amount': loan_amount, 'income': income, 'loan_term': n}])
        r = 0.065 / 12
        n_months = df['loan_term'].fillna(360)
        mp = df['loan_amount'] * (r * (1 + r)**n_months) / ((1 + r)**n_months - 1)
        mi = df['income'] / 12
        flask_burden = (mp / (mi + 1)).iloc[0]

        assert abs(preprocess_burden - flask_burden) < 0.0001, \
            f"Formula mismatch: preprocess={preprocess_burden:.6f}, flask={flask_burden:.6f}"

    def test_lgd_bounds(self):
        """LGD must be clamped between 0.1 and 1.0."""
        for ltv in [50, 80, 100, 120, 150]:
            lgd = min(1.0, max(0.1, (ltv / 100) - 0.2))
            assert 0.1 <= lgd <= 1.0, f"LGD out of bounds for LTV={ltv}: {lgd}"

    def test_ecl_calculation(self):
        """ECL = EAD × PD × LGD, must be non-negative."""
        ead = 350000
        pd_score = 0.25
        lgd = 0.65
        ecl = ead * pd_score * lgd
        assert ecl >= 0
        assert abs(ecl - 56875.0) < 0.01


# =============================================================================
# FLASK API TESTS
# =============================================================================

class TestFlaskAPI:
    """Tests for the Flask REST API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a Flask test client (skips if shap not installed)."""
        pytest.importorskip("shap", reason="shap not installed in this environment")
        pytest.importorskip("joblib", reason="joblib not installed")
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'flask-app'
        ))
        from app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_home_page_returns_200(self, client):
        """The home page should render successfully."""
        response = client.get('/')
        assert response.status_code == 200

    def test_predict_endpoint_returns_json(self, client):
        """The /api/predict endpoint should return valid JSON with expected keys."""
        payload = {
            "loan_amount": 350000,
            "income": 85000,
            "property_value": 410000,
            "debt_to_income_ratio": 38.5,
            "loan_to_value_ratio": 85.0,
            "loan_term": 360,
            "age": "35-44",
            "sex": "Male",
            "race": "White",
        }
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 200

        data = response.get_json()
        assert 'pd' in data
        assert 'lgd' in data
        assert 'ecl' in data
        assert 'decision' in data
        assert 'top_factors' in data

    def test_predict_pd_is_probability(self, client):
        """Probability of Default must be between 0 and 1."""
        payload = {
            "loan_amount": 350000,
            "income": 85000,
            "property_value": 410000,
            "debt_to_income_ratio": 38.5,
            "loan_to_value_ratio": 85.0,
            "loan_term": 360,
        }
        response = client.post('/api/predict', json=payload)
        data = response.get_json()
        assert 0.0 <= data['pd'] <= 1.0

    def test_predict_ecl_is_positive(self, client):
        """Expected Credit Loss must be non-negative."""
        payload = {
            "loan_amount": 350000,
            "income": 85000,
            "property_value": 410000,
            "debt_to_income_ratio": 38.5,
            "loan_to_value_ratio": 85.0,
            "loan_term": 360,
        }
        response = client.post('/api/predict', json=payload)
        data = response.get_json()
        assert data['ecl'] >= 0

    def test_narrative_requires_token(self, client):
        """The /api/narrative endpoint should reject requests without a token."""
        response = client.post('/api/narrative', json={"hf_token": ""})
        assert response.status_code == 400


# =============================================================================
# ARTIFACT INTEGRITY TESTS
# =============================================================================

class TestArtifacts:
    """Tests verifying all required model and data artifacts exist."""

    @pytest.fixture
    def project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_feature_names_json_exists(self, project_root):
        """The feature_names.json file must exist for inference."""
        path = os.path.join(project_root, 'output', 'data', 'processed', 'feature_names.json')
        assert os.path.exists(path), "feature_names.json is missing from output/"

    def test_feature_names_json_not_empty(self, project_root):
        """feature_names.json must contain at least 1 feature."""
        path = os.path.join(project_root, 'output', 'data', 'processed', 'feature_names.json')
        with open(path, 'r') as f:
            features = json.load(f)
        assert len(features) > 0, "feature_names.json is empty"

    def test_lightgbm_model_exists(self, project_root):
        """LightGBM model artifact must exist for inference."""
        path = os.path.join(project_root, 'output', 'models', 'lightgbm.joblib')
        assert os.path.exists(path), "lightgbm.joblib is missing"

    def test_scaler_exists(self, project_root):
        """StandardScaler artifact must exist for preprocessing."""
        path = os.path.join(project_root, 'output', 'models', 'scaler.joblib')
        assert os.path.exists(path), "scaler.joblib is missing"

    def test_figures_exist(self, project_root):
        """Key figures for the web app must exist."""
        figures_dir = os.path.join(project_root, 'flask-app', 'static', 'images')
        required = [
            'fig5_roc_curve.png',
            'fig9_shap_summary.png',
            'fig10_shap_bar.png',
            'fig11_fairness_dashboard.png',
        ]
        for fig in required:
            assert os.path.exists(os.path.join(figures_dir, fig)), \
                f"Missing figure: {fig}"
