"""
Utility functions for the CreditRisk-LLM project.
Run on Kaggle or local environment.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_results(results: Dict, experiment_name: str, output_dir: str = "results"):
    """Save experiment results with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    save_json(results, filepath)
    return filepath


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def create_instruction_prompt(row: pd.Series, include_text: bool = True) -> str:
    """
    Create instruction-format prompt for Mistral fine-tuning.
    
    Args:
        row: DataFrame row with loan application data
        include_text: Whether to include narrative text
    
    Returns:
        Formatted instruction prompt string
    """
    base_prompt = f"""You are a credit risk analyst AI. Analyze this loan application and predict default risk.

Loan Application Details:
- Loan Amount: ${row.get('loan_amount', 'N/A'):,.0f}
- Applicant Income: ${row.get('income', 'N/A'):,.0f}
- Loan-to-Income Ratio: {row.get('loan_to_income_ratio', 'N/A'):.3f}
- Debt-to-Income Ratio: {row.get('debt_to_income_ratio', 'N/A'):.1f}%
- Property Value: ${row.get('property_value', 'N/A'):,.0f}
- Loan Type: {row.get('loan_type', 'N/A')}
- Loan Purpose: {row.get('loan_purpose', 'N/A')}
- lien_status: {row.get('lien_status', 'N/A')}
- Occupancy Type: {row.get('occupancy_type', 'N/A')}
- Applicant Sex: {row.get('applicant_sex', 'N/A')}
- Applicant Race: {row.get('applicant_race', 'N/A')}
- Applicant Ethnicity: {row.get('applicant_ethnicity', 'N/A')}
- Applicant Age: {row.get('applicant_age', 'N/A')}
"""
    
    if include_text and 'narrative' in row and pd.notna(row['narrative']):
        base_prompt += f"\nApplicant Narrative: {row['narrative'][:500]}\n"
    
    base_prompt += """
Task: Predict the default risk (HIGH/LOW) and explain your reasoning with 3 key factors.

Response Format:
Default Risk: [HIGH/LOW]

Key Factors:
1. [Factor 1 with explanation]
2. [Factor 2 with explanation]
3. [Factor 3 with explanation]
"""
    return base_prompt


def create_output_rationale(row: pd.Series) -> str:
    """Create target output rationale for fine-tuning."""
    is_default = row.get('action_taken', 0) == 3  # 3 = Application denied
    
    if is_default:
        risk_factors = [
            f"High debt-to-income ratio of {row.get('debt_to_income_ratio', 'N/A'):.1f}% exceeds safe lending thresholds",
            f"Loan amount of ${row.get('loan_amount', 'N/A'):,.0f} represents elevated relative to income of ${row.get('income', 'N/A'):,.0f}",
            f"Loan-to-value characteristics suggest higher collateral risk profile"
        ]
        return f"""Default Risk: HIGH

Key Factors:
1. {risk_factors[0]}
2. {risk_factors[1]}
3. {risk_factors[2]}
"""
    else:
        positive_factors = [
            f"Debt-to-income ratio of {row.get('debt_to_income_ratio', 'N/A'):.1f}% falls within acceptable range",
            f"Income level of ${row.get('income', 'N/A'):,.0f} supports loan affordability",
            f"Overall application profile demonstrates strong repayment capacity"
        ]
        return f"""Default Risk: LOW

Key Factors:
1. {positive_factors[0]}
2. {positive_factors[1]}
3. {positive_factors[2]}
"""


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = datetime.now()
        logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start).total_seconds()
        logger.info(f"Completed: {self.name} in {format_time(self.elapsed)}")
