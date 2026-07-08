"""
===============================================================================
KAGGLE NOTEBOOK 5: Mistral-7B QLoRA Fine-Tuning
===============================================================================
Run this FIFTH on Kaggle (GPU REQUIRED - T4 or P100)

What this notebook does:
1. Loads Mistral-7B-Instruct-v0.3 with 4-bit quantization
2. Configures QLoRA (Low-Rank Adaptation) adapters
3. Prepares instruction-format training data from HMDA
4. Fine-tunes the model for credit risk prediction + rationale generation
5. Evaluates the fine-tuned model
6. Extracts LLM predictions for hybrid fusion

THIS IS THE CORE NOVEL CONTRIBUTION OF THE PAPER.

KAGGLE SETUP:
- Create new notebook on kaggle.com
- Set Accelerator: GPU T4 x2 (or P100)
- Runtime: ~2-4 hours depending on GPU
- Make sure to enable internet access for model download
===============================================================================
"""

# =============================================================================
# CELL 1: Verify GPU & Install Dependencies
# =============================================================================

import torch
print("=" * 70)
print("KAGGLE NOTEBOOK 5: Mistral-7B QLoRA Fine-Tuning")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU detected! This notebook requires GPU.")
    raise RuntimeError("GPU required for LLM fine-tuning")

# Install required packages (if not already available)
import subprocess
packages = ['transformers', 'peft', 'bitsandbytes', 'accelerate', 'trl', 'datasets']
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call(['pip', 'install', '-q', pkg])

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import json
import gc
import warnings
warnings.filterwarnings('ignore')

WORKING_DIR = Path("./output")
PROCESSED_DIR = WORKING_DIR / "data" / "processed"
RESULTS_DIR = WORKING_DIR / "results"
MODELS_DIR = WORKING_DIR / "models"
QLORA_DIR = MODELS_DIR / "mistral_qlora"

for d in [RESULTS_DIR, MODELS_DIR, QLORA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CELL 2: 4-Bit Quantization Config
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: 4-BIT QUANTIZATION CONFIG")
print("=" * 70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 - optimal for LLMs
    bnb_4bit_compute_dtype=torch.float16,   # FP16 for computation
    bnb_4bit_use_double_quant=True,          # Nested quantization for memory savings
)

print("Quantization config:")
print(f"  - 4-bit loading: True")
print(f"  - Quant type: NF4")
print(f"  - Compute dtype: float16")
print(f"  - Double quant: True")

# =============================================================================
# CELL 3: Load Mistral-7B-Instruct
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: LOADING MISTRAL-7B-INSTRUCT")
print("=" * 70)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Model: {MODEL_ID}")
print("Downloading and quantizing (this takes ~5-10 minutes)...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="right"
)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",           # Automatically distribute across GPUs
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print(f"\nModel loaded!")
print(f"Device map: {model.hf_device_map}")

# Print memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory allocated: {allocated:.2f} GB")

# =============================================================================
# CELL 4: QLoRA Configuration
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: QLORA ADAPTER CONFIGURATION")
print("=" * 70)

lora_config = LoraConfig(
    r=16,                        # Rank: capacity vs memory tradeoff
    lora_alpha=32,               # Scaling: 2x rank (standard)
    target_modules=[             # All attention and MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,           # Regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for k-bit training (essential!)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("\nLoRA Configuration:")
print(f"  Rank (r): 16")
print(f"  Alpha: 32")
print(f"  Dropout: 0.05")
print(f"  Target modules: All attention + MLP layers")

# Print trainable parameters
model.print_trainable_parameters()
# Expected: ~42M trainable / ~7B total = ~0.6% trainable

# =============================================================================
# CELL 5: Prepare Training Data (Instruction Format)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: PREPARING INSTRUCTION-FORMAT TRAINING DATA")
print("=" * 70)

# Load HMDA data
df = pd.read_parquet(PROCESSED_DIR / "hmda_binary.parquet")
df = df[df['action_taken'].isin([1, 3])].copy()
df['default'] = (df['action_taken'] == 3).astype(int)

# Sample balanced dataset for fine-tuning (50K: 25K default, 25K non-default)
n_samples = 50000
defaults = df[df['default'] == 1].sample(n=min(25000, len(df[df['default']==1])), random_state=42)
non_defaults = df[df['default'] == 0].sample(n=min(25000, len(df[df['default']==0])), random_state=42)
train_df = pd.concat([defaults, non_defaults]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Training samples: {len(train_df):,}")
print(f"  Default: {(train_df['default']==1).sum():,}")
print(f"  Non-default: {(train_df['default']==0).sum():,}")

# System prompt for credit risk analysis
SYSTEM_PROMPT = """You are an expert credit risk analyst AI. Analyze loan applications and predict default risk with detailed explanations."""

def create_instruction(row):
    """Create instruction-format prompt for credit risk analysis."""
    
    # Build input prompt
    user_msg = f"""Analyze this loan application:

Loan Details:
- Amount: ${row.get('loan_amount', 0):,.0f}
- Applicant Income: ${row.get('income', 0):,.0f}
- Loan-to-Income Ratio: {row.get('loan_amount', 0)/(row.get('income', 1)+1):.3f}
- Debt-to-Income Ratio: {row.get('debt_to_income_ratio', 0):.1f}%
- Loan-to-Value Ratio: {row.get('loan_to_value_ratio', 0):.1f}%
- Property Value: ${row.get('property_value', 0):,.0f}
- Loan Term: {row.get('loan_term', 360):.0f} months
- Loan Type: {row.get('loan_type', 'N/A')}
- Loan Purpose: {row.get('loan_purpose', 'N/A')}
- Lien Status: {row.get('lien_status', 'N/A')}
- Occupancy: {row.get('occupancy_type', 'N/A')}

Task: Predict default risk (HIGH/LOW) and explain the 3 key factors."""
    
    # Build output (what model should learn to generate)
    is_denied = row['default'] == 1
    dti = row.get('debt_to_income_ratio', 0)
    ltv = row.get('loan_to_value_ratio', 0)
    lti = row.get('loan_amount', 0) / (row.get('income', 1) + 1)
    
    if is_denied:
        response = f"""Default Risk: HIGH

Key Factors:
1. Debt-to-Income Ratio: The applicant's DTI of {dti:.1f}% exceeds safe lending thresholds (typically 43%), indicating potential difficulty meeting monthly obligations.

2. Loan-to-Value Profile: The LTV ratio of {ltv:.1f}% suggests higher collateral risk, providing limited buffer against potential property value declines.

3. Loan-to-Income Burden: The loan amount relative to income (ratio: {lti:.3f}) raises concerns about the applicant's capacity to service debt comfortably."""
    else:
        response = f"""Default Risk: LOW

Key Factors:
1. Manageable Debt-to-Income: The DTI of {dti:.1f}% falls within acceptable parameters, suggesting the borrower can comfortably manage monthly payments.

2. Acceptable Loan-to-Value: The LTV of {ltv:.1f}% provides adequate collateral coverage, reducing lender exposure to potential losses.

3. Affordable Loan Burden: The loan-to-income ratio of {lti:.3f} indicates the loan amount is proportionate to the applicant's earning capacity."""
    
    return {"instruction": SYSTEM_PROMPT, "input": user_msg, "output": response}

# Generate instruction dataset
print("\nGenerating instruction-format training data...")
instructions = [create_instruction(row) for _, row in train_df.iterrows()]

# Save for inspection
with open(PROCESSED_DIR / "qlora_instructions.json", 'w') as f:
    json.dump(instructions[:5], f, indent=2)  # Save sample

print(f"Generated {len(instructions)} instruction examples")
print("\nSample instruction:")
print("-" * 50)
print(f"INSTRUCTION: {instructions[0]['instruction'][:100]}...")
print(f"INPUT: {instructions[0]['input'][:200]}...")
print(f"OUTPUT: {instructions[0]['output'][:200]}...")

# =============================================================================
# CELL 6: Format for Mistral Chat Template
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: FORMATTING FOR MISTRAL CHAT TEMPLATE")
print("=" * 70)

def format_mistral_prompt(example):
    """Format instruction data for Mistral-7B-Instruct chat template."""
    text = f"<s>[INST] {example['instruction']}\n\n{example['input']} [/INST]\n{example['output']}</s>"
    return {"text": text}

# Create HuggingFace dataset
dataset = Dataset.from_list(instructions)
dataset = dataset.map(format_mistral_prompt)

print(f"Dataset ready: {len(dataset)} examples")
print(f"\nFormatted example:")
print("-" * 50)
print(dataset[0]['text'][:500] + "...")

# =============================================================================
# CELL 7: Training Arguments
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: TRAINING CONFIGURATION")
print("=" * 70)

training_args = TrainingArguments(
    output_dir=str(QLORA_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2,        # Small batch for 16GB VRAM
    gradient_accumulation_steps=8,         # Effective batch = 16
    gradient_checkpointing=True,           # Save memory
    optim="paged_adamw_32bit",            # Memory-efficient optimizer
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,                     # Gradient clipping
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,                             # Mixed precision
    report_to="none",                      # No wandb on Kaggle
    remove_unused_columns=False,
)

print("Training configuration:")
print(f"  Epochs: 3")
print(f"  Batch size: 2 (effective: 16 with grad accum)")
print(f"  Learning rate: 2e-4")
print(f"  Scheduler: cosine with 3% warmup")
print(f"  Optimizer: paged_adamw_32bit")
print(f"  Max grad norm: 0.3")

# =============================================================================
# CELL 8: Fine-Tune!
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: FINE-TUNING (THIS WILL TAKE 2-4 HOURS)")
print("=" * 70)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,                   # Truncate/pad to 1024 tokens
    packing=False,                         # No packing for clarity
)

print("Starting training...")
print("(Progress will be logged every 50 steps)")
print("=" * 70)

trainer.train()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

# =============================================================================
# CELL 9: Save Fine-Tuned Model
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: SAVING FINE-TUNED MODEL")
print("=" * 70)

# Save adapter weights
FINAL_DIR = QLORA_DIR / "final"
trainer.save_model(str(FINAL_DIR))
tokenizer.save_pretrained(str(FINAL_DIR))

print(f"Saved fine-tuned model to: {FINAL_DIR}")
print(f"Files saved:")
for f in FINAL_DIR.iterdir():
    size_mb = f.stat().st_size / 1024**2
    print(f"  {f.name}: {size_mb:.1f} MB")

# Save training summary
training_summary = {
    'base_model': MODEL_ID,
    'lora_rank': 16,
    'lora_alpha': 32,
    'epochs': 3,
    'batch_size': 2,
    'effective_batch_size': 16,
    'learning_rate': 2e-4,
    'training_samples': len(dataset),
    'max_seq_length': 1024,
    'quantization': '4bit-NF4',
    'output_dir': str(FINAL_DIR)
}

with open(RESULTS_DIR / "qlora_training_summary.json", 'w') as f:
    json.dump(training_summary, f, indent=2)

print(f"\nSaved training summary to {RESULTS_DIR}/qlora_training_summary.json")

# Clear memory
del trainer, dataset
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# CELL 10: Test Fine-Tuned Model
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: TESTING FINE-TUNED MODEL")
print("=" * 70)

# Reload model with adapter
print("Loading fine-tuned model with QLoRA adapter...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model_ft = PeftModel.from_pretrained(base_model, str(FINAL_DIR))
model_ft.eval()

# Test on a few examples
test_samples = train_df.sample(3, random_state=42)

for idx, (_, row) in enumerate(test_samples.iterrows()):
    instruction = create_instruction(row)
    
    # Format prompt
    prompt = f"<s>[INST] {instruction['instruction']}\n\n{instruction['input']} [/INST]\nDefault Risk:"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    
    with torch.no_grad():
        outputs = model_ft.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,           # Low temperature for deterministic output
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    actual = "DENIED" if row['default'] == 1 else "ORIGINATED"
    predicted = "HIGH" if "HIGH" in response.upper() else "LOW"
    
    print(f"\n--- Example {idx+1} ---")
    print(f"Actual: {actual} | Predicted Risk: {predicted}")
    print(f"Response preview: {response[:300]}...")

# =============================================================================
# CELL 11: Extract LLM Predictions for Fusion
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: EXTRACTING LLM PREDICTIONS FOR FUSION")
print("=" * 70)

# Load test data
test_df = pd.read_parquet(PROCESSED_DIR / "hmda_binary.parquet")
test_df = test_df[test_df['action_taken'].isin([1, 3])].copy()
test_df['default'] = (test_df['action_taken'] == 3).astype(int)

# Sample for LLM evaluation (1000 samples for speed)
test_sample = test_df.sample(n=min(1000, len(test_df)), random_state=42)

llm_probas = []
llm_rationales = []

print(f"Processing {len(test_sample)} test samples...")

for i, (_, row) in enumerate(test_sample.iterrows()):
    instruction = create_instruction(row)
    prompt = f"<s>[INST] {instruction['instruction']}\n\n{instruction['input']} [/INST]\nDefault Risk:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    
    with torch.no_grad():
        outputs = model_ft.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract risk score (HIGH = 1, LOW = 0)
    risk_score = 1.0 if "HIGH" in response.upper() else 0.0
    llm_probas.append(risk_score)
    llm_rationales.append(response)
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(test_sample)}")
        # Clear cache periodically
        torch.cuda.empty_cache()

llm_probas = np.array(llm_probas)
y_test_sample = test_sample['default'].values

# Evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp

llm_auc = roc_auc_score(y_test_sample, llm_probas)
llm_pr = average_precision_score(y_test_sample, llm_probas)
llm_ks = ks_2samp(llm_probas[y_test_sample==1], llm_probas[y_test_sample==0])[0]

print(f"\nLLM-Only Results (on {len(test_sample)} samples):")
print(f"  AUC-ROC: {llm_auc:.4f}")
print(f"  PR-AUC: {llm_pr:.4f}")
print(f"  KS: {llm_ks:.4f}")

# Save
np.save(PROCESSED_DIR / "llm_test_proba.npy", llm_probas)
np.save(PROCESSED_DIR / "llm_test_labels.npy", y_test_sample)

# Save rationales for faithfulness audit
with open(PROCESSED_DIR / "llm_rationales.json", 'w') as f:
    json.dump(llm_rationales, f)

print(f"\nSaved LLM predictions to {PROCESSED_DIR}/")

# =============================================================================
# CELL 12: Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("NOTEBOOK 5 COMPLETE!")
print("=" * 70)
print(f"""
Fine-tuning Summary:
  Base Model: Mistral-7B-Instruct-v0.3
  Method: QLoRA (4-bit NF4)
  Trainable Parameters: ~42M (~0.6% of total)
  Training Samples: {len(train_df):,}
  Epochs: 3
  Effective Batch Size: 16

LLM-Only Performance:
  AUC-ROC: {llm_auc:.4f}
  PR-AUC:  {llm_pr:.4f}
  KS:      {llm_ks:.4f}

Saved to:
  - Model: {FINAL_DIR}/
  - Predictions: {PROCESSED_DIR}/llm_test_proba.npy
  - Rationales: {PROCESSED_DIR}/llm_rationales.json

Next: Run Notebook 6 (Hybrid Fusion Model)
""")
