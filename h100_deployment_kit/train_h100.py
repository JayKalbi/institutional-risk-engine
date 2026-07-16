"""
Enterprise H100 Training Script (Mistral-7B QLoRA)
Optimized for 2x NVIDIA H100 GPUs (160GB VRAM)
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==============================================================================
# 1. CONFIGURATION FOR H100
# ==============================================================================
# H100s have 80GB of VRAM each. We do NOT need 4-bit quantization. We use bfloat16.
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_name = "JayKalbi/Credit-Risk-Memorandums" # Replace with your HF dataset if uploaded
output_dir = "./h100_hybrid_weights"

# ==============================================================================
# 2. LOAD MODEL & TOKENIZER
# ==============================================================================
print("🚀 Loading Mistral-7B onto H100 in bfloat16...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the model directly in bfloat16 (H100 native format)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Will automatically span across your 2 H100s
)

# Prepare for LoRA
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=64, # High rank for better learning (H100 can handle it)
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ==============================================================================
# 3. PREPARE DATASET (LOCAL PARQUET METHOD)
# ==============================================================================
print("📊 Loading local parquet dataset...")
import pandas as pd
from datasets import Dataset

# Determine path to the processed data in the main repository
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'output', 'data', 'processed', 'hmda_binary.parquet')

# Read the local data
df = pd.read_parquet(data_path)

# Filter for approved (1) and denied (3)
df = df[df['action_taken'].isin([1, 3])].copy()
df['default'] = (df['action_taken'] == 3).astype(int)

# Create the exact prompt strings that the LLM will train on
def format_prompt(row):
    # This simulates the logic from your Kaggle notebook
    outcome = "HIGH" if row['default'] == 1 else "LOW"
    prompt = f"<s>[INST] Evaluate the default risk for this loan application.\n\n"
    prompt += f"Income: ${row.get('income', 0)*1000:,.2f}\n"
    prompt += f"Loan Amount: ${row.get('loan_amount', 0)*1000:,.2f}\n"
    prompt += f"LTV Ratio: {row.get('loan_to_value_ratio', 0)}%\n"
    prompt += f"DTI Ratio: {row.get('debt_to_income_ratio', 0)}%\n"
    prompt += f"[/INST]\nDefault Risk: {outcome}</s>"
    return prompt

print("Formatting prompts...")
df['text'] = df.apply(format_prompt, axis=1)

# Convert the Pandas DataFrame into a HuggingFace Dataset object
dataset = Dataset.from_pandas(df[['text']])

# ==============================================================================
# 4. H100 OPTIMIZED TRAINING ARGUMENTS
# ==============================================================================
print("🔥 Beginning H100 Training Phase...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16, # High batch size for H100
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # H100 hardware specifically accelerates bf16
    max_grad_norm=0.3,
    max_steps=-1, # Run for full epochs instead of just 200 steps
    num_train_epochs=3, # Train 3 full times over the data
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", # Ensure your dataset has a 'text' column
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# Train the model
trainer.train()

# ==============================================================================
# 5. MERGE AND SAVE WEIGHTS FOR VLLM
# ==============================================================================
print(f"✅ Training Complete. Merging weights for vLLM inference...")
# We must merge the LoRA adapters back into the base model so vLLM can load it instantly
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"🎉 Fully merged model saved to {output_dir}. Ready for serve_h100.sh!")
