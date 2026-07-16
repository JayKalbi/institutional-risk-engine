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
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# ==============================================================================
# 1. CONFIGURATION FOR H100
# ==============================================================================
# H100s have 80GB of VRAM each. We can use massive batch sizes compared to Kaggle.
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_name = "JayKalbi/Credit-Risk-Memorandums" # Replace with your HF dataset if uploaded
output_dir = "./h100_hybrid_weights"

# We use 4-bit quantization to load it incredibly fast
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # H100 loves bfloat16
    bnb_4bit_use_double_quant=True,
)

# ==============================================================================
# 2. LOAD MODEL & TOKENIZER
# ==============================================================================
print("🚀 Loading Mistral-7B onto H100...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # Will automatically span across your 2 H100s
)

# Prepare for LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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
# 3. PREPARE DATASET
# ==============================================================================
# If your dataset is local (e.g. a CSV file), uncomment the next line:
# dataset = load_dataset("csv", data_files="my_local_data.csv", split="train")

# Assuming it's on HuggingFace:
dataset = load_dataset(dataset_name, split="train")

# ==============================================================================
# 4. H100 OPTIMIZED TRAINING ARGUMENTS
# ==============================================================================
print("🔥 Beginning H100 Training Phase...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32, # Massive batch size for H100 (Kaggle was likely 2 or 4)
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
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
# 5. SAVE WEIGHTS
# ==============================================================================
print(f"✅ Training Complete. Saving fine-tuned weights to {output_dir}")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
