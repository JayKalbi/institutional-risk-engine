#!/bin/bash
# Enterprise H100 Inference Server (vLLM)
# This script hosts your fine-tuned Mistral-7B weights as an OpenAI-compatible API

echo "=========================================================="
echo "Starting vLLM Inference Engine on H100 Cluster..."
echo "=========================================================="

# 1. Install vLLM if not installed
pip install vllm

# 2. Launch the server
# We point --model to the folder where train_h100.py saved your weights
# --tensor-parallel-size 2 spans the model across both your H100s for max speed
# --dtype bfloat16 uses the H100's native optimized math

python -m vllm.entrypoints.openai.api_server \
    --model "./h100_hybrid_weights" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000

echo "Server crashed or stopped."
