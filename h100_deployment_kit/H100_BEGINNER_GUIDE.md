# Enterprise H100 Deployment Guide 🚀

This guide assumes you are logging into a Linux server (like Ubuntu) which is standard for AI labs. Don't worry if you've never used Linux before; just copy and paste these commands!

---

## Phase 1: Transfer the Project
First, you need to get your project files onto the H100 server.
1. Log into the H100 server (your professor will give you an SSH command, or you might just sit at a desktop connected to it).
2. Download your repository using git:
```bash
git clone https://github.com/JayKalbi/institutional-risk-engine.git
cd institutional-risk-engine
```
That's it! Because this guide and the kit are included in the repository, you have everything you need right here.

---

## Phase 2: Setup the Environment
On the H100 server, create a fresh Python environment and install the massive enterprise libraries needed to train on multiple GPUs.

Run these exact commands in the terminal:
```bash
# 1. Create a virtual environment
python3 -m venv h100_env
source h100_env/bin/activate

# 2. Install the standard requirements
pip install -r requirements.txt

# 3. Install the heavy-duty GPU libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate bitsandbytes vllm
```

---

## Phase 3: Train the Model (Full Speed)
On Kaggle, you only trained for 200 steps because the GPU was weak. On the H100, we are going to unleash its full power.

```bash
cd h100_deployment_kit
python train_h100.py
```
*What happens next:* You will see the terminal output the training logs. Because the H100 is so powerful, it will use a massive batch size and train across both GPUs simultaneously. When it is finished, it will save your final model weights in a folder called `h100_hybrid_weights`.

---

## Phase 4: Launch the vLLM Inference Server
Now that you have your trained weights, we need to host them as an API. We use **vLLM**, which is the fastest LLM server in the world.

Run this command:
```bash
bash serve_h100.sh
```
*What happens next:* The server will load your model into the H100's VRAM. It will say `Uvicorn running on http://0.0.0.0:8000`. **Do not close this terminal!** Your model is now live.

---

## Phase 5: Launch Your Dashboard
Open a **new** terminal window (leave the vLLM server running in the first one).

Run this command to start your Flask website using the special H100 version of the app:
```bash
# Activate the environment in this new terminal
cd institutional-risk-engine
source h100_env/bin/activate

# Launch the app
cd h100_deployment_kit
python app_h100_flask.py
```

### You Are Done! 🎉
Open a web browser on the server and go to `http://localhost:5000`. 
When you type into the dashboard and click "Run Analysis", the website will **not** reach out to Groq or the internet. Instead, it will send the data directly to your local vLLM server, use your *actual* fine-tuned Mistral-7B weights on the H100, and generate the Credit Memorandum instantly. 

You have successfully deployed an enterprise AI system!
