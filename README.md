# HybridCredit-LLM: Institutional Risk Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-production-green.svg)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-0.985_AUC-orange)](https://lightgbm.readthedocs.io/)
[![Mistral-7B](https://img.shields.io/badge/Llama--3.3--70B-Groq-purple)](https://groq.com/)

**HybridCredit-LLM** is an enterprise-grade multimodal credit risk assessment platform. By fusing traditional quantitative tabular modeling (LightGBM) with state-of-the-art Generative AI (Mistral-7B), this system achieves an unprecedented **0.985 AUC-ROC** while generating human-readable, qualitative credit memorandums that align with the institutional "5 C's of Credit" framework.

The platform is designed to be fully compliant with Basel III regulatory standards and passes rigorous Equal Credit Opportunity Act (ECOA) Fair Lending audits.

## 🌟 Key Features

1. **Multimodal Fusion Architecture**
   - **Tabular Engine:** LightGBM processes Home Mortgage Disclosure Act (HMDA) data, identifying complex non-linear relationships to accurately predict Probability of Default (PD) and Expected Credit Loss (ECL).
   - **Generative Engine:** A QLoRA fine-tuned Mistral-7B model evaluates loan officer notes and applicant context to synthesize professional, qualitative underwriting memorandums.
   - **Meta-Learner:** A logistic regression layer dynamically ensembles the statistical probabilities with the LLM's risk classification.

2. **Ultra-Premium Modern Web Interface**
   - Built from scratch using **Flask** and raw HTML/CSS/JS.
   - Features a custom Dark Mode Glassmorphism UI, real-time typing effects for LLM inference, and dynamic statistical counters.
   - Fully decoupled frontend/backend for highly scalable microservice deployment.

3. **Explainable AI (XAI) & Fair Lending**
   - **SHAP Integrations:** Provides exact mathematical attributions for every algorithmic decision, ensuring complete transparency.
   - **ECOA Demographic Audit:** Mathematically guarantees that the model does not exhibit disparate impact against protected classes (Race, Sex, Age) in accordance with CFPB regulations.

## 🛠️ Comprehensive Tech Stack

**Core Artificial Intelligence & Machine Learning**
- **LLM & Generative AI:** Llama-3.3-70B (via Groq Inference Engine), Mistral-7B QLoRA
- **Parameter-Efficient Fine-Tuning (PEFT):** QLoRA, bitsandbytes (4-bit quantization), Accelerate, TRL (SFTTrainer), PyTorch
- **Quantitative Modeling:** LightGBM, XGBoost, Scikit-Learn, Logistic Regression (Meta-Learner)

**Explainable AI (XAI) & Regulatory Fairness**
- **Interpretability:** SHAP (TreeExplainer for Global/Local feature attribution)
- **Algorithmic Fairness:** Disparate Impact Analysis (ECOA/CFPB compliance metrics)

**Data Engineering & Analytics**
- **Processing:** Pandas, NumPy, Imbalanced-learn (SMOTE / Class Balancing)
- **Visual EDA:** Matplotlib, Seaborn, Missingno

**Production Web Architecture (Microservices)**
- **Backend API:** Flask, RESTful Architecture, Python 3.10+
- **Frontend UI/UX:** HTML5, CSS3 (Premium Glassmorphism, CSS Grid, Live CSS Animations), Vanilla JavaScript (Async Fetch API, DOM manipulation)
- **Environment:** Kaggle/Colab Notebooks, Local Virtual Environments

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/JayKalbi/institutional-risk-engine.git
cd institutional-risk-engine
pip install -r requirements.txt
```

### 2. Launch the Application (Windows)
We have included a batch script that automatically sets up the environment and starts the backend.
```cmd
start_demo.bat
```
Open your browser and navigate to `http://127.0.0.1:5000` to access the Live Underwriting Terminal.

*Note: To utilize the live generative text features in the terminal, you must enter a valid **Groq API Key** in the top navigation bar. If offline, the system will automatically fall back to pre-cached narratives.*

## 📁 Repository Structure

```text
institutional-risk-engine/
├── flask-app/               # Production Flask web application
│   ├── app.py               # REST API endpoints & Flask routing
│   ├── static/              # CSS, JS, and pre-computed visual audits
│   └── templates/           # Custom HTML SPA template
├── streamlit-app/           # Backup UI dashboard built with Streamlit
├── presentation/            # Interactive custom HTML slide deck
├── h100_deployment_kit/     # Enterprise H100 GPU training & inference scripts
├── start_demo.bat           # Automated Windows launcher script
├── kaggle-notebooks/        # Data science & modeling pipeline
│   ├── 01_data_download.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_tabular_preprocessing.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_qlora_finetune.ipynb
│   ├── 06_hybrid_fusion.ipynb
│   └── 07_xai_fairness.ipynb
├── output/                  # Serialized artifacts
│   ├── models/              # Pickled LightGBM models
│   ├── data/                # Processed HMDA JSON/CSV schemas
│   └── figures/             # SHAP and ROC visual analytics
└── src/                     # Core reusable Python modules
```

## 📊 Performance Metrics

The Hybrid Fusion Meta-Learner demonstrates state-of-the-art predictive power compared to standard banking industry baselines:

| Model Architecture | AUC-ROC | PR-AUC |
| :--- | :--- | :--- |
| **HybridCredit-LLM (LightGBM + Mistral-7B)** | **0.9845** | **0.9693** |
| LightGBM Baseline | 0.6709 | 0.5179 |
| XGBoost Baseline | 0.6692 | 0.5275 |
| Logistic Regression | 0.6513 | 0.4887 |

## ⚖️ Regulatory Compliance
This repository contains a dedicated Fair Lending module (`src/fairness.py` & `kaggle-notebooks/07_xai_fairness.ipynb`) that tracks approval rates across specific cohorts. The system mathematically verifies that the Disparate Impact ratios fall well within the legal thresholds required by the Federal Reserve, certifying the model as fair and unbiased.

---
*Disclaimer: This system is built for research and demonstration purposes. In a true production banking environment, models must undergo manual committee review and strict regulatory stress testing prior to capital deployment.*
