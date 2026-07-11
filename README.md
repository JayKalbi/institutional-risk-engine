# Institutional-Risk-Engine: Mistral-7B Fine-Tuned for Credit Default Prediction

> A regulatory-compliant multimodal credit risk framework fusing QLoRA-adapted Mistral-7B with gradient boosting, featuring SHAP explainability, counterfactual reasoning, and ECOA fairness auditing.

---

## Project Overview

This project implements **Institutional-Risk-Engine**, a hybrid AI architecture for credit default prediction that:

- **Fine-tunes Mistral-7B-Instruct-v0.3** with QLoRA (4-bit quantization) on the HMDA mortgage dataset
- **Fuses LLM text embeddings** with LightGBM tabular predictions via late-fusion meta-learning
- **Generates natural language rationales** for every prediction
- **Provides SHAP explainability** + counterfactual explanations (EU AI Act compliant)
- **Passes ECOA fairness audits** across demographic groups
- **Deploys locally** via a lightweight Streamlit web interface

### Research Questions

1. **RQ1**: Does fusing LLM-extracted embeddings with tabular features improve default prediction over unimodal baselines?
2. **RQ2**: Do LLM self-generated rationales align with SHAP feature attributions?
3. **RQ3**: Can this architecture satisfy EU AI Act / Basel III explainability requirements?
4. **RQ4**: Does the model exhibit demographic fairness under ECOA criteria?

---

## Folder Structure

```
institutional-risk-engine/
|-- kaggle-notebooks/          # 7 Kaggle notebooks (run in order)
|   |-- 01_data_download.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_tabular_preprocessing.ipynb
|   |-- 04_baselines.ipynb
|   |-- 05_qlora_finetune.ipynb
|   |-- 06_hybrid_fusion.ipynb
|   |-- 07_xai_fairness.ipynb
|
|-- src/                       # Reusable Python modules
|   |-- preprocess.py
|   |-- models.py
|   |-- evaluate.py
|   |-- explain.py
|   |-- fairness.py
|   |-- utils.py
|

|-- streamlit-app/             # Streamlit frontend
|   |-- app.py
|   |-- components.py
|   |-- Dockerfile
|

|-- research-paper/            # LaTeX research paper
|   |-- main.tex
|   |-- references.bib
|   |-- figures/
|
|-- data/
|   |-- raw/                   # Downloaded datasets
|   |-- processed/             # Cleaned/preprocessed data
|   |-- embeddings/            # Saved embeddings
|
|-- models/                    # Saved model checkpoints
|-- figures/                   # Paper figures
|-- results/                   # Experiment metrics (JSON/CSV)
|-- requirements.txt
|-- setup.sh
|-- README.md
```

---

## Quick Start (Kaggle)

### Step 1: Setup Kaggle Environment

1. Go to [kaggle.com](https://www.kaggle.com) and create an account
2. Download your `kaggle.json` API token (Profile → Account → API → Create New Token)
3. In a Kaggle notebook, upload `kaggle.json` or use the integrated dataset download

### Step 2: Run Notebooks in Order

| Notebook | Purpose | GPU Required |
|----------|---------|-------------|
| `01_data_download.ipynb` | Download HMDA dataset | No |
| `02_eda.ipynb` | Exploratory data analysis | No |
| `03_tabular_preprocessing.ipynb` | Feature engineering & preprocessing | No |
| `04_baselines.ipynb` | Train XGBoost, LightGBM, Logistic Regression | No |
| `05_qlora_finetune.ipynb` | Mistral-7B QLoRA fine-tuning | **T4/P100 required** |
| `06_hybrid_fusion.ipynb` | Late fusion & evaluation | No |
| `07_xai_fairness.ipynb` | SHAP, counterfactuals, fairness audit | No |

### Kaggle GPU Setup for Notebook 5

- In Kaggle: Notebook → Accelerator → Select **GPU T4 x2**
- Alternatively use **Kaggle P100** for faster training
- Mistral-7B QLoRA with 4-bit quantization fits in ~16GB VRAM

---

## Local Deployment (Streamlit)

### Prerequisites

- Python 3.9+
- Trained model artifacts downloaded from Kaggle (`output/models/`)

### Run Locally

```bash
pip install -r requirements.txt
cd streamlit-app
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Backbone | Mistral-7B-Instruct-v0.3 |
| Fine-tuning | QLoRA (4-bit NF4) via PEFT |
| Classical ML | LightGBM, XGBoost |
| Fusion | Logistic Regression meta-learner |
| Explainability | SHAP, DiCE-ML |
| Fairness | Aequitas, Fairlearn |
| Frontend | Streamlit |
| Deployment | Local Execution |

---

## Key Results (Expected)

| Model | AUC-ROC | PR-AUC | KS |
|-------|---------|--------|-----|
| Logistic Regression | ~0.72 | ~0.45 | ~0.35 |
| XGBoost | ~0.78 | ~0.55 | ~0.42 |
| LightGBM | ~0.80 | ~0.58 | ~0.45 |
| Mistral-7B (text-only) | ~0.75 | ~0.50 | ~0.38 |
| **Institutional-Risk-Engine (ours)** | **~0.84** | **~0.63** | **~0.50** |

---

## Citation

If you use this work, please cite:

```bibtex
@article{institutionalriskengine2025,
  title={Institutional-Risk-Engine: A Regulatory-Compliant Multimodal Framework 
         Fusing QLoRA-Adapted Mistral-7B with Gradient Boosting 
         for Explainable Credit Default Prediction},
  author={Jay Kalbi},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License - Academic and commercial use permitted.
