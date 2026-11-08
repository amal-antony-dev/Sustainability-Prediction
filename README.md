# Sustainability-Prediction

The sustainability-scoring model behind **[EcoSwapChain](https://github.com/amal-antony-dev/ecoswapchain)**, a recommerce platform where every item is an NFT. When a product is listed, this model scores it from 0 to 100. The score, the price and the number of previous owners decide the SwapCoin reward for keeping the item in use.

> **Status:** part of my final-year college project (2025). See [Limitations](#limitations) before reading anything into the scores.
>
> Case study: [amal-antony-dev.vercel.app](https://amal-antony-dev.vercel.app/#work/ecoswapchain/overview)

## Part of EcoSwapChain

| Repo | Role |
|---|---|
| [ecoswapchain](https://github.com/amal-antony-dev/ecoswapchain) | Django API, Celery workers, websockets. Loads this model to score new listings |
| [ESC-Frontend](https://github.com/amal-antony-dev/ESC-Frontend) | React + TypeScript web app |
| [ESC-BlockChain](https://github.com/amal-antony-dev/ESC-BlockChain) | Node.js treasury service on Solana |
| **Sustainability-Prediction** | This repo: dataset, training and the scoring model |

## How it works

1. **Data.** No public dataset links second-hand products to sustainability, so `generate_dataset.py` creates **2,000 synthetic products** across 12 categories. Each has materials, condition, recycled content, recyclability, carbon footprint, energy efficiency, durability, repairability and ethical flags (ethically sourced, cruelty-free, plastic-free and so on), with realistic missing values. A rule-based function scores each product 0–100.
2. **Preprocessing.** `preprocessing.py` splits the data into train and test sets and builds a scikit-learn `ColumnTransformer`: mean imputation and scaling for numbers, constant imputation and one-hot encoding for categories.
3. **Model selection.** `train.py` compares **Random Forest**, **Gradient Boosting** and **Histogram Gradient Boosting**, each tuned with `RandomizedSearchCV` (20 iterations, 3-fold CV), and reports test MSE and R² with prediction plots.
4. **Deployment.** The **Gradient Boosting** pipeline was chosen and ships inside the EcoSwapChain backend, where a Celery task scores each new listing.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python generate_dataset.py   # writes sustainability_data.csv (2,000 rows)
python preprocessing.py      # writes train_data.csv, test_data.csv, preprocessor.pkl
python train.py              # trains all three models, saves *_model.pkl and plots
python predict.py            # scores synthetic_prediction_input.csv → synthetic_predictions.csv
```

`model.py` is an earlier all-in-one version of the same pipeline, kept for reference.

## Files

| Path | Contents |
|---|---|
| `generate_dataset.py` | Synthetic product generator and the rule-based scoring function |
| `preprocessing.py`, `train.py`, `predict.py` | The pipeline above |
| `*_model.pkl`, `preprocessor.pkl` | Trained pipelines |
| `*.png` | Prediction, residual and feature-importance plots |
| `*.csv` | Generated data, splits and sample predictions |

## Limitations

The labels come from a formula, so the model largely learns to reproduce that formula, and its accuracy says little about real products. Trustworthy scores would need verified product data, such as lifecycle assessments or certifications, instead of synthetic labels.

## Tech

Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · joblib
