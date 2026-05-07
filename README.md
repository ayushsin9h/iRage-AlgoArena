# 📈 iRage-AlgoArena: Short-Horizon Return Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/Ensemble-LightGBM-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Quantitative Finance Challenge by iRage:** A machine learning pipeline engineered to predict short-horizon percentage returns from an anonymized, high-dimensional tabular dataset. The goal is to extract realistic alpha (signal) under strict real-world constraints where temporal identifiers are stripped and sequence models cannot be used.

---

## 📌 The Quantitative Challenge
Financial markets do not reward hindsight; they reward foresight. The objective of this repository is to predict the `TARGET` (defined as the H-window-forward percentage change in 'Price') using a combination of current state measurements and predefined historical lag differences (`LagT1`, `LagT2`, `LagT3`). 

**The Constraints:**
* **No Temporal Identifiers:** The dataset consists of shuffled row-level samples. Traditional sequential models (LSTMs, GRUs, ARIMA) are mathematically impossible to apply.
* **Pure Tabular Extraction:** Success relies entirely on learning predictive structure from cross-sectional features and hardcoded historical differences.
* **Evaluation Metric:** Coefficient of Determination ($R^2$). In short-horizon financial forecasting, capturing even a fraction of the variance (e.g., $R^2$ of 0.02) represents highly profitable market alpha.

---

## 🚀 The Alpha Extraction Strategy
To solve this, the pipeline bypasses deep learning sequence models and leverages a highly tuned **Gradient Boosting Framework (LightGBM)**, supported by aggressive memory optimization and row-wise feature engineering.

### 1. Row-Wise Feature Engineering
Since rows are shuffled, we cannot engineer time-based rolling means. Instead, we engineer spatial relationships across the features *within* the same row:
* **Row-Wise Volatility:** Computed the Standard Deviation across all historical lag features for a given row.
* **Row-Wise Momentum:** Computed the mean movement and specific feature acceleration (e.g., `featureX_LagT1` - `featureX_LagT2`) to capture directional velocity without crossing data samples.

### 2. Aggressive Memory Compression
To survive local and Kaggle-cloud RAM limits while processing massive DataFrames, a custom memory reduction algorithm aggressively downcasts 64-bit floats and integers to their lowest possible memory footprint without losing precision.

### 3. LightGBM Ensemble Architecture
A high-capacity `LGBMRegressor` is trained using a customized hyperparameter grid:
* **Objective:** Regression (RMSE metric)
* **Regularization:** Heavy feature fraction (`0.83`) and bagging fraction (`0.97`) to aggressively combat overfitting to market noise.
* **Capacity:** Deep tree depth (`13`) and high leaf count (`106`) to capture complex non-linear interactions across the anonymized variables.

---

## 💻 Setup & Reproducibility

### Prerequisites

- Python 3.9+
- `lightgbm`
- `pandas`
- `numpy`
- `scikit-learn`

---

## ⚙️ Installation & Execution

### 1. Clone the Repository

```bash
git clone https://github.com/ayushsin9h/iRage-AlgoArena.git
cd iRage-AlgoArena
pip install -r requirements.txt
```

### 2. Acquire the Dataset

Due to GitHub file size constraints and data licensing, the raw `.parquet` files are not hosted in this repository.

Download the official dataset from the [iRage Kaggle Competition](https://www.kaggle.com/competitions/short-horizon-return-prediction-challenge-by-i-rage/data?utm_source=chatgpt.com).

Place the following files directly into the `data/raw/` directory:

Place the following files directly into the `data/raw/` directory:

- `train.parquet`
- `test.parquet`
- `sample_submission.csv`

---

## ⚖️ Legal Disclaimer

The datasets utilized in this project are proprietary to iRage and were provided strictly for the duration of the *Short-Horizon Return Prediction Challenge*. The original raw data is not distributed within this repository.

The analytical code and feature engineering pipelines are released under the MIT License for educational and portfolio demonstration purposes.

----

## 📂 Production MLOps Architecture

```text
iRage-AlgoArena/
├── data/
│   ├── raw/                  # iRage training/testing datasets (Git-ignored)
│   └── submissions/          # Generated Alpha signals (e.g., submission.csv)
├── notebooks/                
│   └── mycode_ayushs1ngh_cpu.ipynb  # Reproducible Kaggle submission notebook
├── src/                      # Extracted production logic
│   ├── feature_engineering.py       # Downcasting and quant momentum features
│   └── train.py                     # LightGBM model initialization and training
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
