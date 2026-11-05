# ParaBoost Credit Card Fraud Benchmark 💳⚖️

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-FF9D00?style=for-the-badge&logo=optuna&logoColor=white)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end benchmarking suite for credit card fraud detection on extreme class-imbalance datasets. This repository compares a proprietary/custom **ParaBoostForest** classifier against robust baselines (**RandomForest**, **XGBoost**) under a rigorous and fair optimization protocol.

---

## 🚀 Key Features

- **Extreme Imbalance Handling:** Tailored evaluation protocols focusing on Precision-Recall curves, PR-AUC, and F1-score optimization rather than misleading accuracy.
- **Automated Hyperparameter Tuning:** Integrates **Optuna** (with TPE Sampler and Median Pruner) for unbiased, robust model tuning.
- **Dynamic Threshold Selection:** Automatically determines the optimal decision threshold from validation Precision-Recall curves to maximize F1-score under heavy skew.
- **Advanced Metrics & Diagnostics:** Evaluates Brier Score, Expected Calibration Error (ECE), Kolmogorov-Smirnov (KS) statistic, Precision@K, Recall@K, Cumulative Gains, and Lift Curves.
- **Publication-Grade Visualizations:** Automatically generates and saves high-resolution comparison plots (ROC, PR, Calibration, Confusion Matrices, Probability Densities, and Gains/Lift charts) inside `outputs/plots/`.

---

## 📂 Project Structure
```text
├── paraboost_forest.py      # Core implementation of ParaBoostForest classifier
├── benchmark.py             # Main evaluation, Optuna tuning, threshold selection & plotting pipeline
├── creditcard.csv           # Dataset (Kaggle Credit Card Fraud Detection)
└── outputs/
├── plots/               # Generated publication-grade evaluation charts
└── results.csv          # Consolidated metrics and performance comparison table
