# ParaBoost Credit Card Fraud Benchmark 💳⚖️

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-FF9D00?style=for-the-badge&logo=optuna&logoColor=white)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end benchmarking suite for credit card fraud detection on extreme class-imbalance datasets. This repository compares a proprietary/custom **ParaBoostForest** classifier against robust baselines (**RandomForest**, **XGBoost**) under a rigorous and fair optimization protocol.

---

## 💡 About the Project

Real-world financial fraud detection datasets—such as credit card transaction records—suffer from extreme class imbalance, where fraudulent cases constitute a microscopic fraction of the overall data. Standard classification models and conventional metrics (like raw accuracy or fixed $0.5$ decision thresholds) fail completely in this regime, often resulting in high false-negative rates or misleadingly optimistic performance reports.

This project implements a rigorous, end-to-end benchmarking framework designed specifically to address extreme skewness. By integrating automated hyperparameter optimization via **Optuna**, dynamic probability threshold tuning on Precision-Recall curves, and a comprehensive set of calibration and ranking diagnostics, this suite ensures that model comparison is both fair and clinically/financially meaningful. The core engine evaluates custom ensemble strategies (`ParaBoostForest`) alongside established industrial baselines like `RandomForest` and `XGBoost`, providing deep statistical insights into probabilistic reliability, early-stage fraud capture, and decision-boundary stability.

---

## 🚀 Key Features

- **Extreme Imbalance Handling:** Tailored evaluation protocols focusing on Precision-Recall curves, PR-AUC, and F1-score optimization rather than misleading accuracy.
- **Automated Hyperparameter Tuning:** Integrates **Optuna** (with TPE Sampler and Median Pruner) for unbiased, robust model tuning.
- **Dynamic Threshold Selection:** Automatically determines the optimal decision threshold from validation Precision-Recall curves to maximize F1-score under heavy skew.
- **Advanced Metrics & Diagnostics:** Evaluates Brier Score, Expected Calibration Error (ECE), Kolmogorov-Smirnov (KS) statistic, Precision@K, Recall@K, Cumulative Gains, and Lift Curves.
- **Publication-Grade Visualizations:** Automatically generates and saves high-resolution comparison plots (ROC, PR, Calibration, Confusion Matrices, Probability Densities, and Gains/Lift charts) inside dedicated output directories.

---

## 📊 Evaluation Metrics

The pipeline records and exports a rich set of evaluation indicators:
1. **Ranking & Probabilistic Quality:** ROC-AUC, PR-AUC, Brier Loss, Expected Calibration Error (ECE), and KS-Statistic.
2. **Classification Performance:** Precision, Recall, F1-Score, and Accuracy (at both dynamic PR-optimized thresholds and fixed 0.5 thresholds).
3. **Information Retrieval Metrics:** Precision@K and Recall@K (evaluated across logarithmic steps).
4. **Business/Financial Impact:** Cumulative Gains and Lift Curves to measure early fraud capture efficiency.

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.11 or higher installed. Install the required dependencies:
```bash
pip install numpy pandas scikit-learn optuna xgboost lightgbm imbalanced-learn matplotlib seaborn
