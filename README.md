# Telco Customer Churn: Prediction & Retention Insights

An end-to-end data science project that analyses customer churn, compares machine-learning models, explains risk drivers, and provides an interactive app for exploring churn risk.

## Problem
Telecom providers lose recurring revenue when customers cancel subscriptions. This project uses customer account, service, and billing data to estimate churn risk and identify patterns that may inform retention campaigns.

## What this project demonstrates
- Data validation and cleaning (including blank `TotalCharges` values)
- Exploratory data analysis and business-focused visualisations
- Feature preprocessing with `ColumnTransformer` and leakage-safe pipelines
- Model comparison: Logistic Regression and Random Forest
- Stratified train/test split and cross-validation on training data
- Evaluation with ROC-AUC, average precision, precision, recall, F1, and confusion matrix
- Permutation importance for model interpretation
- A Streamlit dashboard for data exploration and individual risk scoring
- Basic automated tests

## Dataset
IBM Telco Customer Churn sample dataset: 7,043 customer records. Each row represents a customer; `Churn` indicates whether they left. The dataset is a cross-sectional sample, not a time series, so results should not be described as a forecast for a specific future month or as evidence that retention actions cause customers to stay.

Download source: [IBM's public sample dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The raw CSV is not committed to this repository. Download it with:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data
```

## Run the project

```bash
python -m src.eda
python -m src.train
streamlit run app.py
```

Training writes model metrics, test predictions, a serialized pipeline, and permutation-importance results under `artifacts/` and `outputs/`. The Streamlit app expects the dataset and a trained model; run the steps above first.

Run tests:

```bash
pytest -q
```

## Project structure

```text
├── app.py                      # Interactive Streamlit app
├── data/raw/                   # Downloaded raw dataset (not committed)
├── outputs/                    # EDA figures and prediction outputs
├── artifacts/                  # Trained model and evaluation outputs
├── src/
│   ├── download_data.py        # Fetch dataset from IBM's public repository
│   ├── data.py                 # Data loading, validation, cleaning
│   ├── features.py             # Feature/target separation and preprocessing
│   ├── eda.py                  # EDA figures and summary tables
│   └── train.py                # CV model selection, test evaluation, importance
├── tests/                      # Data and feature tests
├── requirements.txt
└── README.md
```

## Important interpretation notes
- Risk scores are estimates from this dataset and should not be treated as certainty.
- Feature importance describes predictive association, not causation.
- A real deployment would require time-based validation, monitoring for drift, privacy review, and testing whether retention interventions actually work.

## Next improvements
1. Add a time-based dataset and evaluate temporal generalisation.
2. Calibrate predicted probabilities and choose a threshold based on campaign capacity/cost.
3. Add SHAP explanations and model monitoring.
4. Deploy the app and document a reproducible experiment run.
