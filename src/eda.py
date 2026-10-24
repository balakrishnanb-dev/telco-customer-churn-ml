"""Generate exploratory data analysis charts and summary tables."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data import load_data, clean_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    raw = load_data()
    df = clean_data(raw)
    summary = pd.DataFrame({
        "metric": ["rows", "columns", "overall_churn_rate", "duplicate_customer_ids", "missing_total_charges_after_cleaning"],
        "value": [len(df), len(df.columns), round(df["Churn"].mean(), 4), int(df["customerID"].duplicated().sum()), int(df["TotalCharges"].isna().sum())],
    })
    summary.to_csv(OUT / "eda_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set(title="Customer churn distribution", xlabel="Churn (0 = retained, 1 = churned)", ylabel="Customers")
    fig.tight_layout(); fig.savefig(OUT / "churn_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    contract_rate = df.groupby("Contract", observed=False)["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=contract_rate.index, y=contract_rate.values, ax=ax)
    ax.set(title="Churn rate by contract type", xlabel="Contract", ylabel="Churn rate")
    ax.set_ylim(0, 1); fig.tight_layout(); fig.savefig(OUT / "churn_by_contract.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=24, stat="density", common_norm=False, element="step", ax=ax)
    ax.set(title="Tenure distribution by churn status", xlabel="Tenure (months)", ylabel="Density")
    fig.tight_layout(); fig.savefig(OUT / "tenure_by_churn.png", dpi=160); plt.close(fig)
    print(summary.to_string(index=False))
    print(f"EDA outputs saved to {OUT}")


if __name__ == "__main__":
    main()
