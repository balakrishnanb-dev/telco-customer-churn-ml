"""Dataset loading, schema validation, and cleaning."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
REQUIRED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `python -m src.download_data` first.")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df["customerID"].duplicated().any():
        raise ValueError("customerID contains duplicate values")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; do not mutate the original dataframe."""
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    # In this dataset, blank TotalCharges values are associated with zero-tenure customers.
    zero_tenure = cleaned["tenure"].eq(0)
    cleaned.loc[zero_tenure & cleaned["TotalCharges"].isna(), "TotalCharges"] = 0.0
    # Any remaining invalid values are kept as NaN for the pipeline's imputer.
    cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
    if cleaned["Churn"].isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No'")
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype("category")
    return cleaned
