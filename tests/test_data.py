import pandas as pd
from src.data import clean_data


def test_clean_data_converts_total_charges_and_target_without_mutating_input():
    raw = pd.DataFrame({
        "customerID": ["A", "B"], "tenure": [0, 4], "TotalCharges": ["", "120.50"],
        "Churn": ["No", "Yes"], "SeniorCitizen": [0, 1]
    })
    cleaned = clean_data(raw)
    assert pd.isna(raw.loc[0, "TotalCharges"]) is False  # original value remains the empty string
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 120.50
    assert cleaned["Churn"].tolist() == [0, 1]
    assert str(cleaned["SeniorCitizen"].dtype) == "category"
