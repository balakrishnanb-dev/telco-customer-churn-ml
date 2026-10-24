import pandas as pd
from src.features import FEATURES, make_preprocessor


def test_preprocessor_handles_unseen_categories():
    row_a = {name: 1 for name in FEATURES}
    row_b = {name: 2 for name in FEATURES}
    for name in FEATURES:
        if name not in {"tenure", "MonthlyCharges", "TotalCharges"}:
            row_a[name] = "category-a"
            row_b[name] = "category-b"
    train = pd.DataFrame([row_a, row_a])
    unseen = pd.DataFrame([row_b])
    preprocessor = make_preprocessor().fit(train)
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
