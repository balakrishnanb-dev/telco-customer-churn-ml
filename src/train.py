"""Train, compare, and evaluate churn classifiers."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from src.data import load_data, clean_data
from src.features import FEATURES, TARGET, ID_COLUMN, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
RANDOM_STATE = 42


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_data())
    customer_ids = df[ID_COLUMN].copy()
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    }
    cv_rows, pipelines = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_rows.append({"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()})
        pipelines[name] = pipe
        print(f"{name}: CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    cv_df.to_csv(ARTIFACTS / "cross_validation_results.csv", index=False)
    best_name = str(cv_df.iloc[0]["model"])
    best_model = pipelines[best_name].fit(X_train, y_train)

    probabilities = best_model.predict_proba(X_test)[:, 1]
    # Threshold 0.5 is kept as a transparent baseline; campaign thresholds need cost analysis.
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "selected_model": best_name,
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_average_precision": float(average_precision_score(y_test, probabilities)),
        "test_precision_at_0_5": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall_at_0_5": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1_at_0_5": float(f1_score(y_test, predictions, zero_division=0)),
        "test_size": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "threshold": 0.5,
        "random_state": RANDOM_STATE,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(ARTIFACTS / "model_metrics.csv", index=False)
    pd.DataFrame({"customerID": id_test.values, "actual_churn": y_test.values,
                  "churn_probability": probabilities, "predicted_churn_at_0_5": predictions}).to_csv(
                      OUTPUTS / "test_predictions.csv", index=False)
    cm = confusion_matrix(y_test, predictions).tolist()
    (ARTIFACTS / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["Retained", "Churned"]) +
        "\nConfusion matrix [ [TN, FP], [FN, TP] ]:\n" + str(cm) + "\n"
    )
    joblib.dump(best_model, ARTIFACTS / "churn_pipeline.joblib")

    # Permutation importance on the held-out test set; interpret as predictive association only.
    from sklearn.inspection import permutation_importance
    result = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                                    n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.DataFrame({"feature": FEATURES, "importance_mean": result.importances_mean,
                               "importance_std": result.importances_std}).sort_values(
                                   "importance_mean", ascending=False)
    importance.to_csv(ARTIFACTS / "permutation_importance.csv", index=False)
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nTop permutation importances:")
    print(importance.head(10).to_string(index=False))
    print(f"\nSaved model and results under {ARTIFACTS}")


if __name__ == "__main__":
    main()
