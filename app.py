"""Streamlit interface for exploring churn and scoring an example customer."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.data import load_data, clean_data
from src.features import FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"

st.set_page_config(page_title="Telco Churn Insights", page_icon="📊", layout="wide")
st.title("Telco Customer Churn | Prediction & Retention Insights")
st.caption("Portfolio project using IBM's public sample dataset. Scores are estimates, not certainties.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python -m src.download_data` and `python -m src.train` first.")
    st.stop()
try:
    df = clean_data(load_data())
except Exception as exc:
    st.error(str(exc)); st.stop()
model = joblib.load(MODEL_PATH)

rate = df["Churn"].mean()
c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Observed churn rate", f"{rate:.1%}")
c3.metric("Average tenure", f"{df['tenure'].mean():.1f} months")

st.subheader("Explore churn patterns")
left, right = st.columns(2)
with left:
    contract = df.groupby("Contract", observed=False)["Churn"].mean().sort_values(ascending=False)
    st.bar_chart(contract.rename("Churn rate"))
with right:
    tenure_bins = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"])
    tenure_rates = df.assign(tenure_band=tenure_bins).groupby("tenure_band", observed=False)["Churn"].mean()
    st.bar_chart(tenure_rates.rename("Churn rate"))

st.subheader("Score a customer profile")
st.write("Adjust the example inputs to estimate churn probability. The prediction is for demonstration and should not be used as a real customer decision without validation.")
with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract_value = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    with col2:
        monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0, 1.0)
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        partner = st.selectbox("Has partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["Yes", "No"])
    with col3:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
    submitted = st.form_submit_button("Estimate churn risk")

if submitted:
    # TotalCharges is derived as a simple illustrative approximation for a hypothetical profile.
    row = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": tech_support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract_value, "PaperlessBilling": paperless, "PaymentMethod": payment,
    }])
    probability = float(model.predict_proba(row[FEATURES])[:, 1][0])
    st.metric("Estimated churn probability", f"{probability:.1%}")
    if probability >= 0.65:
        st.warning("Higher estimated risk in this model. Consider reviewing the profile and validating the model before any action.")
    elif probability >= 0.35:
        st.info("Moderate estimated risk. This is an uncertain prediction, not a conclusion about an individual.")
    else:
        st.success("Lower estimated risk in this model. Low risk does not mean zero risk.")

importance_path = ROOT / "artifacts" / "permutation_importance.csv"
if importance_path.exists():
    st.subheader("Model interpretation")
    imp = pd.read_csv(importance_path).head(10).sort_values("importance_mean")
    st.bar_chart(imp.set_index("feature")["importance_mean"])
    st.caption("Permutation importance measures how much test ROC-AUC changes when a feature is shuffled; it does not establish causality.")
