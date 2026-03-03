import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

artifact = joblib.load(MODEL_PATH)

if isinstance(artifact, dict):
    pipe = artifact.get("pipeline") or artifact.get("model")
    threshold = float(artifact.get("threshold", 0.5))
else:
    pipe = artifact
    threshold = 0.5

if pipe is None:
    st.error("Model artifact is missing a 'pipeline' or 'model' key.")
    st.stop()

def get_classifier(estimator):
    if hasattr(estimator, "named_steps"):
        for key in ["clf", "model", "lr", "logreg", "classifier"]:
            if key in estimator.named_steps:
                return estimator.named_steps[key]
        return list(estimator.named_steps.values())[-1]
    return estimator

clf = get_classifier(pipe)

def churn_probability(estimator, X_df):
    probs = estimator.predict_proba(X_df)[0]
    local_clf = get_classifier(estimator)

    if hasattr(local_clf, "classes_"):
        classes = list(local_clf.classes_)
        if 1 in classes:
            return float(probs[classes.index(1)])
        if "Yes" in classes:
            return float(probs[classes.index("Yes")])

    return float(probs[-1])

st.title("Customer Churn Prediction")
st.caption(f"Decision Threshold: {threshold}")

st.subheader("Customer Details")

c1, c2 = st.columns(2)

with c1:
    tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=10, step=1)
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0, format="%.2f")
    total = st.number_input("Total Charges", min_value=0.0, max_value=50000.0, value=700.0, step=10.0, format="%.2f")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)

with c2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=2)
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=2)
    tech_support = st.selectbox("Tech Support", ["Yes", "No"], index=1)
    online_security = st.selectbox("Online Security", ["Yes", "No"], index=1)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)

X_input = pd.DataFrame([{
    "Tenure Months": tenure,
    "Monthly Charges": monthly,
    "Total Charges": total,
    "Contract": contract.strip(),
    "Internet Service": internet.strip(),
    "Payment Method": payment.strip(),
    "Tech Support": tech_support.strip(),
    "Online Security": online_security.strip(),
    "Paperless Billing": paperless.strip(),
}])

st.markdown("---")

if st.button("Predict Churn"):
    prob = churn_probability(pipe, X_input)
    pred = int(prob >= threshold)

    st.metric("Churn Probability", f"{prob:.3f}")

    if pred == 1:
        st.error("⚠️ Customer likely to CHURN")
    else:
        st.success("✅ Customer likely to STAY")