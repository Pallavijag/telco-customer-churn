import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# ----------------------------
# Load CLEANED (BEFORE ENCODING) dataset
# Must contain raw categorical columns like Contract, Payment Method etc.
# and target column: Churn Value (0/1)
# ----------------------------
df = pd.read_csv("data/cleaned_telco_data.csv")

# Safety: drop leakage columns if present
leak_cols = ["Churn Label", "Churn Score", "Churn Reason"]
df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

target = "Churn Value"
assert target in df.columns, "Churn Value column missing in cleaned_telco_data.csv"

# ----------------------------
# Keep only the features you want in Streamlit (simple + usable)
# ----------------------------
FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract",
    "Internet Service",
    "Payment Method",
    "Tech Support",
    "Online Security",
    "Paperless Billing",
]

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in cleaned_telco_data.csv: {missing}")

X = df[FEATURES].copy()
y = df[target].astype(int)

# ----------------------------
# Train/test split (stratified)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Preprocessing
# ----------------------------
num_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
cat_cols = [c for c in FEATURES if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# ----------------------------
# Model pipeline
# ----------------------------
model = LogisticRegression(max_iter=3000)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# ----------------------------
# Cross-validation (proper, no leakage)
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

print("CV ROC-AUC scores:", cv_auc)
print("Mean CV ROC-AUC:", cv_auc.mean())
print("Std CV ROC-AUC:", cv_auc.std())

# ----------------------------
# Fit + evaluate
# ----------------------------
pipe.fit(X_train, y_train)

y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.4).astype(int)   # choose your threshold here

print("\nTest ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------------
# Save artifact (pipeline + dropdown options)
# ----------------------------
dropdown_options = {}
for c in cat_cols:
    dropdown_options[c] = sorted(X[c].dropna().unique().tolist())

artifact = {
    "pipeline": pipe,
    "threshold": 0.4,
    "features": FEATURES,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "dropdown_options": dropdown_options,
}

joblib.dump(artifact, "models/churn_pipeline.joblib")
print("\n✅ Saved: churn_pipeline.joblib")

import os
print("Saved at:", os.path.abspath("churn_pipeline.joblib"))
print("Current folder:", os.getcwd())