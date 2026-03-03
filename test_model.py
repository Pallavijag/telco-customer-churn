import os
import joblib
import pandas as pd

print("Running from:", os.getcwd())
print("Files here:", os.listdir())

artifact = joblib.load("models/churn_pipeline.joblib")
print("Artifact keys:", artifact.keys())

pipe = artifact["pipeline"]
features = artifact["features"]

df = pd.read_csv("data/cleaned_telco_data.csv")

X = df[features]
y = df["Churn Value"]

preds = pipe.predict(X)

print("Predicted churn count:", int(preds.sum()))
print("Actual churn count:", int(y.sum()))