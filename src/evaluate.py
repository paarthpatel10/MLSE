import pandas as pd
import joblib
import yaml
import json
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load test data
test = pd.read_csv("data/processed/test.csv")
X_test = test.drop(columns=["target"])
y_test = test["target"]

# Load model
model = joblib.load("models/model.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred, average="weighted"),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# Save results
os.makedirs("experiments", exist_ok=True)
with open("experiments/results.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation done. Results saved to experiments/results.json")
