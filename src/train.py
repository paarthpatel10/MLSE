import pandas as pd
import yaml
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load training data
train = pd.read_csv("data/processed/train.csv")
target = params["target"]

X_train = train.drop(columns=[target])
y_train = train[target]

# Select model
model_type = params["model"]["type"]

if model_type == "RandomForestClassifier":
    model = RandomForestClassifier(**params["model"]["params"])

elif model_type == "LogisticRegression":
    # Ensure max_iter is present
    model = LogisticRegression(max_iter=500, **params["model"]["params"])

elif model_type == "SVC":
    model = SVC(**params["model"]["params"])

elif model_type == "KNeighborsClassifier":
    model = KNeighborsClassifier(**params["model"]["params"])

else:
    raise ValueError(f"Unsupported model type: {model_type}")

# Train model
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print(f"✅ {model_type} trained and saved.")
