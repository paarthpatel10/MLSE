import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load dataset
df = pd.read_csv(params["dataset"])

# Example: Assume target column is "target"
target = "target"
X = df.drop(columns=[target])
y = df[target]

# Encoding
if params["preprocessing"]["encode"]:
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

# Scaling
if params["preprocessing"]["scale"]:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["training"]["test_size"], random_state=params["training"]["random_state"]
)

# Save processed data
os.makedirs("data/processed", exist_ok=True)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✅ Preprocessing done. Train and test saved.")
