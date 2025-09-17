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

# Use target from params.yaml
target = params["target"]

# Drop Id column if present (not useful for training)
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Encode target if categorical
if y.dtype == "object":
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

# Encode categorical features if requested
if params["preprocessing"]["encode"]:
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

# Scale features if requested
if params["preprocessing"]["scale"]:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["training"]["test_size"], random_state=params["training"]["random_state"]
)

# Reset indices to avoid NaNs
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = pd.Series(y_train, name=target).reset_index(drop=True)
y_test = pd.Series(y_test, name=target).reset_index(drop=True)

# Build final DataFrames
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Save processed data
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

# Debug check
print("✅ Preprocessing done. Train and test saved.")
print("NaNs in train.csv:\n", train.isna().sum())
print("NaNs in test.csv:\n", test.isna().sum())
