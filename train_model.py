import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset/train.csv")

print("Dataset Loaded Successfully")
print("Columns:", df.columns.tolist())

# Drop Loan_ID
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values for categorical columns
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing values for numeric columns
numeric_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
label_encoder = LabelEncoder()

for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# Verify target column exists
if "Loan_Status" not in df.columns:
    raise Exception("Loan_Status column not found in dataset")

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train XGBoost Model
model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nTraining Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy :", round(test_accuracy * 100, 2), "%")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nmodel.pkl created successfully!")