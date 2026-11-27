import os
import glob
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

def load_and_split_data():
    path = kagglehub.dataset_download("shobhak27/clean-customer-churn-dataset")

    files = glob.glob(os.path.join(path, "*"))
    excel_files = [f for f in files if f.endswith((".xlsx", ".xls"))]
    csv_files = [f for f in files if f.endswith(".csv")]

    if excel_files:
        df = pd.read_excel(excel_files[0])
    elif csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(f"No CSV or Excel file found in {path}")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor