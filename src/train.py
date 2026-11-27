import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from data_pipeline import load_and_split_data, build_preprocessor

def evaluate_predictions(model, X, y, name: str):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, probs)
    f1_macro = f1_score(y, preds, average="macro")
    print(f"[{name}] Validation PR-AUC: {pr_auc:.4f} | F1-Macro: {f1_macro:.4f}")
    return pr_auc

def main():
    print("Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # 1. Baseline Model: Logistic Regression
    baseline_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    evaluate_predictions(baseline_pipeline, X_val, y_val, "Baseline (Logistic Regression)")
    
    # 2. Primary Model: Random Forest (Cost-sensitive via class_weight)
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    evaluate_predictions(rf_pipeline, X_val, y_val, "Primary Model (Random Forest)")
    
    # 3. Serialize the production artifact
    os.makedirs("artifacts", exist_ok=True)
    artifact_path = "artifacts/churn_model.joblib"
    joblib.dump(rf_pipeline, artifact_path)
    print(f"\nModel pipeline successfully saved to {artifact_path}")

if __name__ == "__main__":
    main()
