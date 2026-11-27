import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from data_pipeline import load_and_split_data


def main():
    _, X_val, X_test, _, y_val, y_test = load_and_split_data()

    model_path = "artifacts/churn_model.joblib"
    model = joblib.load(model_path)

    for split_name, X_split, y_split in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        preds = model.predict(X_split)
        probs = model.predict_proba(X_split)[:, 1]

        pr_auc = average_precision_score(y_split, probs)
        f1_macro = f1_score(y_split, preds, average="macro")
        recall_churn = recall_score(y_split, preds, pos_label=1)
        precision_churn = precision_score(y_split, preds, pos_label=1)

        print(f"=== {split_name} Set Metrics ===")
        print(f"PR-AUC:           {pr_auc:.4f}")
        print(f"F1-Macro:         {f1_macro:.4f}")
        print(f"Churn Recall:     {recall_churn:.4f}")
        print(f"Churn Precision:  {precision_churn:.4f}")
        print(classification_report(y_split, preds, target_names=["No Churn", "Churn"]))

    # Save confusion matrix on the test set for the report
    os.makedirs("artifacts", exist_ok=True)
    y_test_preds = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_preds, display_labels=["No Churn", "Churn"], cmap="Blues"
    )
    plt.title("Test Set Confusion Matrix (Random Forest)")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to artifacts/confusion_matrix.png")


if __name__ == "__main__":
    main()