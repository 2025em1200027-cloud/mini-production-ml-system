# mini-production-ml-system
 Customer Churn Prediction (Classification - Imbalanced Data): Predict whether a subscriber will cancel before the next cycle. Key Focus: Handling skewed distributions via cost-sensitive learning / weighted loss; temporal data split to prevent look-ahead leakage; batch or event-driven scoring; evaluation via PR-AUC, Recall@Precision, and F1-macro.
mini-production-ml-system/
├── README.md               (Instructions on how to run your code)
├── requirements.txt        (List of Python libraries you used)
├── data/                   (Put your downloaded CSV file here)
├── artifacts/              (Your saved model files will go here)
├── src/
│   ├── data_pipeline.py    (Code to clean and split data)
│   ├── train.py            (Code to train and save the model)
│   └── evaluate.py         (Code to test accuracy/metrics)
├── serving/
│   ├── app.py              (The web API that makes predictions)
│   ├── schemas.py          (Rules for what valid input looks like)
│   └── benchmark.py        (Script to test how fast your API is)
└── tests/
    └── test_api.py         (1 or 2 simple automated tests)


    # Mini Production ML System: Telco Customer Churn Prediction

End-to-end production machine learning pipeline for predicting customer churn using a cost-sensitive Random Forest classifier served via FastAPI.

## Project Structure
- `src/data_pipeline.py`: Data ingestion, cleaning, stratified splitting (70/15/15), and leak-free `ColumnTransformer`.
- `src/train.py`: Trains baseline Logistic Regression and primary Random Forest models; serializes pipeline to `artifacts/churn_model.joblib`.
- `src/evaluate.py`: Computes PR-AUC, F1-Macro, Recall, and generates `artifacts/confusion_matrix.png`.
- `serving/schemas.py`: Pydantic input/output validation schemas.
- `serving/app.py`: FastAPI REST service (`/health`, `/predict`) with rule-based fallback logic.
- `serving/benchmark.py`: Latency (p50/p95/p99) and throughput profiler.
- `tests/test_api.py`: Automated schema and endpoint tests.

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
uvicorn serving.app:app --host 127.0.0.1 --port 8000
