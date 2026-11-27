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
