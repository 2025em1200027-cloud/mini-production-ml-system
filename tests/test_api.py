from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure": 5,
    "MonthlyCharges": 89.10,
    "TotalCharges": 445.50,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ["Yes", "No"]


def test_predict_rejects_malformed_payload():
    bad_payload = VALID_CUSTOMER.copy()
    bad_payload["tenure"] = -5  # Invalid negative tenure
    bad_payload["Contract"] = "Lifetime"  # Unobserved/invalid category
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
