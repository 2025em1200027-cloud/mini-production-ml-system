import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from serving.schemas import CustomerFeatures, PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/churn_model.joblib")

app = FastAPI(title="Telco Customer Churn API", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model = None
    model_load_error = str(exc)
else:
    model_load_error = None


def fallback_heuristic(customer: CustomerFeatures) -> tuple[float, str]:
    # Simple rule-based backup if the ML model is unavailable or fails
    risk_score = 0.2
    if customer.Contract == "Month-to-month":
        risk_score += 0.35
    if customer.tenure < 6:
        risk_score += 0.25
    if customer.MonthlyCharges > 80:
        risk_score += 0.15
    prob = min(round(risk_score, 4), 0.99)
    label = "Yes" if prob >= 0.5 else "No"
    return prob, label


@app.get("/health")
def health():
    if model is None:
        return {"status": "degraded", "model_loaded": False, "detail": model_load_error}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        prob, label = fallback_heuristic(customer)
        return PredictionResponse(churn_probability=prob, churn_prediction=label, used_fallback=True)

    try:
        input_df = pd.DataFrame([customer.model_dump()])
        prob = float(model.predict_proba(input_df)[0, 1])
        label = "Yes" if prob >= 0.5 else "No"
        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=label,
            used_fallback=False,
        )
    except Exception as exc:
        try:
            prob, label = fallback_heuristic(customer)
            return PredictionResponse(
                churn_probability=prob,
                churn_prediction=label,
                used_fallback=True,
            )
        except Exception:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
