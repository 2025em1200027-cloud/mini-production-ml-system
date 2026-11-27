from typing import Literal
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Months with the company")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges so far")

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: Literal["Yes", "No"]
    used_fallback: bool = False
