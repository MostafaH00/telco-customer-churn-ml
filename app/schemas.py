
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Reusable categorical types
YesNo = Literal["Yes", "No"]

InternetOption = Literal[
    "Yes",
    "No",
    "No internet service",
]


class CustomerFeatures(BaseModel):
    """Input features required for churn prediction."""

    model_config = ConfigDict(extra="forbid")

    # Customer information
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)

    # Phone services
    PhoneService: YesNo
    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    # Internet services
    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption

    # Contract and billing
    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    # Numerical charges
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)



class ChurnPrediction(BaseModel):
    """Response schema for a churn prediction."""

    churn_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    churn_prediction: Literal[0, 1]
