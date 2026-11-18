
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.predictor import load_model, predict_customer
from app.schemas import CustomerFeatures, ChurnPrediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    load_model()
    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "REST API for predicting customer churn "
        "using a trained XGBoost pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Check whether the API is ready."""
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/model-info")
def model_info():
    """Return information about the deployed model."""
    artifact = load_model()

    pipeline = artifact["pipeline"]

    return {
        "model_type": type(pipeline.steps[-1][1]).__name__,
        "threshold": float(artifact["threshold"]),
        "n_features": len(artifact["feature_names"]),
        "positive_class": int(artifact["positive_class"]),
    }


@app.post(
    "/predict",
    response_model=ChurnPrediction,
)
def predict(customer: CustomerFeatures):
    """Predict churn probability for one customer."""
    return predict_customer(customer)
