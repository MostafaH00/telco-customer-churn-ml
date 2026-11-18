
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import CustomerFeatures


# Absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "final_churn_model.joblib"


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model artifact once."""

    artifact = joblib.load(MODEL_PATH)

    return artifact


def predict_customer(customer: CustomerFeatures) -> dict:
    """Predict churn for a single customer."""

    # Load the trained artifact
    artifact = load_model()

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    feature_names = list(artifact["feature_names"])
    positive_class = artifact["positive_class"]

    # Convert validated input into a DataFrame
    customer_data = customer.model_dump()

    X = pd.DataFrame(
        [customer_data],
        columns=feature_names,
    )

    # Predict class probabilities
    probabilities = pipeline.predict_proba(X)[0]

    # Find the churn class
    classes = list(pipeline.classes_)
    positive_index = classes.index(positive_class)

    churn_probability = float(
        probabilities[positive_index]
    )

    # Apply the optimized classification threshold
    churn_prediction = int(
        churn_probability >= threshold
    )

    return {
        "churn_probability": churn_probability,
        "threshold": threshold,
        "churn_prediction": churn_prediction,
    }
