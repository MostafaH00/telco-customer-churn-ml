
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.predictor import load_model


@pytest.fixture(scope="module")
def client():
    """Create a test client and run the API lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_customer():
    """Return a valid customer example."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 70.70,
    }


def test_health(client):
    """The API should report that it is healthy."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info(client):
    """Model metadata should match the saved artifact."""
    response = client.get("/model-info")
    artifact = load_model()

    assert response.status_code == 200
    assert response.json()["n_features"] == 19
    assert response.json()["threshold"] == pytest.approx(
        artifact["threshold"]
    )


def test_valid_prediction(client, sample_customer):
    """A valid customer should produce a valid prediction."""
    response = client.post(
        "/predict",
        json=sample_customer,
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]

    expected_class = int(
        result["churn_probability"] >= result["threshold"]
    )

    assert result["churn_prediction"] == expected_class


def test_prediction_matches_pipeline(client, sample_customer):
    """API output must match the original trained pipeline."""
    artifact = load_model()

    X = pd.DataFrame(
        [sample_customer],
        columns=artifact["feature_names"],
    )

    classes = list(artifact["pipeline"].classes_)
    positive_index = classes.index(
        artifact["positive_class"]
    )

    expected_probability = float(
        artifact["pipeline"].predict_proba(X)[
            0, positive_index
        ]
    )

    response = client.post(
        "/predict",
        json=sample_customer,
    )

    assert response.status_code == 200
    assert response.json()["churn_probability"] == pytest.approx(
        expected_probability
    )


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("tenure", -5),
        ("Contract", "Weekly"),
        ("MonthlyCharges", -20),
    ],
)
def test_invalid_values(
    client, sample_customer, field, invalid_value
):
    """Invalid values must be rejected."""
    sample_customer[field] = invalid_value

    response = client.post(
        "/predict",
        json=sample_customer,
    )

    assert response.status_code == 422


def test_missing_feature(client, sample_customer):
    """Missing required features must be rejected."""
    del sample_customer["tenure"]

    response = client.post(
        "/predict",
        json=sample_customer,
    )

    assert response.status_code == 422


def test_extra_feature(client, sample_customer):
    """Unexpected input features must be rejected."""
    sample_customer["Churn"] = "Yes"

    response = client.post(
        "/predict",
        json=sample_customer,
    )

    assert response.status_code == 422
