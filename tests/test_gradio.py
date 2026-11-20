
from unittest.mock import Mock, patch

import gradio as gr
import pytest
import requests

from app.gradio_ui import (
    API_URL,
    FEATURE_NAMES,
    predict_churn,
    validate_customer,
)


@pytest.fixture
def valid_customer():
    """Return a valid customer profile."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 10,
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
        "MonthlyCharges": 70,
        "TotalCharges": 840,
    }


def test_valid_customer(valid_customer):
    """A valid customer should pass validation."""
    assert validate_customer(valid_customer) is None


@pytest.mark.parametrize(
    "changes, error_message",
    [
        ({"tenure": -1}, "Tenure cannot be negative"),
        (
            {"MonthlyCharges": -10},
            "Monthly charges cannot be negative",
        ),
        (
            {"TotalCharges": -50},
            "Total charges cannot be negative",
        ),
        (
            {"PhoneService": "No", "MultipleLines": "Yes"},
            "Multiple Lines must be",
        ),
        (
            {"PhoneService": "Yes",
             "MultipleLines": "No phone service"},
            "Multiple Lines must be",
        ),
        (
            {"InternetService": "No", "OnlineSecurity": "Yes"},
            "OnlineSecurity must be",
        ),
        (
            {"InternetService": "DSL",
             "TechSupport": "No internet service"},
            "TechSupport must be",
        ),
    ],
)
def test_invalid_customer(valid_customer, changes, error_message):
    """Invalid customer combinations should be rejected."""
    valid_customer.update(changes)

    with pytest.raises(gr.Error, match=error_message):
        validate_customer(valid_customer)


def test_predict_churn_success(valid_customer):
    """Test prediction with a mocked FastAPI response."""

    mock_response = Mock()
    mock_response.json.return_value = {
        "churn_probability": 0.5474133491516113,
        "threshold": 0.2872343361377716,
        "churn_prediction": 1,
    }

    with patch(
        "app.gradio_ui.requests.post",
        return_value=mock_response,
    ) as mock_post:

        values = [
            valid_customer[feature]
            for feature in FEATURE_NAMES
        ]

        probability, threshold, decision = predict_churn(*values)

        assert probability == pytest.approx(54.74)
        assert threshold == pytest.approx(28.72)
        assert "Churn predicted" in decision

        mock_post.assert_called_once_with(
            API_URL,
            json=valid_customer,
            timeout=15,
        )


def test_predict_no_churn(valid_customer):
    """Test the negative prediction scenario."""

    mock_response = Mock()
    mock_response.json.return_value = {
        "churn_probability": 0.15,
        "threshold": 0.2872343361377716,
        "churn_prediction": 0,
    }

    with patch(
        "app.gradio_ui.requests.post",
        return_value=mock_response,
    ):
        values = [
            valid_customer[feature]
            for feature in FEATURE_NAMES
        ]

        probability, threshold, decision = predict_churn(*values)

        assert probability == pytest.approx(15.0)
        assert threshold == pytest.approx(28.72)
        assert "No churn predicted" in decision


def test_api_connection_error(valid_customer):
    """Test the behavior when FastAPI is unavailable."""

    with patch(
        "app.gradio_ui.requests.post",
        side_effect=requests.exceptions.ConnectionError(
            "API unavailable"
        ),
    ):
        values = [
            valid_customer[feature]
            for feature in FEATURE_NAMES
        ]

        with pytest.raises(
            gr.Error,
            match="Cannot connect to FastAPI",
        ):
            predict_churn(*values)
