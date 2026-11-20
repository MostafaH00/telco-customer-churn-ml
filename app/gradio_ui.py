
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"

FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def validate_customer(customer):
    """Validate customer information before prediction."""

    # 1. Numerical validation
    if customer["tenure"] < 0:
        raise gr.Error("Tenure cannot be negative.")

    if customer["MonthlyCharges"] < 0:
        raise gr.Error("Monthly charges cannot be negative.")

    if customer["TotalCharges"] < 0:
        raise gr.Error("Total charges cannot be negative.")

    # 2. Phone service consistency
    phone = customer["PhoneService"]
    multiple_lines = customer["MultipleLines"]

    if phone == "No" and multiple_lines != "No phone service":
        raise gr.Error(
            "Multiple Lines must be 'No phone service' "
            "when Phone Service is 'No'."
        )

    if phone == "Yes" and multiple_lines == "No phone service":
        raise gr.Error(
            "Multiple Lines must be 'Yes' or 'No' "
            "when Phone Service is 'Yes'."
        )

    # 3. Internet service consistency
    internet = customer["InternetService"]

    internet_features = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for feature in internet_features:
        value = customer[feature]

        if internet == "No" and value != "No internet service":
            raise gr.Error(
                f"{feature} must be 'No internet service' "
                "when Internet Service is 'No'."
            )

        if internet != "No" and value == "No internet service":
            raise gr.Error(
                f"{feature} must be 'Yes' or 'No' "
                "when Internet Service is available."
            )


def predict_churn(*values):
    """Send customer data to the FastAPI prediction endpoint."""

    customer = dict(zip(FEATURE_NAMES, values))

    customer["SeniorCitizen"] = int(customer["SeniorCitizen"])
    customer["tenure"] = int(customer["tenure"])
    # Validate customer information
    validate_customer(customer)

    try:
        response = requests.post(
            API_URL,
            json=customer,
            timeout=15,
        )

        response.raise_for_status()

        result = response.json()

        probability = result["churn_probability"]
        threshold = result["threshold"]
        prediction = result["churn_prediction"]

        if prediction == 1:
            decision = "### 🔴 Churn predicted"
            explanation = (
                "The customer is predicted to leave the company."
            )
        else:
            decision = "### 🟢 No churn predicted"
            explanation = (
                "The customer is predicted to remain with the company."
            )

        return (
            round(probability * 100, 2),
            round(threshold * 100, 2),
            f"{decision}\n\n{explanation}",
        )

    except requests.exceptions.ConnectionError as exc:
        raise gr.Error(
            "Cannot connect to FastAPI. Check that the API is running."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise gr.Error(f"Prediction failed: {exc}") from exc


def create_interface():
    with gr.Blocks(title="Telco Customer Churn Prediction") as demo:

        gr.Markdown("# Telco Customer Churn Prediction")
        gr.Markdown(
            "Enter customer information to estimate churn risk."
        )

        # 1. Customer profile
        gr.Markdown("## Customer Profile")

        with gr.Row():
            gender = gr.Dropdown(
                ["Male", "Female"], label="Gender", value="Male"
            )
            senior = gr.Radio(
                [("No", 0), ("Yes", 1)],
                label="Senior Citizen",
                value=0,
            )
            partner = gr.Dropdown(
                ["Yes", "No"], label="Partner", value="No"
            )
            dependents = gr.Dropdown(
                ["Yes", "No"], label="Dependents", value="No"
            )

        tenure = gr.Number(
            label="Tenure (months)",
            value=12,
            minimum=0,
            precision=0,
        )

        # 2. Services
        gr.Markdown("## Services")

        with gr.Row():
            phone = gr.Dropdown(
                ["Yes", "No"], label="Phone Service", value="Yes"
            )
            multiple_lines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                label="Multiple Lines",
                value="No",
            )
            internet = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="Fiber optic",
            )

        internet_options = ["Yes", "No", "No internet service"]

        with gr.Row():
            security = gr.Dropdown(
                internet_options, label="Online Security", value="No"
            )
            backup = gr.Dropdown(
                internet_options, label="Online Backup", value="No"
            )
            protection = gr.Dropdown(
                internet_options, label="Device Protection", value="No"
            )

        with gr.Row():
            support = gr.Dropdown(
                internet_options, label="Tech Support", value="No"
            )
            streaming_tv = gr.Dropdown(
                internet_options, label="Streaming TV", value="No"
            )
            streaming_movies = gr.Dropdown(
                internet_options, label="Streaming Movies", value="No"
            )

        # 3. Contract and billing
        gr.Markdown("## Contract & Billing")

        with gr.Row():
            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract",
                value="Month-to-month",
            )
            paperless = gr.Dropdown(
                ["Yes", "No"],
                label="Paperless Billing",
                value="Yes",
            )

        payment = gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
            value="Electronic check",
        )

        with gr.Row():
            monthly = gr.Number(
                label="Monthly Charges ($)", value=70, minimum=0
            )
            total = gr.Number(
                label="Total Charges ($)", value=840, minimum=0
            )

        # 4. Form inputs
        inputs = [
            gender, senior, partner, dependents, tenure,
            phone, multiple_lines, internet, security, backup,
            protection, support, streaming_tv, streaming_movies,
            contract, paperless, payment, monthly, total,
        ]

        # 5. Output
        predict_button = gr.Button(
            "Predict Churn",
            variant="primary",
        )

        gr.Markdown("## Prediction Results")

        with gr.Row():
            probability_output = gr.Number(
                label="Churn Probability (%)",
                precision=2,
                interactive=False,
            )

            threshold_output = gr.Number(
                label="Decision Threshold (%)",
                precision=2,
                interactive=False,
            )

            decision_output = gr.Markdown()

            predict_button.click(
                fn=lambda: (None, None, ""),
                inputs=[],
                outputs=[
                    probability_output,
                    threshold_output,
                    decision_output,
                ],
                queue=False,
            ).then(
                fn=predict_churn,
                inputs=inputs,
                outputs=[
                    probability_output,
                    threshold_output,
                    decision_output,
                ],
            )

    return demo


demo = create_interface()


if __name__ == "__main__":
    demo.launch()
