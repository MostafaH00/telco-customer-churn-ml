
import pandas as pd


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.

    - Convert TotalCharges to numeric.
    - Set missing TotalCharges to 0 for customers
      with tenure = 0.
    - Validate remaining missing values.

    Returns a cleaned copy of the DataFrame.
    """
    df_clean = df.copy()

    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"],
        errors="coerce"
    )

    missing_mask = df_clean["TotalCharges"].isna()

    if not (df_clean.loc[missing_mask, "tenure"] == 0).all():
        raise ValueError(
            "Missing TotalCharges found for tenure > 0."
        )

    df_clean.loc[missing_mask, "TotalCharges"] = 0.0

    if df_clean.isna().any().any():
        raise ValueError("Missing values remain after cleaning.")

    return df_clean


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(
    numeric_features,
    categorical_features,
    binary_features,
    scale_numeric=True
):
    """
    Build a reusable preprocessing pipeline.

    Parameters
    ----------
    numeric_features : list
        Numerical feature names.
    categorical_features : list
        Categorical feature names.
    binary_features : list
        Binary numerical feature names.
    scale_numeric : bool
        Whether to standardize numerical features.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline.
    """

    # Numerical preprocessing
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            ("binary", "passthrough", binary_features)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor
