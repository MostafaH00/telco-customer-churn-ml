
import pandas as pd


def detect_non_numeric(series: pd.Series) -> pd.Series:
    """
    Return a boolean mask identifying values that
    cannot be converted to numeric.

    Missing or blank values are considered invalid.
    """
    numeric_values = pd.to_numeric(
        series.astype("string").str.strip(),
        errors="coerce"
    )

    return numeric_values.isna()


def summarize_categories(df: pd.DataFrame) -> dict:
    """
    Return unique values and their frequencies
    for categorical columns.
    """
    categorical_cols = df.select_dtypes(
        include=["object", "string", "str", "category"]
    ).columns

    return {
        col: df[col].value_counts(dropna=False)
        for col in categorical_cols
    }



def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Return a boolean mask identifying potential
    outliers using the 1.5 * IQR rule.

    Missing values are not considered outliers.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (
        (series < lower_bound) |
        (series > upper_bound)
    )

