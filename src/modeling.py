
from time import perf_counter
from pathlib import Path
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocessing import build_preprocessor


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display



def create_model_pipeline(
    estimator,
    numeric_features,
    categorical_features,
    binary_features,
    scale_numeric=False,
    use_ros=False
):
    """
    Build a machine learning pipeline with preprocessing,
    optional random oversampling, and classification.
    """

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        scale_numeric=scale_numeric
    )

    steps = [
        ("preprocessor", preprocessor)
    ]

    if use_ros:
        steps.append(
            ("sampler", RandomOverSampler(random_state=42))
        )

    steps.append(
        ("classifier", estimator)
    )

    return ImbPipeline(steps)



def evaluate_model_cv(model, X, y, cv, model_name, fold_scale_pos_weight=False):

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        estimator = clone(model)

        # Compute class weight from the current training fold only
        if fold_scale_pos_weight:
            n_negative = (y_tr == 0).sum()
            n_positive = (y_tr == 1).sum()

            fold_weight = n_negative / n_positive

            estimator.set_params(
                model__scale_pos_weight=fold_weight
            )

        start = perf_counter()
        estimator.fit(X_tr, y_tr)
        fit_time = perf_counter() - start

        start = perf_counter()
        y_score = estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        prediction_time = perf_counter() - start

        results.append({
            "model": model_name,
            "fold": fold,
            "average_precision": average_precision_score(y_val, y_score),
            "roc_auc": roc_auc_score(y_val, y_score),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
            "fit_time": fit_time,
            "prediction_time": prediction_time,
        })

    return pd.DataFrame(results)



from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display


def compare_models_cv(
    results_df,
    title="Model Comparison",
    save_path=None
):
    """
    Compare cross-validation results across multiple models.

    Displays:
    - Summary table (mean ± std).
    - Grouped bar chart with error bars for up to 5 models.
    - Heatmap for more than 5 models.

    Models are ranked by mean Average Precision.

    Parameters
    ----------
    results_df : pd.DataFrame
        Cross-validation results containing model names,
        evaluation metrics, and fold-level scores.

    title : str, default="Model Comparison"
        Title of the comparison figure.

    save_path : str or Path, optional
        Destination path for exporting the figure.
        Parent directories are created automatically.

    Returns
    -------
    means : pd.DataFrame
        Mean cross-validation metrics for each model.

    stds : pd.DataFrame
        Standard deviation of cross-validation metrics.
    """

    metrics = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "fit_time",
        "prediction_time",
    ]

    score_metrics = metrics[:6]

    metric_labels = [
        "AP",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1",
        "Balanced Acc.",
    ]

    # 1. Aggregate cross-validation results
    grouped = results_df.groupby("model")[metrics]

    means = (
        grouped.mean()
        .sort_values("average_precision", ascending=False)
    )

    stds = grouped.std().loc[means.index]

    # 2. Create comparison table (mean ± std)
    comparison = pd.DataFrame(index=means.index)

    for metric in metrics:
        comparison[metric] = [
            f"{mean:.4f} ± {std:.4f}"
            for mean, std in zip(
                means[metric],
                stds[metric]
            )
        ]

    print(f"\n{title}")
    display(comparison)

    # 3. Generate visualization
    n_models = len(means)

    if n_models <= 5:

        # Grouped bar chart
        x = np.arange(len(score_metrics))
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(13, 6))

        for i, model in enumerate(means.index):

            offset = (i - (n_models - 1) / 2) * width

            ax.bar(
                x + offset,
                means.loc[model, score_metrics],
                width,
                yerr=stds.loc[model, score_metrics],
                capsize=3,
                label=model,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Cross-validation score")
        ax.set_title(title)

        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    else:

        # Heatmap for larger comparisons
        fig, ax = plt.subplots(
            figsize=(11, max(5, n_models * 0.55))
        )

        values = means[score_metrics].to_numpy()

        im = ax.imshow(
            values,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(score_metrics)))
        ax.set_xticklabels(metric_labels)

        ax.set_yticks(range(n_models))
        ax.set_yticklabels(means.index)

        # Annotate heatmap cells
        for i in range(n_models):
            for j in range(len(score_metrics)):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if values[i, j] < 0.5
                        else "black"
                    ),
                )

        fig.colorbar(
            im,
            ax=ax,
            label="Mean CV score"
        )

        ax.set_title(title)

    # 4. Finalize layout
    fig.tight_layout()

    # 5. Export figure
    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

        print(f"Figure saved to: {save_path}")

    # 6. Display figure
    plt.show()
    plt.close(fig)

    return means, stds

