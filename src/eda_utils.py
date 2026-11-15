
"""Reusable visualization functions for customer churn EDA."""
import pandas as pd

import numpy as np
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sns


# Consistent colors across all charts
CHURN_COLORS = {
    "No": "#2A9D8F",
    "Yes": "#E76F51"
}

CHURN_ORDER = ["No", "Yes"]


def setup_plot_style():
    """Configure visualization style."""

    sns.set_theme(style="whitegrid", context="notebook")

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "grid.alpha": 0.25
    })


def plot_churn_distribution(df):
    """Plot customer churn counts and proportions."""

    counts = (
        df["Churn"]
        .value_counts()
        .reindex(CHURN_ORDER, fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    bars = axes[0].bar(
        counts.index,
        counts.values,
        color=[CHURN_COLORS[c] for c in CHURN_ORDER],
        width=0.55
    )

    axes[0].bar_label(bars, padding=5)
    axes[0].set_title("Customer Churn Distribution")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_ylim(0, counts.max() * 1.15)

    # Donut chart
    axes[1].pie(
        counts.values,
        labels=CHURN_ORDER,
        colors=[CHURN_COLORS[c] for c in CHURN_ORDER],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.8,
        wedgeprops={
            "width": 0.4,
            "edgecolor": "white"
        }
    )

    axes[1].set_title("Churn Proportion")

    fig.tight_layout()

    return fig


def plot_numerical_features(df, features, mode="distribution"):
    """Visualize numerical features using different plot types.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    features : list
        Numerical features to visualize.
    mode : str
        One of: distribution, by_churn, box, violin.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    valid_modes = ["distribution", "by_churn", "box", "violin"]

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode. Choose from {valid_modes}")

    fig, axes = plt.subplots(
        1,
        len(features),
        figsize=(5.5 * len(features), 5),
        squeeze=False
    )

    for ax, feature in zip(axes.flat, features):

        if mode == "distribution":
            sns.histplot(
                data=df,
                x=feature,
                bins=30,
                kde=True,
                color="#457B9D",
                ax=ax
            )

        elif mode == "by_churn":
            sns.kdeplot(
                data=df,
                x=feature,
                hue="Churn",
                hue_order=CHURN_ORDER,
                palette=CHURN_COLORS,
                common_norm=False,
                ax=ax,
                cut=0
            )

        elif mode == "box":
            sns.boxplot(
                data=df,
                x="Churn",
                y=feature,
                order=CHURN_ORDER,
                hue="Churn",
                hue_order=CHURN_ORDER,
                palette=CHURN_COLORS,
                dodge=False,
                ax=ax
            )

        elif mode == "violin":
            sns.violinplot(
                data=df,
                x="Churn",
                y=feature,
                order=CHURN_ORDER,
                hue="Churn",
                hue_order=CHURN_ORDER,
                palette=CHURN_COLORS,
                dodge=False,
                inner="quart",
                ax=ax,
                cut=0
            )

        # Remove redundant legend
        if mode in ["box", "violin"]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        ax.set_title(f"{feature} — {mode.replace('_', ' ').title()}")

    fig.tight_layout()

    return fig



def plot_categorical_features(
    df, features, mode="count", ncols=2
):
    """Plot categorical distributions or churn rates."""

    valid_modes = ["count", "rate"]

    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")

    nrows = (len(features) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4.5 * nrows),
        squeeze=False
    )

    for ax, feature in zip(axes.flat, features):

        # Customer counts by churn status
        counts = pd.crosstab(
            df[feature],
            df["Churn"]
        ).reindex(
            columns=CHURN_ORDER,
            fill_value=0
        )

        # Calculate churn rates
        totals = counts.sum(axis=1)
        rates = counts["Yes"] / totals * 100

        # Sort categories by churn rate
        order = rates.sort_values(ascending=False).index

        counts = counts.loc[order]
        totals = totals.loc[order]
        rates = rates.loc[order]

        labels = order.astype(str)

        if mode == "count":

            ax.barh(
                labels,
                counts["No"],
                color=CHURN_COLORS["No"],
                label="No"
            )

            ax.barh(
                labels,
                counts["Yes"],
                left=counts["No"],
                color=CHURN_COLORS["Yes"],
                label="Yes"
            )

            ax.set_xlabel("Number of Customers")
            ax.set_title(f"{feature} — Customer Distribution")
            ax.legend(title="Churn")

        elif mode == "rate":

            bars = ax.barh(
                labels,
                rates,
                color=CHURN_COLORS["Yes"],
                alpha=0.85
            )

            # Annotate churn rates and sample sizes
            for bar, rate, n in zip(
                bars, rates, totals
            ):
                ax.text(
                    rate + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{rate:.1f}% (n={n})",
                    va="center",
                    fontsize=9
                )

            ax.set_xlabel("Churn Rate (%)")
            ax.set_xlim(0, max(rates.max() * 1.35, 10))
            ax.set_title(f"{feature} — Churn Rate")

        ax.invert_yaxis()
        ax.set_ylabel("")

    # Hide unused subplots
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)

    fig.tight_layout()

    return fig



def compute_cramers_v(df, feature, target="Churn"):
    """Compute bias-corrected Cramér's V."""

    # Contingency table
    contingency = pd.crosstab(df[feature], df[target])

    n = contingency.to_numpy().sum()
    r, k = contingency.shape

    if n <= 1 or min(r, k) < 2:
        return np.nan

    # Chi-square statistic
    chi2, _, _, _ = chi2_contingency(
        contingency,
        correction=False
    )

    # Bias correction
    phi2 = chi2 / n
    phi2_corr = max(
        0,
        phi2 - ((r - 1) * (k - 1)) / (n - 1)
    )

    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(r_corr - 1, k_corr - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(phi2_corr / denominator)


def plot_cramers_v(scores):
    """Plot categorical features ranked by Cramér's V."""

    scores = scores.dropna().sort_values()

    fig, ax = plt.subplots(
        figsize=(10, max(5, len(scores) * 0.45))
    )

    bars = ax.barh(
        scores.index,
        scores.values,
        color="#457B9D",
        alpha=0.9
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=4,
        fontsize=9
    )

    ax.set_xlim(0, max(0.05, scores.max() * 1.2))
    ax.set_xlabel("Bias-Corrected Cramér's V")
    ax.set_ylabel("")
    ax.set_title("Categorical Feature Associations with Churn")

    fig.tight_layout()

    return fig



def plot_churn_heatmap(
    df,
    row_feature,
    col_feature,
    target="Churn",
    min_count=30
):
    """Plot churn rates across two categorical features."""

    # Compute churn rates and group sizes
    summary = (
        df.assign(
            _churn=df[target].eq("Yes").astype(int)
        )
        .groupby(
            [row_feature, col_feature],
            observed=True
        )["_churn"]
        .agg(["mean", "size"])
    )

    # Create pivot tables
    rates = summary["mean"].unstack() * 100
    counts = summary["size"].unstack()

    # Build annotations: churn rate and sample size
    annotations = pd.DataFrame(
        "",
        index=rates.index,
        columns=rates.columns
    )

    for i in range(len(rates.index)):
        for j in range(len(rates.columns)):

            rate = rates.iloc[i, j]
            n = counts.iloc[i, j]

            if pd.notna(rate):
                flag = "*" if n < min_count else ""

                annotations.iloc[i, j] = (
                    f"{rate:.1f}%{flag}\n(n={int(n)})"
                )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.heatmap(
        rates,
        annot=annotations,
        fmt="",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        mask=rates.isna(),
        cbar_kws={"label": "Churn Rate (%)"},
        ax=ax
    )

    ax.set_title(
        f"Churn Rate — {row_feature} vs {col_feature}"
    )

    ax.set_xlabel(col_feature)
    ax.set_ylabel(row_feature)

    fig.text(
        0.5, 0.01,
        f"* Groups with fewer than {min_count} customers",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    return fig
