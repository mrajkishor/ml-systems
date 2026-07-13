import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from house_price.config import FIGURES_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, sorted descending, missing-only."""
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    percent = (counts / len(df)) * 100
    return pd.DataFrame({"missing_count": counts, "missing_percent": percent.round(2)})


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").describe().T


def plot_target_distribution(
    df: pd.DataFrame, target: str = TARGET_COLUMN, out_dir: Path = FIGURES_DIR
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    df[target].hist(bins=50, ax=ax)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target}_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved target distribution plot to %s", path)
    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    top_n: int = 15,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    numeric_df = df.select_dtypes(include="number")
    correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
    top_features = correlations.head(top_n + 1).index  # +1 to include target itself
    corr_matrix = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90)
    ax.set_yticklabels(top_features)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Top {top_n} features correlated with {target}")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", path)
    return path


def plot_price_vs_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET_COLUMN,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    fig = px.scatter(
        df,
        x=feature,
        y=target,
        trendline="ols",
        title=f"{target} vs {feature}",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target}_vs_{feature}.html"
    fig.write_html(str(path))
    logger.info("Saved interactive scatter plot to %s", path)
    return path
