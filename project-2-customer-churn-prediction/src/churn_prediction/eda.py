import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churn_prediction.config import FIGURES_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)


def class_balance_report(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    counts = df[target].value_counts()
    percent = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "percent": percent})


def plot_churn_rate(
    df: pd.DataFrame, target: str = TARGET_COLUMN, out_dir: Path = FIGURES_DIR
) -> Path:
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Class balance: {target}")
    ax.set_ylabel("Count")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "churn_rate.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved churn rate plot to %s", path)
    return path


def plot_numeric_by_churn(
    df: pd.DataFrame, column: str, target: str = TARGET_COLUMN, out_dir: Path = FIGURES_DIR
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [g[column].dropna().values for _, g in df.groupby(target)]
    labels = [str(name) for name, _ in df.groupby(target)]
    ax.boxplot(groups, tick_labels=labels)
    ax.set_title(f"{column} by {target}")
    ax.set_ylabel(column)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{column}_by_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s by %s boxplot to %s", column, target, path)
    return path


def plot_categorical_churn_rate(
    df: pd.DataFrame, column: str, target: str = TARGET_COLUMN, out_dir: Path = FIGURES_DIR
) -> Path:
    positive_label = sorted(df[target].unique())[-1]
    rate = df.groupby(column)[target].apply(lambda s: (s == positive_label).mean() * 100)
    rate = rate.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rate.index.astype(str), rate.values)
    ax.set_title(f"Churn rate by {column}")
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=45)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"churn_rate_by_{column}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved churn rate by %s plot to %s", column, path)
    return path
