import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline

from house_price.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Human-readable feature names from a fitted feature pipeline (strips the
    ColumnTransformer's `numeric__`/`categorical__` prefixes).
    """
    raw_names = pipeline.named_steps["column_transform"].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def compute_shap_values(model, X: np.ndarray, feature_names: list[str] | None = None):
    """SHAP values for a tree-based model (exact, fast TreeExplainer)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    if feature_names is not None:
        shap_values.feature_names = feature_names
    return shap_values


def plot_shap_summary(shap_values, out_dir: Path = FIGURES_DIR, max_display: int = 20) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shap_summary.png"

    fig = plt.figure()
    shap.summary_plot(shap_values, show=False, max_display=max_display)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary plot to %s", path)
    return path


def plot_shap_importance_bar(shap_values, out_dir: Path = FIGURES_DIR, max_display: int = 20) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shap_feature_importance.png"

    fig = plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=max_display)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP feature importance bar plot to %s", path)
    return path


def plot_shap_waterfall(shap_values, index: int, out_dir: Path = FIGURES_DIR) -> Path:
    """Explain a single prediction: how each feature pushed it above/below the baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shap_waterfall_{index}.png"

    fig = plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP waterfall plot (row %d) to %s", index, path)
    return path


def top_features_by_mean_abs_shap(shap_values, top_n: int = 10) -> list[tuple[str, float]]:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    names = shap_values.feature_names
    return [(names[i], float(mean_abs[i])) for i in order]
