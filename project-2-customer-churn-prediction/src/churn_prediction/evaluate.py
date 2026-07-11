import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve

from churn_prediction.config import FIGURES_DIR

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


def compute_classification_metrics(y_true, y_pred, y_proba) -> ClassificationMetrics:
    return ClassificationMetrics(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
    )


def plot_roc_curve(y_true, y_proba, label: str, out_dir: Path = FIGURES_DIR) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "roc_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve to %s", path)
    return path


def plot_pr_curve(y_true, y_proba, label: str, out_dir: Path = FIGURES_DIR) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"{label} (PR-AUC={pr_auc:.3f})")
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pr_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve to %s", path)
    return path


def plot_learning_curve(estimator, X, y, cv: int = 5, out_dir: Path = FIGURES_DIR) -> Path:
    """Training vs validation ROC-AUC as training set size grows -- a direct look at
    bias (both curves low and close together) vs variance (a wide gap between them).
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 6)
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training score")
    ax.plot(train_sizes, val_mean, marker="o", label="Validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning curve")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "learning_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved learning curve to %s", path)
    return path
