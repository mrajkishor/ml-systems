import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from churn_prediction.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def compute_balanced_class_weight(y) -> dict:
    """Class weights inversely proportional to class frequency (sklearn's 'balanced')."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def apply_smote(X: np.ndarray, y, random_state: int = RANDOM_STATE):
    """Oversample the minority class with SMOTE.

    Call this ONLY on the training split, never on validation/test data -- SMOTE
    synthesizes new minority points from nearest neighbors, so applying it before
    a train/test split would leak information about test-set points into training.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "SMOTE resampled %d -> %d rows (class counts now %s)",
        len(y), len(y_res), pd.Series(y_res).value_counts().to_dict(),
    )
    return X_res, y_res


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> tuple[float, float]:
    """Scan candidate thresholds and return the one maximizing the given metric.

    The default 0.5 threshold is arbitrary for an imbalanced problem -- it implicitly
    assumes false positives and false negatives cost the same, which usually isn't true.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])
