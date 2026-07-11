import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from churn_prediction.config import RANDOM_STATE

logger = logging.getLogger(__name__)

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(2, 8),
    "learning_rate": uniform(0.01, 0.29),
    "num_leaves": randint(15, 63),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}


def tune_lightgbm(
    X: np.ndarray,
    y: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = RANDOM_STATE,
    class_weight: str | None = "balanced",
) -> RandomizedSearchCV:
    """Randomized hyperparameter search for LightGBM, maximizing ROC-AUC via stratified CV."""
    base_model = LGBMClassifier(random_state=random_state, class_weight=class_weight, verbosity=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base_model,
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
    )
    search.fit(X, y)
    logger.info("Best LightGBM params: %s (CV ROC-AUC=%.4f)", search.best_params_, search.best_score_)
    return search
