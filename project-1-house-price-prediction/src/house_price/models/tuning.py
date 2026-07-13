import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from house_price.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def _objective(trial: optuna.Trial, X: np.ndarray, y_log: pd.Series, cv: int, random_state: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    model = XGBRegressor(random_state=random_state, n_jobs=-1, **params)
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = -cross_val_score(model, X, y_log, cv=kfold, scoring="neg_root_mean_squared_error")
    return float(scores.mean())


def tune_xgboost(
    X: np.ndarray,
    y_log: pd.Series,
    n_trials: int = 30,
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> optuna.Study:
    """Bayesian hyperparameter search (TPE) for XGBoost, minimizing CV RMSE on the log target."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(lambda trial: _objective(trial, X, y_log, cv, random_state), n_trials=n_trials)
    logger.info("Best XGBoost params: %s (CV RMSE=%.4f)", study.best_params, study.best_value)
    return study
