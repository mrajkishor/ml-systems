import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

from house_price.config import RANDOM_STATE, TARGET_COLUMN
from house_price.evaluate import RegressionMetrics, compute_regression_metrics
from house_price.features.engineering import build_feature_pipeline
from house_price.features.outliers import remove_known_outliers

logger = logging.getLogger(__name__)


@dataclass
class TrainTestData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    pipeline: object
    feature_columns: list[str]


def prepare_train_test_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> TrainTestData:
    """Split raw data into train/test, then fit the feature pipeline on train only.

    Fitting on train only (and calling `.transform`, not `.fit_transform`, on test) avoids
    leaking test-set statistics into imputation/scaling/encoding.
    """
    df = remove_known_outliers(df, target=target)
    X = df.drop(columns=[target])
    y = df[target]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_feature_pipeline(X_train_raw)
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    return TrainTestData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        pipeline=pipeline,
        feature_columns=list(X_train_raw.columns),
    )


def get_model_candidates(random_state: int = RANDOM_STATE) -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def cross_validate_model(
    model, X: np.ndarray, y_log: pd.Series, cv: int = 5, random_state: int = RANDOM_STATE
) -> tuple[float, float]:
    """Mean and std of RMSE (on the log-target) across k folds.

    The std across folds is a direct read on variance: a model whose fold scores swing
    widely is overfitting to whichever data it happened to see (high variance), while a
    model with a high mean but low std is consistently underfitting (high bias).
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = -cross_val_score(model, X, y_log, cv=kfold, scoring="neg_root_mean_squared_error")
    return float(scores.mean()), float(scores.std())


@dataclass
class ModelResult:
    name: str
    cv_rmse_log_mean: float
    cv_rmse_log_std: float
    test_metrics: RegressionMetrics
    model: object


def train_and_evaluate_all(df: pd.DataFrame) -> tuple[list[ModelResult], TrainTestData]:
    """Cross-validate and fit every candidate model, evaluating each on the held-out test set.

    Targets are log1p-transformed for training (SalePrice is right-skewed) and predictions
    are inverse-transformed (expm1) before computing metrics, so MAE/RMSE come out in dollars.
    """
    data = prepare_train_test_data(df)
    y_train_log = np.log1p(data.y_train)

    results = []
    for name, model in get_model_candidates().items():
        logger.info("Training %s", name)
        cv_mean, cv_std = cross_validate_model(model, data.X_train, y_train_log)

        model.fit(data.X_train, y_train_log)
        preds = np.expm1(model.predict(data.X_test))

        metrics = compute_regression_metrics(data.y_test, preds)
        results.append(
            ModelResult(
                name=name,
                cv_rmse_log_mean=cv_mean,
                cv_rmse_log_std=cv_std,
                test_metrics=metrics,
                model=model,
            )
        )
        logger.info(
            "%s: CV RMSE(log)=%.4f (+/-%.4f) | Test MAE=$%.0f RMSE=$%.0f R2=%.3f",
            name,
            cv_mean,
            cv_std,
            metrics.mae,
            metrics.rmse,
            metrics.r2,
        )

    return results, data
