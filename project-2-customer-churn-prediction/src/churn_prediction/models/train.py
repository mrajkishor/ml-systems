import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn_prediction.config import RANDOM_STATE, TARGET_COLUMN
from churn_prediction.evaluate import ClassificationMetrics, compute_classification_metrics
from churn_prediction.features.engineering import build_feature_pipeline
from churn_prediction.features.imbalance import find_best_threshold

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
    """Stratified split (preserves the 26.5% churn rate in both splits), then fit the
    feature pipeline on train only -- `.transform`, not `.fit_transform`, on test.
    """
    y_all = df[target].map({"Yes": 1, "No": 0})
    X_all = df.drop(columns=[target])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
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


def get_model_candidates(random_state: int = RANDOM_STATE, class_weight: str | None = "balanced") -> dict:
    catboost_kwargs = {"auto_class_weights": "Balanced"} if class_weight == "balanced" else {}
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight=class_weight, random_state=random_state, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            class_weight=class_weight, random_state=random_state, verbosity=-1
        ),
        "CatBoost": CatBoostClassifier(random_state=random_state, verbose=False, **catboost_kwargs),
    }


def cross_validate_model(
    model, X: np.ndarray, y: pd.Series, cv: int = 5, random_state: int = RANDOM_STATE
) -> tuple[float, float]:
    """Mean and std of ROC-AUC across stratified folds (each fold keeps the same
    ~26.5% churn rate, so no fold is accidentally starved of positive examples).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


@dataclass
class ModelResult:
    name: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    test_metrics: ClassificationMetrics
    best_threshold: float
    model: object


def train_and_evaluate_all(
    df: pd.DataFrame, class_weight: str | None = "balanced"
) -> tuple[list[ModelResult], TrainTestData]:
    """Cross-validate and fit every candidate model, evaluating each on the held-out test set.

    Class weighting (not SMOTE) is used here for the main comparison -- see
    `features.imbalance.apply_smote` for the resampling alternative, benchmarked
    separately since it needs to run inside CV to avoid leakage.
    """
    data = prepare_train_test_data(df)

    results = []
    for name, model in get_model_candidates(class_weight=class_weight).items():
        logger.info("Training %s", name)
        cv_mean, cv_std = cross_validate_model(model, data.X_train, data.y_train)

        model.fit(data.X_train, data.y_train)
        y_proba = model.predict_proba(data.X_test)[:, 1]

        threshold, _ = find_best_threshold(data.y_test, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = compute_classification_metrics(data.y_test, y_pred, y_proba)
        results.append(
            ModelResult(
                name=name,
                cv_roc_auc_mean=cv_mean,
                cv_roc_auc_std=cv_std,
                test_metrics=metrics,
                best_threshold=threshold,
                model=model,
            )
        )
        logger.info(
            "%s: CV ROC-AUC=%.4f (+/-%.4f) | threshold=%.2f | "
            "Precision=%.3f Recall=%.3f F1=%.3f ROC-AUC=%.3f PR-AUC=%.3f",
            name, cv_mean, cv_std, threshold,
            metrics.precision, metrics.recall, metrics.f1, metrics.roc_auc, metrics.pr_auc,
        )

    return results, data
