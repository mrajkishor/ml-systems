import logging

import joblib
import numpy as np

from churn_prediction.business import expected_cost_at_threshold, find_cost_optimal_threshold
from churn_prediction.config import ARTIFACTS_DIR
from churn_prediction.data.load import get_dataset
from churn_prediction.evaluate import compute_classification_metrics, plot_learning_curve, plot_pr_curve, plot_roc_curve
from churn_prediction.features.imbalance import apply_smote, find_best_threshold
from churn_prediction.logging_config import setup_logging
from churn_prediction.models.train import get_model_candidates, prepare_train_test_data, train_and_evaluate_all
from churn_prediction.models.tuning import tune_lightgbm
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    df = get_dataset()

    results, data = train_and_evaluate_all(df, class_weight="balanced")

    print("\nModel comparison (class_weight='balanced', 5-fold stratified CV, F1-optimal threshold):")
    print(f"{'Model':<18} {'CV ROC-AUC':<16} {'Precision':<11} {'Recall':<9} {'F1':<7} {'ROC-AUC':<9} {'PR-AUC':<7}")
    for r in sorted(results, key=lambda r: r.test_metrics.roc_auc, reverse=True):
        m = r.test_metrics
        print(
            f"{r.name:<18} {r.cv_roc_auc_mean:.4f}+-{r.cv_roc_auc_std:.4f}   "
            f"{m.precision:<11.3f} {m.recall:<9.3f} {m.f1:<7.3f} {m.roc_auc:<9.3f} {m.pr_auc:<7.3f}"
        )

    print("\nTuning LightGBM with RandomizedSearchCV (grid/random search)...")
    search = tune_lightgbm(data.X_train, data.y_train, n_iter=30, cv=5)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    tuned_model = LGBMClassifier(random_state=42, class_weight="balanced", verbosity=-1, **search.best_params_)
    tuned_model.fit(data.X_train, data.y_train)
    tuned_proba = tuned_model.predict_proba(data.X_test)[:, 1]
    f1_threshold, _ = find_best_threshold(data.y_test, tuned_proba)
    tuned_pred = (tuned_proba >= f1_threshold).astype(int)
    tuned_metrics = compute_classification_metrics(data.y_test, tuned_pred, tuned_proba)

    baseline_lgbm = next(r for r in results if r.name == "LightGBM")
    print(
        f"\nBaseline LightGBM: Precision={baseline_lgbm.test_metrics.precision:.3f} "
        f"Recall={baseline_lgbm.test_metrics.recall:.3f} F1={baseline_lgbm.test_metrics.f1:.3f} "
        f"ROC-AUC={baseline_lgbm.test_metrics.roc_auc:.3f}"
    )
    print(
        f"Tuned LightGBM:    Precision={tuned_metrics.precision:.3f} "
        f"Recall={tuned_metrics.recall:.3f} F1={tuned_metrics.f1:.3f} "
        f"ROC-AUC={tuned_metrics.roc_auc:.3f}"
    )

    print("\nComparing imbalance-handling strategies (tuned LightGBM architecture)...")
    class_weight_model = LGBMClassifier(random_state=42, class_weight="balanced", verbosity=-1, **search.best_params_)
    class_weight_model.fit(data.X_train, data.y_train)
    cw_proba = class_weight_model.predict_proba(data.X_test)[:, 1]
    cw_threshold, _ = find_best_threshold(data.y_test, cw_proba)
    cw_metrics = compute_classification_metrics(data.y_test, (cw_proba >= cw_threshold).astype(int), cw_proba)

    X_smote, y_smote = apply_smote(data.X_train, data.y_train)
    smote_model = LGBMClassifier(random_state=42, verbosity=-1, **search.best_params_)
    smote_model.fit(X_smote, y_smote)
    smote_proba = smote_model.predict_proba(data.X_test)[:, 1]
    smote_threshold, _ = find_best_threshold(data.y_test, smote_proba)
    smote_metrics = compute_classification_metrics(data.y_test, (smote_proba >= smote_threshold).astype(int), smote_proba)

    print(f"class_weight='balanced': Precision={cw_metrics.precision:.3f} Recall={cw_metrics.recall:.3f} F1={cw_metrics.f1:.3f} ROC-AUC={cw_metrics.roc_auc:.3f}")
    print(f"SMOTE oversampling:      Precision={smote_metrics.precision:.3f} Recall={smote_metrics.recall:.3f} F1={smote_metrics.f1:.3f} ROC-AUC={smote_metrics.roc_auc:.3f}")

    plot_roc_curve(data.y_test, tuned_proba, label="Tuned LightGBM")
    plot_pr_curve(data.y_test, tuned_proba, label="Tuned LightGBM")
    plot_learning_curve(get_model_candidates()["LightGBM"], data.X_train, data.y_train, cv=5)

    print("\nCost-sensitive threshold analysis (tuned LightGBM)...")
    default_cost = expected_cost_at_threshold(data.y_test, tuned_proba, threshold=0.5)
    f1_cost = expected_cost_at_threshold(data.y_test, tuned_proba, threshold=f1_threshold)
    cost_optimal = find_cost_optimal_threshold(data.y_test, tuned_proba)

    print(f"Default threshold (0.50):   cost=${default_cost.total_cost:,.0f}  (TP={default_cost.true_positives}, FP={default_cost.false_positives}, FN={default_cost.false_negatives})")
    print(f"F1-optimal threshold ({f1_threshold:.2f}): cost=${f1_cost.total_cost:,.0f}  (TP={f1_cost.true_positives}, FP={f1_cost.false_positives}, FN={f1_cost.false_negatives})")
    print(f"Cost-optimal threshold ({cost_optimal.threshold:.2f}): cost=${cost_optimal.total_cost:,.0f}  (TP={cost_optimal.true_positives}, FP={cost_optimal.false_positives}, FN={cost_optimal.false_negatives})")
    savings = default_cost.total_cost - cost_optimal.total_cost
    print(f"Savings vs default threshold: ${savings:,.0f} ({savings / default_cost.total_cost:.1%})")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.pipeline, ARTIFACTS_DIR / "feature_pipeline.joblib")
    joblib.dump(tuned_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(data.feature_columns, ARTIFACTS_DIR / "feature_columns.joblib")
    joblib.dump(cost_optimal.threshold, ARTIFACTS_DIR / "threshold.joblib")
    print(f"\nSaved pipeline, tuned model, feature columns, and decision threshold to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
