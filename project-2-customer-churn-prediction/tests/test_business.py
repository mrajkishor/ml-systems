import numpy as np

from churn_prediction.business import (
    CostAssumptions,
    expected_cost_at_threshold,
    find_cost_optimal_threshold,
)


def test_expected_cost_at_threshold_perfect_predictions_only_pay_offer_cost():
    # Even with perfect predictions, catching every churner still costs a retention
    # offer per catch -- only a genuinely free intervention could reach zero cost.
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])
    assumptions = CostAssumptions(retention_cost=50.0, retention_success_rate=1.0)

    result = expected_cost_at_threshold(y_true, y_proba, threshold=0.5, assumptions=assumptions)

    assert result.total_cost == 2 * 50.0
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_find_cost_optimal_threshold_beats_arbitrary_threshold():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 80 + [1] * 20)
    y_proba = np.concatenate([rng.uniform(0, 0.6, 80), rng.uniform(0.4, 1.0, 20)])

    optimal = find_cost_optimal_threshold(y_true, y_proba)
    arbitrary = expected_cost_at_threshold(y_true, y_proba, threshold=0.01)

    assert optimal.total_cost <= arbitrary.total_cost
