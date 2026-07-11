from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class CostAssumptions:
    """Dollar assumptions behind the cost-sensitive threshold choice.

    These are illustrative, not fitted from real business data -- swap in real
    numbers (average customer lifetime value, actual retention-offer cost, and a
    measured save rate from a past campaign) before using this for a real decision.
    """

    retention_cost: float = 50.0
    customer_value_lost: float = 800.0
    retention_success_rate: float = 0.5


@dataclass
class CostResult:
    threshold: float
    total_cost: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def expected_cost_at_threshold(
    y_true, y_proba, threshold: float, assumptions: CostAssumptions = CostAssumptions()
) -> CostResult:
    """Total expected cost of operating at a given threshold.

    Every predicted churner receives a retention offer (cost regardless of whether they'd
    actually have churned). Of the true churners caught, only `retention_success_rate` are
    actually saved -- the rest still churn despite the offer, so their value is still lost.
    Missed churners (false negatives) lose their full value with no offset.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    offer_cost = (tp + fp) * assumptions.retention_cost
    unsaved_caught = tp * (1 - assumptions.retention_success_rate)
    value_lost = (fn + unsaved_caught) * assumptions.customer_value_lost

    total_cost = offer_cost + value_lost
    return CostResult(
        threshold=threshold,
        total_cost=total_cost,
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
    )


def find_cost_optimal_threshold(
    y_true, y_proba, assumptions: CostAssumptions = CostAssumptions()
) -> CostResult:
    """Scan thresholds 0.01-0.99 and return the one minimizing total expected cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    results = [expected_cost_at_threshold(y_true, y_proba, t, assumptions) for t in thresholds]
    return min(results, key=lambda r: r.total_cost)
