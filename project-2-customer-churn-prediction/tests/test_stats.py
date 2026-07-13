import numpy as np
import pandas as pd

from churn_prediction.stats import (
    chi_square_association,
    churn_rate_confidence_interval,
    ttest_numeric_by_target,
)


def make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    churn = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    tenure = rng.normal(30, 8, n) - (churn == "Yes") * 15
    contract = np.where(
        churn == "Yes",
        rng.choice(["Month-to-month", "One year"], size=n, p=[0.9, 0.1]),
        rng.choice(["Month-to-month", "One year"], size=n, p=[0.3, 0.7]),
    )
    return pd.DataFrame({"tenure": tenure, "Contract": contract, "Churn": churn})


def test_churn_rate_confidence_interval_brackets_true_rate():
    df = make_df()
    ci = churn_rate_confidence_interval(df)

    assert ci.lower < ci.proportion < ci.upper
    assert 0 <= ci.lower <= ci.upper <= 1


def test_ttest_numeric_by_target_detects_difference():
    df = make_df()
    result = ttest_numeric_by_target(df, "tenure")

    assert result.mean_b > result.mean_a or result.mean_a > result.mean_b
    assert result.significant


def test_chi_square_association_detects_dependence():
    df = make_df()
    result = chi_square_association(df, "Contract")

    assert result.significant
