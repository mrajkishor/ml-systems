import numpy as np
import pandas as pd

from house_price.stats import (
    compare_groups_ttest,
    confidence_interval_mean,
    correlation_significance,
)


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 300, n)
    price = area * 100 + rng.normal(0, 5000, n)
    central_air = rng.choice(["Y", "N"], size=n, p=[0.8, 0.2])
    price = price + (central_air == "Y") * 20000
    return pd.DataFrame({"GrLivArea": area, "SalePrice": price, "CentralAir": central_air})


def test_correlation_significance_detects_strong_relationship():
    df = make_df()
    result = correlation_significance(df, "GrLivArea", "SalePrice")

    assert result.correlation > 0.9
    assert result.significant


def test_confidence_interval_mean_brackets_true_mean():
    df = make_df()
    ci = confidence_interval_mean(df["SalePrice"])

    assert ci.lower < ci.mean < ci.upper


def test_compare_groups_ttest_detects_difference():
    df = make_df()
    result = compare_groups_ttest(df, "CentralAir", "SalePrice", group_a="Y", group_b="N")

    assert result.mean_a > result.mean_b
    assert result.significant
