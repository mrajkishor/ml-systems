from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from house_price.config import TARGET_COLUMN


@dataclass
class CorrelationTestResult:
    feature: str
    correlation: float
    p_value: float
    significant: bool


@dataclass
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    confidence: float


@dataclass
class TTestResult:
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    t_statistic: float
    p_value: float
    significant: bool


def correlation_significance(
    df: pd.DataFrame, feature: str, target: str = TARGET_COLUMN, alpha: float = 0.05
) -> CorrelationTestResult:
    """Pearson correlation between a numeric feature and the target, with a significance test."""
    paired = df[[feature, target]].dropna()
    corr, p_value = stats.pearsonr(paired[feature], paired[target])
    return CorrelationTestResult(
        feature=feature, correlation=corr, p_value=p_value, significant=p_value < alpha
    )


def confidence_interval_mean(
    series: pd.Series, confidence: float = 0.95
) -> ConfidenceInterval:
    """Confidence interval for the mean of a numeric series, via the t-distribution."""
    clean = series.dropna()
    mean = clean.mean()
    sem = stats.sem(clean)
    lower, upper = stats.t.interval(confidence, len(clean) - 1, loc=mean, scale=sem)
    return ConfidenceInterval(mean=mean, lower=lower, upper=upper, confidence=confidence)


def compare_groups_ttest(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = TARGET_COLUMN,
    group_a=None,
    group_b=None,
    alpha: float = 0.05,
) -> TTestResult:
    """Welch's t-test comparing `value_col` between two categories of `group_col`.

    If group_a/group_b aren't given, uses the two most frequent categories.
    """
    if group_a is None:
        counts = df[group_col].value_counts()
        group_a, group_b = counts.index[0], counts.index[1]

    values_a = df.loc[df[group_col] == group_a, value_col].dropna()
    values_b = df.loc[df[group_col] == group_b, value_col].dropna()

    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
    return TTestResult(
        group_a=str(group_a),
        group_b=str(group_b),
        mean_a=values_a.mean(),
        mean_b=values_b.mean(),
        t_statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
    )
