from dataclasses import dataclass

import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from churn_prediction.config import TARGET_COLUMN


@dataclass
class ProportionCI:
    proportion: float
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


@dataclass
class ChiSquareResult:
    statistic: float
    p_value: float
    dof: int
    significant: bool


def churn_rate_confidence_interval(
    df: pd.DataFrame, target: str = TARGET_COLUMN, confidence: float = 0.95
) -> ProportionCI:
    """Wilson-score confidence interval for the churn rate (proportion of positive class)."""
    positive_label = sorted(df[target].unique())[-1]
    successes = int((df[target] == positive_label).sum())
    n = len(df)
    alpha = 1 - confidence
    lower, upper = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return ProportionCI(proportion=successes / n, lower=lower, upper=upper, confidence=confidence)


def ttest_numeric_by_target(
    df: pd.DataFrame, column: str, target: str = TARGET_COLUMN, alpha: float = 0.05
) -> TTestResult:
    """Welch's t-test comparing a numeric column between the two target classes."""
    labels = sorted(df[target].unique())
    group_a, group_b = labels[0], labels[1]
    values_a = df.loc[df[target] == group_a, column].dropna()
    values_b = df.loc[df[target] == group_b, column].dropna()

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


def chi_square_association(
    df: pd.DataFrame, column: str, target: str = TARGET_COLUMN, alpha: float = 0.05
) -> ChiSquareResult:
    """Chi-square test of independence between a categorical column and the target."""
    contingency = pd.crosstab(df[column], df[target])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    return ChiSquareResult(statistic=chi2, p_value=p_value, dof=dof, significant=p_value < alpha)
