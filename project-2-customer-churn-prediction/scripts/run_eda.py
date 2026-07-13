import logging

import pandas as pd

from churn_prediction.config import TARGET_COLUMN
from churn_prediction.data.load import get_dataset
from churn_prediction.eda import (
    class_balance_report,
    plot_categorical_churn_rate,
    plot_churn_rate,
    plot_numeric_by_churn,
)
from churn_prediction.logging_config import setup_logging
from churn_prediction.stats import (
    chi_square_association,
    churn_rate_confidence_interval,
    ttest_numeric_by_target,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    df = get_dataset()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Dataset shape: %s", df.shape)

    print("\nClass balance:")
    print(class_balance_report(df))

    plot_churn_rate(df)
    plot_numeric_by_churn(df, "tenure")
    plot_numeric_by_churn(df, "MonthlyCharges")
    plot_categorical_churn_rate(df, "Contract")
    plot_categorical_churn_rate(df, "InternetService")

    ci = churn_rate_confidence_interval(df)
    print(f"\n{ci.confidence:.0%} CI for churn rate: {ci.proportion:.1%} [{ci.lower:.1%}, {ci.upper:.1%}]")

    tenure_result = ttest_numeric_by_target(df, "tenure")
    print(
        f"\ntenure by Churn: {tenure_result.group_a}={tenure_result.mean_a:.1f} vs "
        f"{tenure_result.group_b}={tenure_result.mean_b:.1f}, "
        f"t={tenure_result.t_statistic:.2f}, p={tenure_result.p_value:.2e}, "
        f"significant={tenure_result.significant}"
    )

    contract_result = chi_square_association(df, "Contract")
    print(
        f"\nContract vs Churn: chi2={contract_result.statistic:.2f}, dof={contract_result.dof}, "
        f"p={contract_result.p_value:.2e}, significant={contract_result.significant}"
    )

    print("\nFigures saved to reports/figures/")


if __name__ == "__main__":
    main()
