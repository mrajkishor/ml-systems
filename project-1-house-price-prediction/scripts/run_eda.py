import logging

from house_price.config import TARGET_COLUMN
from house_price.data.load import get_dataset
from house_price.eda import (
    missing_value_report,
    numeric_summary,
    plot_correlation_heatmap,
    plot_price_vs_feature,
    plot_target_distribution,
)
from house_price.logging_config import setup_logging
from house_price.stats import (
    compare_groups_ttest,
    confidence_interval_mean,
    correlation_significance,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    df = get_dataset()
    logger.info("Dataset shape: %s", df.shape)

    missing = missing_value_report(df)
    print("\nTop missing-value columns:")
    print(missing.head(10))

    summary = numeric_summary(df)
    print("\nNumeric summary (first 5 columns):")
    print(summary.head())

    plot_target_distribution(df, TARGET_COLUMN)
    plot_correlation_heatmap(df, TARGET_COLUMN)

    top_feature = df.select_dtypes(include="number").corr()[TARGET_COLUMN].abs().sort_values(
        ascending=False
    ).index[1]
    plot_price_vs_feature(df, top_feature, TARGET_COLUMN)

    ci = confidence_interval_mean(df[TARGET_COLUMN])
    print(
        f"\n{ci.confidence:.0%} CI for mean {TARGET_COLUMN}: "
        f"${ci.mean:,.0f} [${ci.lower:,.0f}, ${ci.upper:,.0f}]"
    )

    corr_result = correlation_significance(df, top_feature, TARGET_COLUMN)
    print(
        f"\nCorrelation({top_feature}, {TARGET_COLUMN}) = {corr_result.correlation:.3f}, "
        f"p={corr_result.p_value:.2e}, significant={corr_result.significant}"
    )

    if "CentralAir" in df.columns:
        ttest_result = compare_groups_ttest(df, "CentralAir", TARGET_COLUMN)
        print(
            f"\n{TARGET_COLUMN} by CentralAir: "
            f"{ttest_result.group_a}=${ttest_result.mean_a:,.0f} vs "
            f"{ttest_result.group_b}=${ttest_result.mean_b:,.0f}, "
            f"t={ttest_result.t_statistic:.2f}, p={ttest_result.p_value:.2e}, "
            f"significant={ttest_result.significant}"
        )

    print("\nFigures saved to reports/figures/")


if __name__ == "__main__":
    main()
