import numpy as np
import pandas as pd

from house_price.eda import (
    missing_value_report,
    numeric_summary,
    plot_correlation_heatmap,
    plot_price_vs_feature,
    plot_target_distribution,
)


def make_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 300, n)
    price = area * 100 + rng.normal(0, 5000, n)
    df = pd.DataFrame({"GrLivArea": area, "SalePrice": price})
    df.loc[: n // 10, "GrLivArea"] = np.nan
    return df


def test_missing_value_report_only_lists_missing_columns():
    df = make_df()
    report = missing_value_report(df)

    assert "GrLivArea" in report.index
    assert "SalePrice" not in report.index
    assert (report["missing_percent"] > 0).all()


def test_numeric_summary_has_expected_stats():
    df = make_df()
    summary = numeric_summary(df)

    assert "mean" in summary.columns
    assert "SalePrice" in summary.index


def test_plot_target_distribution_writes_file(tmp_path):
    df = make_df()
    path = plot_target_distribution(df, target="SalePrice", out_dir=tmp_path)

    assert path.exists()


def test_plot_correlation_heatmap_writes_file(tmp_path):
    df = make_df()
    path = plot_correlation_heatmap(df, target="SalePrice", top_n=1, out_dir=tmp_path)

    assert path.exists()


def test_plot_price_vs_feature_writes_html(tmp_path):
    df = make_df()
    path = plot_price_vs_feature(df, feature="GrLivArea", target="SalePrice", out_dir=tmp_path)

    assert path.exists()
    assert path.suffix == ".html"
