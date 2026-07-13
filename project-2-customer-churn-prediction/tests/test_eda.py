import numpy as np
import pandas as pd

from churn_prediction.eda import (
    class_balance_report,
    plot_categorical_churn_rate,
    plot_churn_rate,
    plot_numeric_by_churn,
)


def make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    churn = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    tenure = rng.normal(30, 10, n) - (churn == "Yes") * 10
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    return pd.DataFrame({"tenure": tenure, "Contract": contract, "Churn": churn})


def test_class_balance_report_sums_to_total():
    df = make_df()
    report = class_balance_report(df)

    assert report["count"].sum() == len(df)
    assert abs(report["percent"].sum() - 100) < 0.1


def test_plot_churn_rate_writes_file(tmp_path):
    df = make_df()
    path = plot_churn_rate(df, out_dir=tmp_path)

    assert path.exists()


def test_plot_numeric_by_churn_writes_file(tmp_path):
    df = make_df()
    path = plot_numeric_by_churn(df, "tenure", out_dir=tmp_path)

    assert path.exists()


def test_plot_categorical_churn_rate_writes_file(tmp_path):
    df = make_df()
    path = plot_categorical_churn_rate(df, "Contract", out_dir=tmp_path)

    assert path.exists()
