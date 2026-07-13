import pandas as pd

from house_price.features.outliers import remove_known_outliers, remove_outliers_iqr


def test_remove_known_outliers_drops_large_area_low_price():
    df = pd.DataFrame(
        {
            "GrLivArea": [1500, 4700, 4800],
            "SalePrice": [200000, 160000, 700000],
        }
    )
    result = remove_known_outliers(df)

    assert len(result) == 2
    assert 160000 not in result["SalePrice"].values


def test_remove_known_outliers_noop_without_required_columns():
    df = pd.DataFrame({"OtherCol": [1, 2, 3]})
    result = remove_known_outliers(df)

    assert len(result) == 3


def test_remove_outliers_iqr_drops_extreme_values():
    df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 1000]})
    result = remove_outliers_iqr(df, columns=["value"])

    assert 1000 not in result["value"].values
    assert len(result) == 5
