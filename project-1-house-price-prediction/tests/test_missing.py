import numpy as np
import pandas as pd

from house_price.features.missing import impute_domain_missing_values


def test_none_means_absent_columns_filled_with_none():
    df = pd.DataFrame({"PoolQC": [np.nan, "Gd"], "SalePrice": [100, 200]})
    result = impute_domain_missing_values(df)

    assert list(result["PoolQC"]) == ["None", "Gd"]


def test_zero_means_absent_columns_filled_with_zero():
    df = pd.DataFrame({"GarageArea": [np.nan, 400.0], "SalePrice": [100, 200]})
    result = impute_domain_missing_values(df)

    assert list(result["GarageArea"]) == [0.0, 400.0]


def test_lot_frontage_imputed_by_neighborhood_median():
    df = pd.DataFrame(
        {
            "LotFrontage": [60.0, np.nan, 80.0, 100.0],
            "Neighborhood": ["A", "A", "A", "B"],
        }
    )
    result = impute_domain_missing_values(df)

    assert result.loc[1, "LotFrontage"] == 70.0
    assert not result["LotFrontage"].isna().any()


def test_fallback_fills_remaining_missing_values():
    df = pd.DataFrame({"SomeNumeric": [1.0, np.nan, 3.0], "SomeCategory": ["x", np.nan, "x"]})
    result = impute_domain_missing_values(df)

    assert not result.isna().any().any()
    assert result.loc[1, "SomeCategory"] == "x"
