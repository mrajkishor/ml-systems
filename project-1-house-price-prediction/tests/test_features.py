import pandas as pd

from house_price.features.engineering import build_features


def test_build_features_returns_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)
