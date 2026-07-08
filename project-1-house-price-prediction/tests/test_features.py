import numpy as np
import pandas as pd

from house_price.features.engineering import build_features, encode_ordinal_quality


def make_df(n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "GrLivArea": rng.normal(1500, 200, n),
            "LotFrontage": rng.normal(70, 10, n),
            "Neighborhood": rng.choice(["A", "B", "C"], size=n),
            "ExterQual": rng.choice(["TA", "Gd", "Ex", "Fa"], size=n),
            "SalePrice": rng.normal(200000, 30000, n),
        }
    )
    df.loc[0, "LotFrontage"] = np.nan
    return df


def test_encode_ordinal_quality_maps_known_scale():
    df = pd.DataFrame({"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex", None]})
    result = encode_ordinal_quality(df)

    assert list(result["ExterQual"]) == [1, 2, 3, 4, 5, 0]


def test_build_features_returns_transformed_array_and_target():
    df = make_df()
    X, y, pipeline = build_features(df)

    assert X.shape[0] == len(df)
    assert y is not None and len(y) == len(df)
    assert not np.isnan(X).any()


def test_build_features_pipeline_is_reusable_on_new_data():
    df = make_df()
    X_train, _, pipeline = build_features(df)

    new_df = make_df(n=5, seed=1).drop(columns=["SalePrice"])
    X_new = pipeline.transform(new_df)

    assert X_new.shape[1] == X_train.shape[1]


def test_build_features_drops_known_outliers():
    df = make_df()
    df.loc[0, "GrLivArea"] = 4700
    df.loc[0, "SalePrice"] = 150000

    X, y, _ = build_features(df)

    assert X.shape[0] == len(df) - 1
