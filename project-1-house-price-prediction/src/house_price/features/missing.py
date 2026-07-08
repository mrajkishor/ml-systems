import pandas as pd

# Columns where NaN encodes "does not have this feature", not a truly missing value.
NONE_MEANS_ABSENT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
ZERO_MEANS_ABSENT = [
    "GarageYrBlt", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]


def impute_domain_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs using Ames Housing domain knowledge, then fall back to median/mode."""
    df = df.copy()

    for col in NONE_MEANS_ABSENT:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_MEANS_ABSENT:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "LotFrontage" in df.columns:
        if "Neighborhood" in df.columns:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda s: s.fillna(s.median())
            )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].apply(lambda s: s.fillna(s.median()))

    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df
