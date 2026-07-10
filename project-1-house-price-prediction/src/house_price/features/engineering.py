import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from house_price.config import TARGET_COLUMN
from house_price.features.missing import impute_domain_missing_values
from house_price.features.outliers import remove_known_outliers

# Ordered quality scales used throughout the Ames Housing data dictionary.
ORDINAL_QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
BSMT_EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}


def encode_ordinal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordered quality categories (Po/Fa/TA/Gd/Ex/None) to integers instead of one-hot,
    preserving their natural ranking for models that benefit from it.
    """
    df = df.copy()
    for col in ORDINAL_QUALITY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(ORDINAL_QUALITY_MAP).fillna(0).astype(int)
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(BSMT_EXPOSURE_MAP).fillna(0).astype(int)
    return df


def _domain_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = impute_domain_missing_values(df)
    df = encode_ordinal_quality(df)
    return df


def build_feature_pipeline(df: pd.DataFrame) -> Pipeline:
    """Build an unfitted sklearn Pipeline: domain imputation + ordinal encoding, then
    impute+scale remaining numeric columns and impute+one-hot remaining categoricals.
    """
    preview = _domain_preprocess(df)
    numeric_cols = preview.select_dtypes(include="number").columns.tolist()
    categorical_cols = preview.select_dtypes(include="object").columns.tolist()

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    column_transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    return Pipeline([
        ("domain_prep", FunctionTransformer(_domain_preprocess, feature_names_out="one-to-one")),
        ("column_transform", column_transformer),
    ])


def build_features(
    df: pd.DataFrame, target: str = TARGET_COLUMN, drop_outliers: bool = True
) -> tuple[pd.DataFrame | None, pd.Series | None, Pipeline]:
    """Apply outlier treatment, then fit the missing-value/encoding/scaling pipeline.

    Returns (X_transformed, y, fitted_pipeline). The fitted pipeline can be reused via
    `.transform(new_df)` at inference time to apply identical preprocessing.
    """
    df = df.copy()
    if drop_outliers and target in df.columns:
        df = remove_known_outliers(df, target=target)

    y = df[target] if target in df.columns else None
    X = df.drop(columns=[target], errors="ignore")

    pipeline = build_feature_pipeline(X)
    X_transformed = pipeline.fit_transform(X)
    return X_transformed, y, pipeline
