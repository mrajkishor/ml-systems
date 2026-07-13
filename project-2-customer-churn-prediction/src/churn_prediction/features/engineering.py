import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_prediction.config import TARGET_COLUMN
from churn_prediction.features.missing import fix_total_charges

# Yes/No columns encoded as 0/1 rather than one-hot (keeps the feature count down;
# these are strictly binary, not multi-category nominal fields).
BINARY_YES_NO_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
GENDER_MAP = {"Female": 0, "Male": 1}
# Contract length is ordered (month-to-month < one year < two year) and that order
# is meaningful for churn risk, so it's ordinal-encoded rather than one-hot.
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def encode_binary_and_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAP)
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(CONTRACT_MAP)
    return df


def _domain_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = fix_total_charges(df)
    df = encode_binary_and_ordinal(df)
    return df


def build_feature_pipeline(df: pd.DataFrame) -> Pipeline:
    """Domain preprocessing (TotalCharges fix, binary/ordinal encoding), then
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


def build_features(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """Encode the target (Yes/No -> 1/0) and fit the feature pipeline.

    Returns (X_transformed, y, fitted_pipeline). The fitted pipeline can be reused via
    `.transform(new_df)` at inference time to apply identical preprocessing.
    """
    df = df.copy()
    y = df[target].map({"Yes": 1, "No": 0}) if target in df.columns else None
    X = df.drop(columns=[target], errors="ignore")

    pipeline = build_feature_pipeline(X)
    X_transformed = pipeline.fit_transform(X)
    return X_transformed, y, pipeline
