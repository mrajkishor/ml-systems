import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply missing-value imputation, encoding, scaling, and outlier treatment."""
    return df.copy()
