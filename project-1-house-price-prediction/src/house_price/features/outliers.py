import logging

import pandas as pd

from house_price.config import TARGET_COLUMN

logger = logging.getLogger(__name__)


def remove_known_outliers(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Drop the two documented Ames Housing outliers (per De Cock): very large homes
    (GrLivArea > 4000) sold at an anomalously low price (< $300,000).
    """
    if "GrLivArea" not in df.columns or target not in df.columns:
        return df

    mask = (df["GrLivArea"] > 4000) & (df[target] < 300000)
    if mask.any():
        logger.info("Dropping %d known outlier row(s) (large GrLivArea, low SalePrice)", mask.sum())
    return df.loc[~mask].copy()


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """Drop rows where any of `columns` falls outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    df = df.copy()
    keep_mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        keep_mask &= df[col].between(lower, upper)

    dropped = (~keep_mask).sum()
    if dropped:
        logger.info("Dropping %d row(s) outside IQR bounds for %s", dropped, columns)
    return df.loc[keep_mask]
