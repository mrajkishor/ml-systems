import pandas as pd


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges is stored as an object column with a blank ' ' string for the
    11 customers with tenure=0 (they haven't been billed yet) -- coerce to numeric
    and fill those with 0.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df
