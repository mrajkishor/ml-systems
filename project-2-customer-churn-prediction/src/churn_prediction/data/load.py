import logging
from pathlib import Path

import pandas as pd

from churn_prediction.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)


def load_raw_data(filename: str = "train.csv", data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    path = data_dir / filename
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path)


def fetch_telco_churn() -> pd.DataFrame:
    """Fetch the IBM Telco Customer Churn dataset via OpenML (data_id=42178).

    7043 rows, 20 columns, binary `Churn` target (Yes/No), ~26.5% positive rate.
    """
    from sklearn.datasets import fetch_openml

    logger.info("Fetching Telco Customer Churn dataset from OpenML (data_id=42178)")
    bunch = fetch_openml(data_id=42178, as_frame=True, parser="auto")
    return bunch.frame


def get_dataset(filename: str = "train.csv", data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the dataset from disk if present, otherwise fetch it and cache it locally."""
    path = data_dir / filename
    if path.exists():
        return load_raw_data(filename=filename, data_dir=data_dir)

    df = fetch_telco_churn()
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Cached fetched dataset to %s", path)
    return df
