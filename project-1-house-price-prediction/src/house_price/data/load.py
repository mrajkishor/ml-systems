import logging
from pathlib import Path

import pandas as pd

from house_price.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)


def load_raw_data(filename: str = "train.csv", data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    path = data_dir / filename
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path)


def fetch_ames_housing() -> pd.DataFrame:
    """Fetch the Ames Housing dataset (Kaggle House Prices competition data) via OpenML."""
    from sklearn.datasets import fetch_openml

    logger.info("Fetching Ames Housing dataset from OpenML (house_prices)")
    bunch = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    return bunch.frame


def get_dataset(filename: str = "train.csv", data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the dataset from disk if present, otherwise fetch it and cache it locally.

    Prefers a manually downloaded Kaggle `train.csv` in `data_dir` if one exists;
    falls back to the OpenML mirror of the same dataset when it doesn't.
    """
    path = data_dir / filename
    if path.exists():
        return load_raw_data(filename=filename, data_dir=data_dir)

    df = fetch_ames_housing()
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Cached fetched dataset to %s", path)
    return df
