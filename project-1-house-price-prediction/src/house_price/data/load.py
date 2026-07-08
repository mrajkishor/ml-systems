import logging
from pathlib import Path

import pandas as pd

from house_price.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)


def load_raw_data(filename: str = "train.csv", data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    path = data_dir / filename
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path)
