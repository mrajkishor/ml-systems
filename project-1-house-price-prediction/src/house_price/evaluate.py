from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


def compute_regression_metrics(y_true, y_pred) -> RegressionMetrics:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)
