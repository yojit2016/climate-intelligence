"""Purged chronological partitions and shared baseline/model scoring."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelUnavailable(ValueError):
    """Insufficient usable historical data for a defensible forecast."""


def chronological_split(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = samples.sort_index()
    if len(ordered) < 120:
        raise ModelUnavailable("At least 120 complete feature/target rows are required; select roughly six months or more.")
    first, second = int(len(ordered) * .70), int(len(ordered) * .85)
    validation_start, test_start = ordered.index[first], ordered.index[second]
    train = ordered.iloc[:first]
    validation = ordered.iloc[first:second]
    test = ordered.iloc[second:]
    # Labels crossing a boundary would reveal information from the next block.
    train = train.loc[train["target_date"] < validation_start]
    validation = validation.loc[validation["target_date"] < test_start]
    if min(len(train), len(validation), len(test)) < 15:
        raise ModelUnavailable("Not enough independent rows in chronological evaluation blocks.")
    return train, validation, test


def scores(actual, predicted) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}


def persistence(samples: pd.DataFrame) -> np.ndarray:
    return samples["latest"].to_numpy()
