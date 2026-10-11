"""Direct multi-horizon features: origin t uses data no later than t."""
import numpy as np
import pandas as pd

FEATURES = ["latest", "lag_1", "lag_2", "lag_3", "lag_7", "mean_7", "mean_14",
            "target_day_sin", "target_day_cos", "target_month"]


def daily_series(frame: pd.DataFrame) -> pd.Series:
    if frame["city"].nunique() != 1:
        raise ValueError("Train each city independently.")
    series = frame.sort_values("date").set_index("date")["tmax"]
    if series.index.has_duplicates or series.empty:
        raise ValueError("Daily temperatures must have unique dates and at least one row.")
    # Reindex calendar gaps to NaN so a lag always means an actual day.
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))


def make_features(series: pd.Series, horizon: int) -> pd.DataFrame:
    if horizon not in (1, 2, 3):
        raise ValueError("Supported horizons are 1, 2 and 3 days.")
    target_dates = series.index + pd.Timedelta(days=horizon)
    features = pd.DataFrame(index=series.index)
    features["latest"] = series
    for lag in (1, 2, 3, 7):
        features[f"lag_{lag}"] = series.shift(lag)
    features["mean_7"] = series.rolling(7, min_periods=7).mean()
    features["mean_14"] = series.rolling(14, min_periods=14).mean()
    angle = 2 * np.pi * target_dates.dayofyear / 365.25
    features["target_day_sin"] = np.sin(angle)
    features["target_day_cos"] = np.cos(angle)
    features["target_month"] = target_dates.month
    features["target"] = series.shift(-horizon)
    features["target_date"] = target_dates
    features.index.name = "origin"
    return features
