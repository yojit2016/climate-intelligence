"""Descriptive, centred Savitzky-Golay filtering, never used for ML."""
import pandas as pd
from scipy.signal import savgol_filter


def smooth_temperature(frame: pd.DataFrame, window: int = 7) -> pd.Series:
    if window < 3 or window % 2 == 0:
        raise ValueError("Smoothing window must be odd and at least three days.")
    result = pd.Series(float("nan"), index=frame.index, dtype=float)
    # Filter only contiguous daily runs of valid values; do not bridge gaps.
    for _, group in frame.groupby("city", sort=False):
        group = group.sort_values("date")
        valid = group["tmax"].notna()
        breaks = (~valid | ~valid.shift(fill_value=False)
                  | group["date"].diff().ne(pd.Timedelta(days=1))).cumsum()
        for _, run in group.loc[valid].groupby(breaks[valid]):
            if len(run) >= window:
                result.loc[run.index] = savgol_filter(run["tmax"], window, polyorder=2)
    return result
