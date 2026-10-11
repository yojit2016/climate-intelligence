import numpy as np
import pandas as pd
import pytest

from climate.processing import aggregate_hourly, clean_daily, normalize_time
from climate.smoothing import smooth_temperature


def test_clean_preserves_raw_and_reports_bad_rows():
    raw = pd.DataFrame({"date": ["2025-01-01", "2025-01-01", "2025-01-02", "bad"],
        "city": ["Delhi"] * 4, "tmax": [25, 26, 200, 22],
        "tmin": [15, 16, 10, 12], "tmean": [20, 21, 18, 17]})
    original = raw.copy(deep=True)
    cleaned, report = clean_daily(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert len(cleaned) == 2 and cleaned.iloc[0].tmax == 26
    assert pd.isna(cleaned.iloc[1].tmax)
    assert report["duplicate_rows"] == 1 and report["invalid_dates"] == 1
    assert report["invalid_values_or_rows"] == 1


def test_inconsistent_temperatures_are_flagged():
    raw = pd.DataFrame({"date": ["2025-01-01"], "city": ["Delhi"],
                        "tmax": [20], "tmin": [25], "tmean": [22]})
    cleaned, _ = clean_daily(raw)
    assert cleaned[["tmax", "tmin", "tmean"]].isna().all(axis=None)


def test_utc_hourly_values_group_in_local_day_and_deduplicate():
    times = pd.Series(pd.to_datetime(["2025-01-01 20:00Z", "2025-01-01 21:00Z"]))
    assert normalize_time(times, "Asia/Kolkata").dt.day.tolist() == [2, 2]
    raw = pd.DataFrame({"time": [times[0], times[1], times[1]],
                        "city": ["Delhi"] * 3, "temperature": [20, 24, 24]})
    daily = aggregate_hourly(raw)
    assert daily.iloc[0].tmax == 24 and daily.iloc[0].tmean == 22
    assert daily.iloc[0].valid_hours == 2 and daily.iloc[0].partial_day


def test_smoothing_does_not_cross_missing_days_or_values():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=20, tz="Asia/Kolkata"),
                          "city": "Delhi", "tmax": np.arange(20, dtype=float)})
    frame.loc[9, "tmax"] = np.nan
    filtered = smooth_temperature(frame, 7)
    assert pd.isna(filtered[9])
    assert filtered[:9].notna().all() and filtered[10:].notna().all()
    with pytest.raises(ValueError):
        smooth_temperature(frame, 4)
    gap = frame.drop(index=range(5, 15))
    assert smooth_temperature(gap, 7).isna().all()
