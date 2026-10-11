import numpy as np
import pandas as pd
import pytest

from climate.analytics import daily_trends, monthly_patterns, regional_summary, seasonal_patterns, summary
from climate.charts import comparison_chart, pattern_chart, range_chart, trend_chart
from climate.export import csv_bytes


def frame():
    return pd.DataFrame({"date": pd.date_range("2025-01-25", periods=20, tz="Asia/Kolkata"),
                         "city": "Delhi", "tmax": np.arange(20.) + 25,
                         "tmin": np.arange(20.) + 15, "tmean": np.arange(20.) + 20})


def test_stats_patterns_and_threshold_are_computed_from_data():
    data = frame()
    stats = summary(data, 40)
    assert stats["exceedance_days"] == 4 and stats["highest"] == 44
    assert stats["lowest"] == 15 and stats["mean_max"] == 34.5
    months = monthly_patterns(data)
    assert months.valid_days.tolist() == [7, 13]
    assert seasonal_patterns(data).season.tolist() == ["Winter"]
    trend = daily_trends(data, 40)
    assert trend.rolling_7d.iloc[6] == 28
    assert trend.exceeds_threshold.sum() == 4
    assert len(regional_summary(data, 40)) == 1


def test_charts_render_and_exports_preserve_timezone_and_missingness():
    data = frame()
    figures = [trend_chart(daily_trends(data, 40), 40), range_chart(data),
               comparison_chart(data), pattern_chart(monthly_patterns(data), "month", "Monthly")]
    for fig in figures:
        assert len(fig.data) > 0
        assert len(fig.to_json()) > 100
    data.loc[0, "tmax"] = np.nan
    encoded = csv_bytes(data)
    assert "+0530" in encoded.decode("utf-8-sig")
    exported = pd.read_csv(__import__("io").BytesIO(encoded))
    assert pd.isna(exported.loc[0, "tmax"])
    assert not any(name.startswith("Unnamed") for name in exported.columns)
