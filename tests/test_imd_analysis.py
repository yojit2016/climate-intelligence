"""Tiny explicit fixtures test aggregation semantics, not substitute weather."""
import numpy as np
import pandas as pd
import pytest

from climate.imd_analysis import grid_statistics, nearest_available
from climate.imd_format import IMDFormatError


def test_nearest_available_ignores_masked_cell_and_reports_distance_and_coverage():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"]),
                          "latitude": [28.5, 28.5, 29.5, 28.5],
                          "longitude": [77.5, 78.5, 77.5, 78.5],
                          "maximum_temperature_c": [np.nan, 30., 31., 32.]})
    cell = nearest_available(frame, 28.5, 77.5)
    assert (cell.latitude, cell.longitude, cell.valid_days) == (28.5, 78.5, 2)
    assert 97 < cell.distance_km < 99
    with pytest.raises(IMDFormatError, match="extent"):
        nearest_available(frame, 90, 0)
    with pytest.raises(IMDFormatError, match="No available"):
        nearest_available(frame.assign(maximum_temperature_c=np.nan), 28.5, 77.5)


def test_statistics_exclude_missing_and_retain_all_missing_dates_and_counts():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-03-01"]),
                          "maximum_temperature_c": [10., 20., np.nan, 30.], "data_source": "IMD test"})
    tables = grid_statistics(frame)
    assert len(tables["daily"]) == 3 and len(tables["monthly"]) == 2
    january = tables["monthly"].iloc[0]
    assert january.sample_mean_tmax_c == 15 and january.valid_cell_days == 2
    assert january.total_cell_days == 3 and january.missing_cell_days == 1
    assert np.isnan(tables["daily"].iloc[1].sample_mean_tmax_c)
    assert set(tables["seasonal"].season) == {"Winter", "Pre-monsoon"}
    assert tables["seasonal"].valid_cell_days.sum() == 3
    assert tables["daily"].data_source.eq("IMD test").all()
