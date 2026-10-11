"""Explicit small grid fixtures verify analysis semantics, not weather data."""
import numpy as np
import pandas as pd
import pytest

from climate.imd_analysis import hottest_cells, compare_dates, difference_findings
from climate.imd_format import IMDFormatError


def day(values, stamp="2025-01-01"):
    return pd.DataFrame({"date": pd.Timestamp(stamp, tz="Asia/Kolkata"),
        "latitude": [10.5, 11.5, 12.5, 13.5], "longitude": [75.5] * 4,
        "maximum_temperature_c": values, "data_source": "IMD test product"})


def test_hotspots_rank_only_valid_cells_with_deterministic_ties_and_dates():
    table = hottest_cells(day([np.nan, 35., 40., 40.]))
    assert table["rank"].tolist() == [1, 2, 3]
    assert table.latitude.tolist() == [12.5, 13.5, 11.5]
    assert table.date.nunique() == 1 and table.data_source.eq("IMD test product").all()
    assert hottest_cells(day([np.nan] * 4)).empty
    with pytest.raises(IMDFormatError):
        hottest_cells(pd.concat([day([1.] * 4), day([2.] * 4, "2025-01-02")]))


def test_comparison_uses_exact_coordinates_and_never_fills_missing_values():
    a, b = day([30., np.nan, 28., 32.]), day([33., 35., np.nan, 29.], "2025-01-02")
    result = compare_dates(a, b.iloc[::-1])
    np.testing.assert_equal(result.difference_c.to_numpy(), [3., np.nan, np.nan, -3.])
    assert result.date_a.dt.day.eq(1).all() and result.date_b.dt.day.eq(2).all()
    assert a.maximum_temperature_c.iloc[0] == 30.  # inputs not mutated
    text = difference_findings(result)
    assert "2 cells" in text and "increase: 3.00°C at 10.5" in text and "decrease: 3.00°C at 13.5" in text
    with pytest.raises(IMDFormatError, match="same grid"):
        compare_dates(a, b.assign(longitude=76.5))
    with pytest.raises(IMDFormatError, match="same grid"):
        compare_dates(a, b.assign(data_source="Other provider"))


def test_same_date_no_overlap_and_one_sided_changes_are_honestly_described():
    a = day([np.nan, 20., 25., 30.])
    same = compare_dates(a, a)
    assert same.difference_c.dropna().eq(0).all()
    assert "No increases" in difference_findings(same) and "No decreases" in difference_findings(same)
    result = compare_dates(a, day([22., np.nan, np.nan, np.nan], "2025-01-02"))
    assert result.difference_c.isna().all() and "unavailable" in difference_findings(result)
    increased = compare_dates(a, day([np.nan, 21., 26., 31.], "2025-01-02"))
    assert "No decreases" in difference_findings(increased)
