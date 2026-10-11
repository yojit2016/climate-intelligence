from datetime import date

import numpy as np
import pytest

from climate.imd_format import IMDFormatError
from climate.imd_parser import parse_grd


def binary(days=365, size=31, byte_order="<"):
    """Deliberately synthetic parser fixture, never an application data source."""
    y, x = np.indices((size, size))
    values = (20 + y[None] * .2 + x[None] * .1 + np.arange(days)[:, None, None] * .01).astype("f4")
    values[0, 0, 0] = 99.9
    return values.astype(f"{byte_order}f4").tobytes()


def ctl(size=61, step=.5, days=1, origin="1JAN2025", options="little_endian"):
    return f"UNDEF 99.9\nXDEF {size} LINEAR 67.5 {step}\nYDEF {size} LINEAR 7.5 {step}\nZDEF 1 LINEAR 1 1\nTDEF {days} LINEAR {origin} 1DY\nOPTIONS {options}\nVARS 1\nT 0 99 Tmax\nENDVARS"


def test_annual_layout_coordinates_missing_values_and_dates():
    parsed = parse_grd(binary(), product="historical", year=2025, byte_order="<")
    frame = parsed.frame
    assert len(frame) == 365 * 31 * 31
    assert frame.iloc[0].latitude == 7.5 and frame.iloc[0].longitude == 67.5
    assert np.isnan(frame.iloc[0].maximum_temperature_c)
    assert frame.iloc[1].longitude == 68.5 and frame.iloc[1].latitude == 7.5
    assert frame.iloc[31].latitude == 8.5 and frame.iloc[31].longitude == 67.5
    assert frame.iloc[1].maximum_temperature_c == pytest.approx(20.1)
    assert frame.iloc[-1].latitude == 37.5 and frame.iloc[-1].longitude == 97.5
    assert frame.date.min().date() == date(2025, 1, 1) and frame.date.max().date() == date(2025, 12, 31)
    assert parsed.metadata["missing_cells"] == 1


def test_leap_year_size_and_explicit_big_endian():
    payload = binary(366, byte_order=">")
    parsed = parse_grd(payload, product="historical", year=2024, byte_order=">")
    assert parsed.metadata["days"] == 366 and len(parsed.frame.date.unique()) == 366
    with pytest.raises(IMDFormatError, match="Expected exactly"):
        parse_grd(payload, product="historical", year=2025, byte_order=">")
    with pytest.raises(IMDFormatError, match="float values"):
        parse_grd(payload, product="historical", year=2024, byte_order="<")


def test_recent_grid_requires_matching_control_and_rejects_orientation_changes():
    payload = binary(1, 61)
    parsed = parse_grd(payload, product="recent", day=date(2025, 1, 1), byte_order="<", control_text=ctl())
    assert len(parsed.frame) == 61 ** 2
    assert parsed.frame.iloc[1].longitude == 68.
    for bad_control in (ctl(options="yrev"), ctl(options="big_endian"), ctl(size=31, step=1), ctl(origin="2JAN2025")):
        with pytest.raises(IMDFormatError):
            parse_grd(payload, product="recent", day=date(2025, 1, 1), byte_order="<", control_text=bad_control)
    with pytest.raises(IMDFormatError, match="matching official"):
        parse_grd(payload, product="recent", day=date(2025, 1, 1), byte_order="<")


def test_corrupt_truncated_missing_and_constant_payloads_are_rejected():
    good = binary()
    for payload in (good[:-4], good + b"junk", b"<html>authentication required</html>"):
        with pytest.raises(IMDFormatError, match="Expected exactly"):
            parse_grd(payload, product="historical", year=2025, byte_order="<")
    for bad in (np.inf, np.nan, 80., -100., 1e-42):
        values = np.frombuffer(good, dtype="<f4").copy()
        values[1] = bad
        with pytest.raises(IMDFormatError, match="float values"):
            parse_grd(values.tobytes(), product="historical", year=2025, byte_order="<")
    for constant in (0., 99.9):
        with pytest.raises(IMDFormatError):
            parse_grd(np.full(365 * 961, constant, dtype="<f4").tobytes(),
                      product="historical", year=2025, byte_order="<")
