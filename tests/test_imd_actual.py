"""Opt-in-by-presence checks of the user's actual official IMD 2025 binary.

No substitute fixture is used when the file is absent from another checkout.
"""
from datetime import date
from pathlib import Path
import struct

import numpy as np
import pytest

from climate.imd_parser import parse_grd
from climate.imd_store import IMDStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_2025():
    paths = [ROOT / "data/raw/imd/Maxtemp_MaxT_2025.GRD",
             ROOT / "data/Maxtemp_MaxT_2025.GRD"]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        pytest.skip("Actual official IMD 2025 GRD not present; no replacement data used")
    payload = path.read_bytes()
    return payload, parse_grd(payload, product="historical", year=2025, byte_order="<")


def test_actual_2025_binary_counts_range_and_official_array_order(actual_2025):
    payload, parsed = actual_2025
    assert len(payload) == 1_403_060
    assert parsed.metadata["sha256"] == "673775eafe685d180c5d7a19018287599697a3f6800332470ab27ca512bd4b88"
    assert parsed.metadata["valid_cells"] == 129_552
    assert parsed.metadata["missing_cells"] == 221_213
    assert parsed.metadata["minimum_c"] == pytest.approx(5.30673885345459)
    assert parsed.metadata["maximum_c"] == pytest.approx(46.15256118774414)
    cube = parsed.cube()
    assert cube.maximum_temperature_c.shape == (365, 31, 31)
    assert cube.dates[0].date() == date(2025, 1, 1)
    assert cube.dates[-1].date() == date(2025, 12, 31)
    assert np.isnan(cube.maximum_temperature_c).sum() == 221_213
    # IMD C sample: t[i][j], j inner. Fortran: ((T(day,I,J),J),I).
    # CTL XDEF 31 LINEAR 67.5 1; YDEF 31 LINEAR 7.5 1, no YREV.
    # Independently read every float at its byte offset to detect transpose/flip.
    for day in (0, 104, 364):
        for i in range(31):
            for j in range(31):
                value = struct.unpack_from("<f", payload, ((day * 31 + i) * 31 + j) * 4)[0]
                actual = cube.maximum_temperature_c[day, i, j]
                assert (np.isnan(actual) if value == np.float32(99.9) else actual == value)
                assert cube.latitudes[i] == 7.5 + i
                assert cube.longitudes[j] == 67.5 + j


def test_actual_2025_parquet_round_trip(actual_2025, tmp_path):
    payload, parsed = actual_2025
    store = IMDStore(tmp_path)
    identity = store.save(parsed, payload, "Maxtemp_MaxT_2025.GRD")
    for selected in (date(2025, 1, 1), date(2025, 4, 15), date(2025, 12, 31)):
        daily = store.day(identity, selected)
        expected = parsed.frame[parsed.frame.date.dt.date == selected].reset_index(drop=True)
        assert daily.reset_index(drop=True).equals(expected)
    assert store.historical_cell([identity], 28.5, 77.5).tmax.notna().sum() == 365
