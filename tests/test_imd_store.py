from datetime import date

import numpy as np
import pandas as pd
import pytest

from climate.imd_charts import grid_heatmap
from climate.imd_format import IMDFormatError
from climate.imd_parser import parse_grd
from climate.imd_store import IMDStore
from climate.model import train_and_predict


def annual(year):
    """Synthetic fixture only; the app has no generated-weather data path."""
    days = 366 if year % 4 == 0 else 365
    y, x = np.indices((31, 31))
    values = 25 + 6 * np.sin(np.arange(days)[:, None, None] / 45) + y[None] * .1 + x[None] * .05
    payload = values.astype("<f4").tobytes()
    return payload, parse_grd(payload, product="historical", year=year, byte_order="<")


def test_storage_day_filter_heatmap_and_export_schema(tmp_path):
    payload, parsed = annual(2025)
    store = IMDStore(tmp_path)
    identity = store.save(parsed, payload, "../MAX2025.GRD")
    catalog = store.catalog()
    assert len(catalog) == 1 and catalog[0]["original_filename"] == "MAX2025.GRD"
    day = store.day(identity, date(2025, 6, 1))
    assert len(day) == 961 and day.date.nunique() == 1
    assert list(day.columns) == ["date", "latitude", "longitude", "maximum_temperature_c", "data_source"]
    figure = grid_heatmap(day, "IMD test fixture", parsed.metadata["minimum_c"], parsed.metadata["maximum_c"])
    assert np.asarray(figure.data[0].z).shape == (31, 31)
    assert figure.data[0].y[0] == 7.5 and figure.data[0].x[0] == 67.5
    assert "heatmap" in figure.to_json()
    assert len(list((tmp_path / "raw").glob("*.grd"))) == 1
    assert store.day(identity, date(2024, 1, 1)).empty
    with pytest.raises(IMDFormatError):
        store.parquet_path("../../bad")


def test_grid_cell_series_trains_with_real_pipeline_and_rejects_overlapping_imports(tmp_path):
    store = IMDStore(tmp_path)
    payload, parsed = annual(2025)
    identity = store.save(parsed, payload, "MAX2025.GRD")
    cell = store.historical_cell([identity], 28.5, 77.5)
    assert cell.city.nunique() == 1 and cell.source.str.startswith("IMD").all()
    assert cell.tmin.isna().all() and cell.tmean.isna().all()
    result = train_and_predict(cell)
    assert len(result.predictions) == 3 and len(result.metrics) == 12
    assert result.predictions.date.min().date() == date(2026, 1, 1)
    with pytest.raises(IMDFormatError, match="coordinate"):
        store.historical_cell([identity], 28.6, 77.5)
    changed = bytearray(payload)
    changed[:4] = np.array([26.], dtype="<f4").tobytes()
    revised = parse_grd(bytes(changed), product="historical", year=2025, byte_order="<")
    second = store.save(revised, bytes(changed), "MAX2025_revised.GRD")
    with pytest.raises(IMDFormatError, match="overlap"):
        store.historical_cell([identity, second], 28.5, 77.5)
