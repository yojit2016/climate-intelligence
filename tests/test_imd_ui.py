from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from climate.imd_parser import parse_grd
from climate.imd_store import IMDStore

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_empty_imd_workspace_never_calls_open_meteo(tmp_path):
    store = IMDStore(tmp_path)
    with patch("climate.imd_ui.IMDStore", return_value=store), \
         patch("climate.ui.cached_history") as history, patch("climate.ui.live_weather") as current, \
         patch("climate.ui.cached_acquisition") as acquisition:
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state["data_workspace"] = "IMD gridded temperature"
        app.run()
        assert not app.exception
        assert any("No IMD file has been imported" in info.value for info in app.info)
        history.assert_not_called()
        current.assert_not_called()
        acquisition.assert_not_called()
        assert len(app.get("plotly_chart")) == 0


def test_imported_imd_workspace_maps_dates_and_evaluates_grid_cell(tmp_path):
    # Explicit synthetic integration fixture, not an application data provider.
    y, x = np.indices((31, 31))
    values = 25 + 5 * np.sin(np.arange(365)[:, None, None] / 40) + y[None] * .1 + x[None] * .05
    payload = values.astype("<f4").tobytes()
    store = IMDStore(tmp_path)
    store.save(parse_grd(payload, product="historical", year=2025, byte_order="<"), payload, "TEST_MAX2025.GRD")
    with patch("climate.imd_ui.IMDStore", return_value=store):
        app = AppTest.from_string("from climate.imd_ui import render_imd\nrender_imd()", default_timeout=60).run()
        assert not app.exception
        assert [tab.label for tab in app.tabs] == ["Overview", "Historical Analysis", "ML Forecasting", "Data Processing"]
        assert len(app.get("plotly_chart")) == 3
        assert any("What the displayed data shows" in item.value for item in app.markdown)
        assert any("historical/static" in item.value for item in app.caption)
        assert sum(item.label == "How to read this chart" for item in app.expander) == len(app.get("plotly_chart"))
        app.date_input[0].set_value(date(2025, 7, 1)).run()
        assert not app.exception
        next(item for item in app.checkbox if item.label.startswith("Evaluate Random Forest")).check().run()
        assert not app.exception
        assert any("Historical endpoint experiment" in warning.value for warning in app.warning)
        assert len(app.get("plotly_chart")) == 5
