from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from climate.api import WeatherError
from climate.acquisition import AcquisitionResult, training_period
from climate.locations import CITIES
from climate.processing import clean_daily
from climate.weather import local_today

APP = Path(__file__).resolve().parents[1] / "app.py"


def history(city, start, end, **kwargs):
    """Explicitly patched offline test data; never used by app runtime."""
    dates = pd.date_range(start, end, tz="Asia/Kolkata")
    values = 30 + 5 * np.sin(np.arange(len(dates)) / 30)
    raw = pd.DataFrame({"date": dates, "city": city, "tmax": values,
                        "tmin": values - 10, "tmean": values - 5,
                        "source": "OFFLINE TEST FIXTURE", "retrieved_at": pd.Timestamp.now(tz="UTC")})
    processed, report = clean_daily(raw)
    return raw, processed, report


def current(city, **kwargs):
    today = local_today(CITIES[city])
    dates = pd.date_range(today, today + timedelta(days=3), tz="Asia/Kolkata")
    forecast = pd.DataFrame({"date": dates, "tmax": 32., "tmin": 22., "tmean": 27.,
                             "source": "OFFLINE TEST FIXTURE"})
    return {"time": today.isoformat(), "temperature_2m": 29., "relative_humidity_2m": 60.}, forecast


def acquired(as_of, **kwargs):
    start, end = training_period(as_of)
    # Shorter explicit offline training fixture keeps UI tests quick.
    data = {name: history(name, end - timedelta(days=240), end) for name in CITIES}
    return AcquisitionResult(start, end, data)


def test_dashboard_renders_and_filter_submission_changes_city():
    with patch("climate.ui.cached_history", side_effect=history), patch("climate.ui.live_weather", side_effect=current), \
         patch("climate.ui.cached_acquisition", side_effect=acquired):
        app = AppTest.from_file(APP, default_timeout=60).run()
        assert not app.exception
        assert [tab.label for tab in app.tabs] == ["Overview", "Historical Analysis", "ML Forecasting", "Data Processing"]
        assert len(app.metric) == 6
        assert any("What the displayed data shows" in item.value for item in app.markdown)
        assert any("Did the model beat" in item.value for item in app.markdown)
        assert sum(item.label == "How to read this chart" for item in app.expander) == len(app.get("plotly_chart"))
        assert any("Held-out model evaluation" in item.value for item in app.subheader)
        app.selectbox[0].select("Pune")
        app.button[0].click().run()
        assert not app.exception
        assert any("Pune" in caption.value for caption in app.caption)


def test_dashboard_explains_api_failure():
    with patch("climate.ui.cached_history", side_effect=WeatherError("Offline integration test")), \
         patch("climate.ui.live_weather", side_effect=WeatherError("Offline integration test")), \
         patch("climate.ui.cached_acquisition", return_value=AcquisitionResult(
             *training_period(local_today(CITIES["Delhi"])), errors={"Delhi": "Offline integration test"})):
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        assert any("Historical data unavailable" in error.value for error in app.error)


def test_dashboard_explains_insufficient_history_without_predictions():
    def short_history(city, start, end, **kwargs):
        return history(city, end - timedelta(days=30), end)
    def short_acquired(as_of, **kwargs):
        start, end = training_period(as_of)
        return AcquisitionResult(start, end, {name: short_history(name, start, end) for name in CITIES})
    with patch("climate.ui.cached_history", side_effect=short_history), \
         patch("climate.ui.live_weather", side_effect=current), \
         patch("climate.ui.cached_acquisition", side_effect=short_acquired):
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        assert any("Local ML predictions unavailable" in info.value for info in app.info)
        assert not any("Held-out model evaluation" in header.value for header in app.subheader)


def test_refresh_bypasses_download_caches_and_retrains():
    with patch("climate.ui.cached_history", side_effect=history) as historical_loader, \
         patch("climate.ui.live_weather", side_effect=current) as current_loader, \
         patch("climate.ui.cached_acquisition", side_effect=acquired) as acquisition_loader, \
         patch("climate.forecast_panel.cached_model.clear") as clear_models:
        app = AppTest.from_file(APP, default_timeout=60).run()
        assert not app.exception
        next(button for button in app.button if button.label == "Refresh weather data").click().run()
        assert not app.exception
        assert any(call.kwargs.get("force_refresh") for call in historical_loader.call_args_list)
        assert historical_loader.call_args.kwargs["force_refresh"]
        assert current_loader.call_args.kwargs["force_refresh"]
        assert acquisition_loader.call_args.kwargs["force_refresh"]
        clear_models.assert_called_once()


def test_chart_filters_do_not_shorten_automatic_training_history():
    def short_history(city, start, end, **kwargs):
        return history(city, end - timedelta(days=30), end)
    with patch("climate.ui.cached_history", side_effect=short_history), \
         patch("climate.ui.live_weather", side_effect=current), \
         patch("climate.ui.cached_acquisition", side_effect=acquired) as acquisition_loader:
        app = AppTest.from_file(APP, default_timeout=60).run()
        assert not app.exception
        assert any("Held-out model evaluation" in header.value for header in app.subheader)
        assert acquisition_loader.call_args.args == (local_today(CITIES["Delhi"]),)


def test_history_views_keep_patterns_comparisons_smoothing_and_processing_accessible():
    with patch("climate.ui.cached_history", side_effect=history) as loader, \
         patch("climate.ui.live_weather", side_effect=current), \
         patch("climate.ui.automatic_training"):
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        view = next(item for item in app.radio if item.label == "Explore history")
        view.set_value("Monthly & seasonal").run()
        assert not app.exception
        assert sum("highest displayed average daily maximum" in item.value for item in app.markdown) == 2
        next(item for item in app.radio if item.label == "Explore history").set_value("Across cities").run()
        assert not app.exception
        assert any("Compare daily maximum estimates" in item.value for item in app.markdown)
        next(item for item in app.radio if item.label == "Explore history").set_value("Daily temperatures")
        next(item for item in app.checkbox if item.label == "Show signal smoothing").check()
        next(item for item in app.button if item.label == "Apply filters").click().run()
        assert not app.exception
        assert any("smoothed daily maximum" in item.value for item in app.markdown)
        assert any("Detailed processing report" in item.label for item in app.expander)
        assert any("Download raw API table CSV" == item.label for item in app.get("download_button"))
        # Refresh while comparisons are hidden, then visit them: requests still bypass disk.
        next(item for item in app.button if item.label == "Refresh weather data").click().run()
        assert not app.exception
        next(item for item in app.radio if item.label == "Explore history").set_value("Across cities").run()
        assert not app.exception
        assert all(call.kwargs["force_refresh"] for call in loader.call_args_list[-2:])


def test_missing_temperatures_do_not_imply_zero_exceedances_or_a_conclusion():
    def missing_history(city, start, end, **kwargs):
        raw, _, _ = history(city, start, end)
        raw[["tmax", "tmin", "tmean"]] = np.nan
        processed, report = clean_daily(raw)
        return raw, processed, report
    with patch("climate.ui.cached_history", side_effect=missing_history), \
         patch("climate.ui.live_weather", side_effect=current), \
         patch("climate.ui.automatic_training"):
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        assert next(item for item in app.metric if item.label == "Days above exploratory threshold").value == "Unavailable"
        assert any("No valid temperatures" in item.value for item in app.markdown)
        assert any("cannot establish complete coverage" in item.value for item in app.info)
