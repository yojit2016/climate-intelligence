from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from climate.api import WeatherError
from climate.locations import CITIES
from climate.processing import clean_daily
from climate.weather import local_today

APP = Path(__file__).resolve().parents[1] / "app.py"


def history(city, start, end):
    """Explicitly patched offline test data; never used by app runtime."""
    dates = pd.date_range(start, end, tz="Asia/Kolkata")
    values = 30 + 5 * np.sin(np.arange(len(dates)) / 30)
    raw = pd.DataFrame({"date": dates, "city": city, "tmax": values,
                        "tmin": values - 10, "tmean": values - 5,
                        "source": "OFFLINE TEST FIXTURE", "retrieved_at": pd.Timestamp.now(tz="UTC")})
    processed, report = clean_daily(raw)
    return raw, processed, report


def current(city):
    today = local_today(CITIES[city])
    dates = pd.date_range(today, today + timedelta(days=3), tz="Asia/Kolkata")
    forecast = pd.DataFrame({"date": dates, "tmax": 32., "tmin": 22., "tmean": 27.,
                             "source": "OFFLINE TEST FIXTURE"})
    return {"time": today.isoformat(), "temperature_2m": 29., "relative_humidity_2m": 60.}, forecast


def test_dashboard_renders_and_filter_submission_changes_city():
    with patch("climate.ui.cached_history", side_effect=history), patch("climate.ui.live_weather", side_effect=current):
        app = AppTest.from_file(APP, default_timeout=60).run()
        assert not app.exception
        assert len(app.tabs) == 5 and len(app.metric) == 6
        assert any("Held-out model evaluation" in item.value for item in app.subheader)
        app.selectbox[0].select("Pune")
        app.button[0].click().run()
        assert not app.exception
        assert any("Pune" in caption.value for caption in app.caption)


def test_dashboard_explains_api_failure():
    with patch("climate.ui.cached_history", side_effect=WeatherError("Offline integration test")), \
         patch("climate.ui.live_weather", side_effect=WeatherError("Offline integration test")):
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        assert any("Historical data unavailable" in error.value for error in app.error)
