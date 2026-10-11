from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from climate.api import WeatherClient, WeatherError
from climate.cache import ParquetCache
from climate.locations import CITIES
from climate.weather import current_and_forecast, historical, parse_daily


def payload(dates):
    return {"daily": {"time": dates, "temperature_2m_max": [32.] * len(dates),
                      "temperature_2m_min": [22.] * len(dates),
                      "temperature_2m_mean": [27.] * len(dates)}}


def test_parser_validates_shape_and_retains_provenance():
    parsed = parse_daily(payload(["2025-01-01"]), CITIES["Delhi"], "test provenance")
    assert str(parsed.date.dt.tz) == "Asia/Kolkata"
    assert parsed.source.iloc[0] == "test provenance"
    assert str(parsed.retrieved_at.dt.tz) == "UTC"
    with pytest.raises(WeatherError):
        parse_daily({"daily": {}}, CITIES["Delhi"], "bad")


def test_http_failure_and_malformed_json_are_explicit():
    session = Mock()
    session.get.side_effect = requests.Timeout("timed out")
    with pytest.raises(WeatherError):
        WeatherClient(session).get("https://example.test", {})
    session.get.side_effect = None
    session.get.return_value.json.return_value = {"error": True, "reason": "rate limit"}
    with pytest.raises(WeatherError, match="rate limit"):
        WeatherClient(session).get("https://example.test", {})


def test_history_uses_archive_only_even_for_recent_completed_days(monkeypatch):
    monkeypatch.setattr("climate.weather.local_today", lambda city: date(2025, 1, 10))
    client = Mock()
    client.get.return_value = payload(["2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07"])
    frame = historical(client, CITIES["Delhi"], date(2025, 1, 4), date(2025, 1, 7))
    assert frame.date.dt.day.tolist() == [4, 5, 6, 7]
    assert frame.source.unique().tolist() == ["Historical reanalysis / model estimate"]
    client.get.assert_called_once()
    url, parameters = client.get.call_args.args
    assert url == "https://archive-api.open-meteo.com/v1/archive"
    assert parameters["start_date"] == "2025-01-04" and parameters["end_date"] == "2025-01-07"
    assert "past_days" not in parameters
    with pytest.raises(ValueError):
        historical(client, CITIES["Delhi"], date(2025, 1, 4), date(2025, 1, 10))


def test_parquet_cache_roundtrip_and_expiry(tmp_path):
    cache = ParquetCache(tmp_path)
    frame = parse_daily(payload(["2025-01-01"]), CITIES["Delhi"], "test")
    key = {"city": "Delhi", "start": "2025-01-01"}
    assert cache.read(key) is None
    cache.write(key, frame)
    pd.testing.assert_frame_equal(cache.read(key), frame)
    assert cache.read(key, ttl=-1) is None
    cache.path(key).write_bytes(b"corrupt parquet")
    assert cache.read(key) is None


def test_invalid_current_structure_is_an_explicit_weather_error():
    client = Mock()
    client.get.return_value = {**payload(["2025-01-01"]), "current": []}
    with pytest.raises(WeatherError, match="current conditions"):
        current_and_forecast(client, CITIES["Delhi"])


def test_cache_cleanup_failure_does_not_break_ingestion(tmp_path, monkeypatch):
    from pathlib import Path
    cache = ParquetCache(tmp_path)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", Mock(side_effect=OSError("read only")))
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError("cannot remove")))
    cache.write({"city": "Delhi"}, pd.DataFrame({"tmax": [32.]}))
