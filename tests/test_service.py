from datetime import date
from unittest.mock import Mock

import pandas as pd

from climate.service import load_current, load_history


def test_ingestion_preserves_raw_and_materializes_absent_calendar_days(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-03"]).tz_localize("Asia/Kolkata"),
                        "city": "Delhi", "tmax": [30., 32.], "tmin": [20., 22.], "tmean": [25., 27.],
                        "source": "OFFLINE TEST FIXTURE", "retrieved_at": pd.Timestamp.now(tz="UTC")})
    upstream = Mock(return_value=raw)
    monkeypatch.setattr("climate.sources.historical", upstream)
    original = raw.copy(deep=True)
    ingested, processed, quality = load_history("Delhi", date(2025, 1, 1), date(2025, 1, 3))
    pd.testing.assert_frame_equal(ingested, original)
    assert len(processed) == 3 and quality["missing_calendar_days"] == 1
    assert pd.isna(processed.iloc[1].tmax)
    assert processed.iloc[1].source == "Missing upstream record"
    assert len(list((tmp_path / "data/raw").glob("*.parquet"))) == 1
    assert len(list((tmp_path / "data/processed").glob("*.parquet"))) == 1
    load_history("Delhi", date(2025, 1, 1), date(2025, 1, 3))
    assert upstream.call_count == 1


def test_refresh_bypasses_history_cache_and_provider_names_isolate_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]).tz_localize("Asia/Kolkata"),
                        "city": "Delhi", "tmax": [30.], "tmin": [20.], "tmean": [25.], "source": "TEST"})
    source = Mock(cache_id="test-provider")
    source.history.return_value = raw
    for refresh in (False, False, True):
        load_history("Delhi", date(2025, 1, 1), date(2025, 1, 1), source=source, force_refresh=refresh)
    assert source.history.call_count == 2
    other = Mock(cache_id="other-provider")
    other.history.return_value = raw
    load_history("Delhi", date(2025, 1, 1), date(2025, 1, 1), source=other)
    other.history.assert_called_once()
    source.close.assert_not_called()  # A supplied adapter remains caller-owned.


def test_current_cache_validates_values_preserves_raw_and_refreshes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    forecast = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]).tz_localize("Asia/Kolkata"),
                             "city": "Delhi", "tmax": [32.], "tmin": [22.], "tmean": [27.], "source": "TEST"})
    source = Mock(cache_id="test-current")
    source.current.return_value = ({"time": "2025-01-01T12:00", "temperature_2m": 200,
                                    "relative_humidity_2m": 50, "source": "TEST"}, forecast)
    current, processed = load_current("Delhi", source=source)
    assert pd.isna(current["temperature_2m"]) and current["relative_humidity_2m"] == 50
    assert str(current["time"].tzinfo) == "Asia/Kolkata"
    load_current("Delhi", source=source)
    assert source.current.call_count == 1
    load_current("Delhi", source=source, force_refresh=True)
    assert source.current.call_count == 2
    raw_files = list((tmp_path / "data/raw/current").glob("*.parquet"))
    current_table = next(pd.read_parquet(path) for path in raw_files if "temperature_2m" in pd.read_parquet(path))
    assert current_table.temperature_2m.iloc[0] == 200
    assert len(list((tmp_path / "data/processed/current").glob("*.parquet"))) == 2
