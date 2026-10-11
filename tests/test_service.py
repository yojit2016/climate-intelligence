from datetime import date
from unittest.mock import Mock

import pandas as pd

from climate.service import load_history


def test_ingestion_preserves_raw_and_materializes_absent_calendar_days(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-03"]).tz_localize("Asia/Kolkata"),
                        "city": "Delhi", "tmax": [30., 32.], "tmin": [20., 22.], "tmean": [25., 27.],
                        "source": "OFFLINE TEST FIXTURE", "retrieved_at": pd.Timestamp.now(tz="UTC")})
    upstream = Mock(return_value=raw)
    monkeypatch.setattr("climate.service.historical", upstream)
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
