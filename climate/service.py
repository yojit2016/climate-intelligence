"""Orchestrate cached ingestion while preserving raw and processed data."""
from datetime import date

import pandas as pd

from climate.cache import ParquetCache
from climate.locations import CITIES
from climate.processing import clean_daily
from climate.sources import OpenMeteoSource, WeatherSource


def load_history(city_name: str, start: date, end: date, *, force_refresh: bool = False,
                 source: WeatherSource | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    city = CITIES[city_name]
    if start > end:
        raise ValueError("The start date must not follow the end date.")
    provider = source or OpenMeteoSource()
    key = {"schema": 2, "provider": provider.cache_id, "city": city_name,
           "start": start.isoformat(), "end": end.isoformat()}
    raw_cache = ParquetCache("data/raw")
    try:
        raw = None if force_refresh else raw_cache.read(key, ttl=21600)
        if raw is None:
            raw = provider.history(city, start, end)
            raw_cache.write(key, raw)
    finally:
        if source is None:
            provider.close()
    processed, report = clean_daily(raw, city.timezone)
    # Explicit missing calendar rows make gaps visible in charts and exports.
    expected = pd.date_range(start, end, tz=city.timezone, freq="D")
    processed = processed.set_index("date").reindex(expected).rename_axis("date").reset_index()
    report["missing_calendar_days"] = int(processed["city"].isna().sum())
    processed["city"] = city_name
    processed["source"] = processed["source"].fillna("Missing upstream record")
    report["missing_temperature_cells"] = int(processed[["tmax", "tmin", "tmean"]].isna().sum().sum())
    report["processed_rows"] = len(processed)
    ParquetCache("data/processed").write(key, processed)
    return raw, processed, report
