"""Orchestrate cached ingestion while preserving raw and processed data."""
from datetime import date

import pandas as pd

from climate.cache import ParquetCache
from climate.locations import CITIES
from climate.processing import clean_daily, normalize_time
from climate.sources import OpenMeteoSource, WeatherSource

HISTORY_TTL = 21600
CURRENT_TTL = 900


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
        raw = None if force_refresh else raw_cache.read(key, ttl=HISTORY_TTL)
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


def load_current(city_name: str, *, force_refresh: bool = False,
                 source: WeatherSource | None = None) -> tuple[dict, pd.DataFrame]:
    """Cache current conditions and provider forecasts, then validate copies."""
    city = CITIES[city_name]
    provider = source or OpenMeteoSource()
    cache = ParquetCache("data/raw/current")
    key = {"schema": 1, "provider": provider.cache_id, "city": city_name}
    current_key, forecast_key = {**key, "kind": "current"}, {**key, "kind": "forecast"}
    try:
        current_raw = None if force_refresh else cache.read(current_key, ttl=CURRENT_TTL)
        forecast_raw = None if force_refresh else cache.read(forecast_key, ttl=CURRENT_TTL)
        if current_raw is None or forecast_raw is None:
            values, forecast_raw = provider.current(city)
            current_raw = pd.DataFrame([values])
            current_raw["retrieved_at"] = pd.Timestamp.now(tz="UTC")
            cache.write(current_key, current_raw)
            cache.write(forecast_key, forecast_raw)
    finally:
        if source is None:
            provider.close()
    current = current_raw.copy(deep=True)
    for field, lower, upper in (("temperature_2m", -90, 65), ("relative_humidity_2m", 0, 100)):
        values = pd.to_numeric(current.get(field, pd.Series(index=current.index, dtype=float)), errors="coerce")
        current[field] = values.where(values.between(lower, upper))
    current["time"] = normalize_time(current.get("time", pd.Series(index=current.index, dtype=str)), city.timezone)
    forecast, _ = clean_daily(forecast_raw, city.timezone)
    processed_cache = ParquetCache("data/processed/current")
    processed_cache.write(current_key, current)
    processed_cache.write(forecast_key, forecast)
    return current.iloc[0].to_dict(), forecast
