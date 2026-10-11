"""Real daily reanalysis and recent model estimates from Open-Meteo."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from climate.api import WeatherClient, WeatherError
from climate.locations import City

DAILY = "temperature_2m_max,temperature_2m_min,temperature_2m_mean"
RENAME = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
          "temperature_2m_mean": "tmean"}


def local_today(city: City) -> date:
    return datetime.now(ZoneInfo(city.timezone)).date()


def parameters(city: City) -> dict:
    return {"latitude": city.latitude, "longitude": city.longitude,
            "timezone": city.timezone, "temperature_unit": "celsius"}


def parse_daily(payload: dict, city: City, source: str) -> pd.DataFrame:
    try:
        values = payload["daily"]
        frame = pd.DataFrame({"date": values["time"], **{
            name: values[field] for field, name in RENAME.items()}})
        if frame.empty:
            raise ValueError("Empty daily series")
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(city.timezone)
        frame["city"] = city.name
        frame["source"] = source
        frame["retrieved_at"] = pd.Timestamp.now(tz="UTC")
        return frame
    except (KeyError, ValueError, TypeError) as exc:
        raise WeatherError("Weather service returned an incomplete daily series.") from exc


def historical(client: WeatherClient, city: City, start: date, end: date) -> pd.DataFrame:
    today = local_today(city)
    if start > end or start < date(1940, 1, 1) or end >= today:
        raise ValueError("Choose an ordered historical range from 1940 through yesterday.")
    # The archive can lag several days. Recent completed days are explicitly
    # sourced from the forecast endpoint's past-days model output.
    cutoff = today - timedelta(days=5)
    parts = []
    if start <= cutoff:
        payload = client.get("https://archive-api.open-meteo.com/v1/archive", {
            **parameters(city), "start_date": start.isoformat(),
            "end_date": min(end, cutoff).isoformat(), "daily": DAILY})
        parts.append(parse_daily(payload, city, "Historical reanalysis / model estimate"))
    if end > cutoff:
        payload = client.get("https://api.open-meteo.com/v1/forecast", {
            **parameters(city), "past_days": 5, "forecast_days": 1, "daily": DAILY})
        recent = parse_daily(payload, city, "Recent model estimate (forecast archive)")
        dates = recent["date"].dt.date
        parts.append(recent.loc[(dates >= max(start, cutoff + timedelta(days=1))) & (dates <= end)])
    frame = pd.concat(parts, ignore_index=True)
    if frame.empty:
        raise WeatherError("No historical data available for this range.")
    return frame


def current_and_forecast(client: WeatherClient, city: City) -> tuple[dict, pd.DataFrame]:
    payload = client.get("https://api.open-meteo.com/v1/forecast", {
        **parameters(city), "current": "temperature_2m,relative_humidity_2m",
        "daily": DAILY, "forecast_days": 4})
    current = payload.get("current", {})
    if not isinstance(current, dict):
        raise WeatherError("Weather service returned invalid current conditions.")
    # These are model estimates, not station measurements.
    return current, parse_daily(payload, city, "Open-Meteo forecast")
