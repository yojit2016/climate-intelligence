"""Streamlit layout and cached data entry points."""
from datetime import timedelta
import logging
import math

import pandas as pd
import streamlit as st

from climate.analytics import summary
from climate.api import WeatherClient, WeatherError
from climate.forecast_panel import forecast_panel
from climate.export import csv_bytes
from climate.locations import CITIES
from climate.panels import patterns_panel, regional_panel, trends_panel
from climate.service import load_history
from climate.weather import current_and_forecast, local_today

LOGGER = logging.getLogger(__name__)
cached_history = st.cache_data(ttl=21600, max_entries=24, show_spinner=False)(load_history)


@st.cache_data(ttl=900, max_entries=6, show_spinner=False)
def live_weather(city_name: str):
    client = WeatherClient()
    try:
        return current_and_forecast(client, CITIES[city_name])
    finally:
        client.close()


def display_value(value, suffix: str = " °C") -> str:
    try:
        number = float(value)
        return f"{number:.1f}{suffix}" if math.isfinite(number) else "Unavailable"
    except (ValueError, TypeError):
        return "Unavailable"


def filters():
    today = local_today(CITIES["Delhi"])
    with st.sidebar:
        st.markdown("### Climate Intelligence")
        st.caption("INDIA / TEMPERATURE MONITORING")
        with st.form("filters"):
            city = st.selectbox("Primary city", list(CITIES), index=1)
            dates = st.date_input("Historical date range", value=(today - timedelta(days=365), today - timedelta(days=1)),
                                  min_value=pd.Timestamp("1940-01-01").date(), max_value=today - timedelta(days=1))
            threshold = st.number_input("Exploratory threshold (°C)", min_value=20., max_value=60., value=40., step=.5)
            smoothing = st.checkbox("Show signal smoothing", value=True)
            window = st.select_slider("Smoothing window (days)", options=[5, 7, 11, 15], value=7)
            comparison = st.multiselect("Comparison cities", list(CITIES), default=["Mumbai", "Pune"])
            st.form_submit_button("Apply filters", type="primary", use_container_width=True)
        st.caption("Thresholds support exploration and are not official heatwave warnings.")
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info("Select both the beginning and end of the historical range.")
        st.stop()
    start, end = dates
    if (end - start).days > 3653:
        st.error("Select at most ten years per request to keep downloads and training responsive.")
        st.stop()
    return city, start, end, threshold, smoothing, window, comparison


def current_cards(city: str):
    try:
        current, forecast = live_weather(city)
        left, right = st.columns(2)
        left.metric("Current temperature · model estimate", display_value(current.get("temperature_2m")))
        right.metric("Current relative humidity · model estimate", display_value(current.get("relative_humidity_2m"), " %"))
        st.caption(f"Open-Meteo current estimate at {current.get('time', 'unavailable')} · Asia/Kolkata. No station measurements are ingested.")
        return forecast
    except WeatherError as exc:
        st.warning(f"Current conditions unavailable: {exc}")
        return None


def render():
    st.set_page_config(page_title="Climate Intelligence", layout="wide")
    city, start, end, threshold, smoothing, window, comparison = filters()
    st.title("Heatwave analytics")
    st.caption(f"{city} · {start:%d %b %Y} — {end:%d %b %Y} · All temperatures in °C")
    st.info("Historical values are reanalysis/model estimates. Recent completed days use forecast-archive model estimates. Exploratory thresholds are not official warnings.")
    provider_forecast = current_cards(city)
    try:
        with st.spinner("Loading historical weather…"):
            raw, frame, report = cached_history(city, start, end)
    except (WeatherError, ValueError, OSError) as exc:
        LOGGER.exception("History unavailable")
        st.error(f"Historical data unavailable: {exc}")
        st.stop()
    stats = summary(frame, threshold)
    columns = st.columns(4)
    for column, label, value in zip(columns,
            ["Mean daily maximum", "Highest maximum", "Lowest minimum", "Threshold exceedance days"],
            [display_value(stats["mean_max"]), display_value(stats["highest"]),
             display_value(stats["lowest"]), str(stats["exceedance_days"])]):
        column.metric(label, value)
    st.caption(f"{stats['valid_days']} valid daily maxima · {stats['missing_days']} missing daily maxima")
    with st.expander("Data quality and provenance"):
        st.json(report)
        st.dataframe(frame.groupby("source").size().rename("Days"), use_container_width=True)
        st.caption("Raw API tables and processed tables are cached independently under data/. Requests expire after 6 hours; current estimates expire after 15 minutes.")
    trends, patterns, regions, prediction, data = st.tabs(["Temperature trends", "Monthly & seasonal", "Regional comparison", "ML predictions", "Data"])
    with trends:
        trends_panel(frame, threshold, smoothing, window)
    with patterns:
        patterns_panel(frame)
    with regions:
        regional_panel(frame, comparison, start, end, threshold, cached_history)
    with prediction:
        forecast_panel(frame, provider_forecast)
    with data:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        filename = f"{city.lower()}_{start}_{end}"
        left.download_button("Download processed CSV", csv_bytes(frame), f"{filename}_processed.csv", "text/csv")
        right.download_button("Download raw API table CSV", csv_bytes(raw), f"{filename}_raw.csv", "text/csv")
        st.caption("Exports retain source labels and timezone offsets. Raw tables preserve API nulls and precede quality controls; missing processed values remain blank.")
    st.caption("Weather data: [Open-Meteo](https://open-meteo.com/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · Analytics and forecasts are exploratory.")
