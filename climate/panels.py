"""Dashboard panels for descriptive analytics and regional comparisons."""
import pandas as pd
import streamlit as st

from climate.analytics import daily_trends, monthly_patterns, regional_summary, seasonal_patterns
from climate.api import WeatherError
from climate.charts import comparison_chart, pattern_chart, range_chart, smoothing_chart, trend_chart
from climate.smoothing import smooth_temperature


def trends_panel(frame: pd.DataFrame, threshold: float, smoothing: bool, window: int):
    st.plotly_chart(trend_chart(daily_trends(frame, threshold), threshold), use_container_width=True)
    st.plotly_chart(range_chart(frame), use_container_width=True)
    if smoothing:
        filtered = frame.copy()
        filtered["smoothed"] = smooth_temperature(filtered, window)
        st.plotly_chart(smoothing_chart(filtered), use_container_width=True)
        st.caption(f"Centred Savitzky–Golay filter, {window}-day window, polynomial order 2. Descriptive only; not used for ML. Gaps and short runs remain blank.")


def patterns_panel(frame: pd.DataFrame):
    monthly, seasonal = monthly_patterns(frame), seasonal_patterns(frame)
    st.plotly_chart(pattern_chart(monthly, "month", "Monthly temperature patterns"), use_container_width=True)
    st.plotly_chart(pattern_chart(seasonal, "season", "Seasonal temperature patterns"), use_container_width=True)
    st.caption("Winter: Dec–Feb · Pre-monsoon: Mar–May · Monsoon: Jun–Sep · Post-monsoon: Oct–Nov. Averages cover only selected dates; partial seasons are not climatological normals.")
    st.dataframe(seasonal, hide_index=True, use_container_width=True)


def regional_panel(frame: pd.DataFrame, cities: list[str], start, end, threshold: float, loader):
    primary = frame.city.iloc[0]
    requested = [name for name in cities if name != primary]
    if not requested:
        st.info("Choose comparison cities in the sidebar and apply filters.")
        return
    parts = [frame]
    with st.spinner("Loading comparison cities…"):
        for name in requested:
            try:
                _, other, _ = loader(name, start, end)
                parts.append(other)
            except (WeatherError, ValueError, OSError) as exc:
                st.warning(f"{name} unavailable: {exc}")
    combined = pd.concat(parts, ignore_index=True)
    st.plotly_chart(comparison_chart(combined), use_container_width=True)
    st.dataframe(regional_summary(combined, threshold), hide_index=True, use_container_width=True)
    st.caption("Each city uses the same requested dates. Valid-day counts reveal differences in available data; means use available values.")
