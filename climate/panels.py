"""Dashboard panels for descriptive analytics and regional comparisons."""
import pandas as pd
import streamlit as st

from climate.analytics import daily_trends, monthly_patterns, regional_summary, seasonal_patterns
from climate.api import WeatherError
from climate.charts import comparison_chart, pattern_chart, range_chart, smoothing_chart, trend_chart
from climate.smoothing import smooth_temperature
from climate.presentation import chart_story
from climate.storytelling import peak_observation, group_observation


def trends_panel(frame: pd.DataFrame, threshold: float, smoothing: bool, window: int):
    st.plotly_chart(trend_chart(daily_trends(frame, threshold), threshold), width="stretch")
    chart_story(frame, "Daily maximum temperature shows how hot each day became. The 7-day average makes short-term patterns easier to see.",
                peak_observation(frame), "Read dates left to right and temperature on the vertical axis. Gaps mean missing data. Highlighted days exceed your exploratory threshold; they are not official IMD heatwave declarations.")
    st.plotly_chart(range_chart(frame), width="stretch")
    chart_story(frame, "Compare each day's maximum, minimum and mean temperature.", peak_observation(frame),
                "Maximum is the day's highest temperature, minimum its lowest, and mean its daily average. These are modeled/reanalysis estimates, not readings from a local station.")
    if smoothing:
        filtered = frame.copy()
        filtered["smoothed"] = smooth_temperature(filtered, window)
        st.plotly_chart(smoothing_chart(filtered), width="stretch")
        chart_story(filtered, f"A {window}-day filter reduces short-lived variation for visual exploration only.",
                    peak_observation(filtered, "smoothed", "smoothed daily maximum"),
                    "The lighter line preserves the original maximum. The darker line is a centered Savitzky–Golay filter (polynomial order 2), which uses neighboring past and future values. It never enters the forecast model; gaps and short runs remain blank.")


def patterns_panel(frame: pd.DataFrame):
    monthly, seasonal = monthly_patterns(frame), seasonal_patterns(frame)
    st.plotly_chart(pattern_chart(monthly, "month", "Monthly temperature patterns"), width="stretch")
    chart_story(frame, "Each bar averages the available daily maxima or minima within a month.",
                group_observation(monthly, "month"), "Compare the height of monthly bars, but check valid-day counts before treating differences as meaningful. Partial months and missing days can affect the comparison.")
    st.plotly_chart(pattern_chart(seasonal, "season", "Seasonal temperature patterns"), width="stretch")
    chart_story(frame, "Seasonal bars group the selected dates into four Indian weather seasons.",
                group_observation(seasonal, "season"), "Winter: December–February; pre-monsoon: March–May; monsoon: June–September; post-monsoon: October–November. These are summaries of this selection, not climate normals or evidence of a long-term trend.")
    st.caption("Winter: Dec–Feb · Pre-monsoon: Mar–May · Monsoon: Jun–Sep · Post-monsoon: Oct–Nov. Averages cover only selected dates; partial seasons are not climatological normals.")
    st.dataframe(seasonal, hide_index=True, width="stretch")


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
    st.plotly_chart(comparison_chart(combined), width="stretch")
    chart_story(combined, "Compare daily maximum estimates at the selected city locations over the same requested dates.",
                peak_observation(combined), "Each line represents a city. A shared request range does not guarantee equal valid-day coverage; gaps are preserved. These point estimates do not describe temperatures everywhere between cities.")
    st.dataframe(regional_summary(combined, threshold), hide_index=True, width="stretch")
    st.caption("Each city uses the same requested dates. Valid-day counts reveal differences in available data; means use available values.")
