"""Manual official IMD imports, geographic exploration and isolated cell ML."""
from datetime import date
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from climate.analytics import daily_trends, monthly_patterns, seasonal_patterns, summary
from climate.charts import evaluation_chart, finish, prediction_chart, trend_chart
from climate.evaluation import ModelUnavailable
from climate.export import csv_bytes
from climate.imd_charts import grid_heatmap
from climate.imd_format import HISTORICAL, HISTORICAL_URL, RECENT_URL
from climate.imd_parser import parse_grd
from climate.imd_store import IMDStore
from climate.locations import CITIES
from climate.model import train_and_predict
from climate.presentation import chart_story, processing_table, model_intro, evaluation_story
from climate.storytelling import SECTIONS, peak_observation, group_observation, prediction_observation, baseline_comparison, sources, imd_processing_summary

LOGGER = logging.getLogger(__name__)


@st.cache_resource(ttl=21600, max_entries=4, show_spinner=False)
def imd_model(frame: pd.DataFrame):
    return train_and_predict(frame)


def import_form(store: IMDStore):
    st.markdown(f"Official downloads: [yearly 1° Tmax]({HISTORICAL_URL}) · [recent daily 0.5° Tmax]({RECENT_URL})")
    st.caption("Automatic IMD download is unavailable in this environment: direct official-page requests timed out, so no download endpoint has been assumed. Download through the official page using its normal access procedure, then import below. No authentication or CAPTCHA is bypassed.")
    with st.expander("Import an official IMD GRD file", expanded=not bool(store.catalog())):
        product = st.selectbox("IMD product", ["Historical annual · 1° / 31×31", "Recent daily · 0.5° / 61×61"])
        kind = "historical" if product.startswith("Historical") else "recent"
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        with st.form("imd_import"):
            year, day = None, None
            if kind == "historical":
                year = int(st.number_input("File calendar year", min_value=1951, max_value=today.year - 1,
                                           value=today.year - 1, step=1))
            else:
                day = st.date_input("IMD product date", value=today, max_value=today)
            byte_order = st.selectbox("Declared file byte order", ["Little endian", "Big endian"])
            uploaded = st.file_uploader("Maximum-temperature binary (.GRD / .bin)", type=["grd", "bin"])
            control = st.file_uploader("Matching IMD .CTL (required for recent daily files)", type=["ctl"])
            st.caption("IMD's historical examples use native 4-byte floats without a byte-order declaration. Confirm byte order from your file's documentation; it is not auto-detected. A supplied CTL must match the file's dates, dimensions, missing value and layout.")
            confirmed = st.checkbox("I confirm this is IMD maximum-temperature data for the selected product/date, with longitude varying fastest and latitude increasing south to north.")
            submitted = st.form_submit_button("Validate and import IMD data", type="primary")
        if submitted:
            if not uploaded or not confirmed:
                st.error("Choose a GRD file and confirm its product, dates and documented orientation.")
                return
            if uploaded.size > 2_000_000:
                st.error("The selected IMD annual/daily product is smaller than 2 MB; this file has the wrong size.")
                return
            try:
                payload = uploaded.getvalue()
                control_text = control.getvalue().decode("utf-8-sig") if control else None
                parsed = parse_grd(payload, product=kind, byte_order="<" if byte_order.startswith("Little") else ">",
                                   year=year, day=day, control_text=control_text)
                identity = store.save(parsed, payload, uploaded.name)
            except (ValueError, OSError, UnicodeError) as exc:
                LOGGER.warning("IMD import rejected: %s", exc)
                st.error(f"IMD file was not accepted: {exc}")
            else:
                st.session_state["imd_dataset"] = identity
                st.session_state["imd_import_message"] = "IMD file validated and stored as a separate raw GRD and structured Parquet dataset."
                st.rerun()


def cell_panel(store: IMDStore, entries: list[dict]):
    annual = [entry for entry in entries if entry["grid"]["product"] == "historical"]
    if not annual:
        st.info("Import a historical annual 1° file for grid-cell analytics and model evaluation. Daily 0.5° files remain a separate map product.")
        return
    labels = {entry["id"]: f"{entry['start'][:4]} · {entry['original_filename']} · {entry['id'][:8]}" for entry in annual}
    selected = st.multiselect("Historical annual datasets (one version per year)", list(labels),
                              default=[annual[-1]["id"]], format_func=labels.get)
    city = CITIES[st.selectbox("City location reference", list(CITIES), index=1)]
    if not selected:
        st.info("Select at least one historical dataset.")
        return
    mode = st.radio("Grid-cell selection", ["Nearest available to location", "Exact grid coordinate"], horizontal=True)
    left, right = st.columns(2)
    if mode.startswith("Nearest"):
        query_lat = left.number_input("Location latitude (°N)", min_value=7.5, max_value=37.5,
                                      value=city.latitude, format="%.4f", key=f"imd_query_lat_{city.name}")
        query_lon = right.number_input("Location longitude (°E)", min_value=67.5, max_value=97.5,
                                       value=city.longitude, format="%.4f", key=f"imd_query_lon_{city.name}")
        try:
            nearest = store.nearest_historical_cell(selected, query_lat, query_lon)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        latitude, longitude = nearest.latitude, nearest.longitude
        st.caption(f"Nearest available 1° cell: {latitude:g}°N, {longitude:g}°E · {nearest.distance_km:.1f} km from the requested location · {nearest.valid_days} valid dates. Availability means at least one reading in the selected history; missing dates remain missing.")
    else:
        latitude = left.select_slider("Grid latitude (°N)", options=HISTORICAL.latitudes.tolist(),
                                      value=float(HISTORICAL.latitudes[np.argmin(abs(HISTORICAL.latitudes - city.latitude))]))
        longitude = right.select_slider("Grid longitude (°E)", options=HISTORICAL.longitudes.tolist(),
                                        value=float(HISTORICAL.longitudes[np.argmin(abs(HISTORICAL.longitudes - city.longitude))]))
    st.caption(f"Analyzing the actual 1° resolution cell at {latitude:g}°N, {longitude:g}°E. City names are location references; these are spatially interpolated gridded values, not measurements at city stations. No Open-Meteo data or recent 0.5° cells are appended.")
    try:
        frame = store.historical_cell(selected, latitude, longitude)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    if not frame.tmax.notna().any():
        st.info("This IMD grid cell has no valid historical temperatures. Select another grid coordinate; missing values are not replaced.")
        return
    threshold = st.number_input("Exploratory IMD cell threshold (°C)", value=40., min_value=20., max_value=60.)
    stats = summary(frame, threshold)
    columns = st.columns(3)
    columns[0].metric("Valid daily maxima", stats["valid_days"])
    columns[1].metric("Highest cell maximum", f"{stats['highest']:.1f} °C" if pd.notna(stats["highest"]) else "Unavailable")
    columns[2].metric("Exploratory exceedance days", stats["exceedance_days"])
    st.plotly_chart(trend_chart(daily_trends(frame, threshold), threshold), width="stretch")
    chart_story(frame, "Daily maximum temperature at this actual IMD grid cell, with a 7-day rolling average.",
                peak_observation(frame), "This is a 1° spatially interpolated grid estimate, not a city-station measurement. Missing dates are blank. Threshold markers are exploratory and are not official heatwave declarations.")
    with st.expander("Daily grid-cell maximum temperatures"):
        st.dataframe(frame[["date", "tmax", "latitude", "longitude", "source"]], hide_index=True, width="stretch")
    monthly, seasonal = monthly_patterns(frame), seasonal_patterns(frame)
    st.plotly_chart(finish(px.bar(monthly, x="month", y="mean_max"), "IMD grid-cell monthly mean maximum"), width="stretch")
    chart_story(frame, "Each monthly bar averages valid daily maximum temperatures at this grid coordinate.",
                group_observation(monthly, "month"), "A mean daily maximum describes daytime peaks, not the average across the full day. Compare valid-day coverage; partial months are not climate normals.")
    st.dataframe(seasonal[["season", "mean_max", "highest", "valid_days"]], hide_index=True, width="stretch")
    st.caption("Tmax-only files contain no daily minimum or daily mean temperatures. Thresholds are exploratory and are not official IMD heatwave warnings.")
    st.download_button("Download IMD cell series CSV", csv_bytes(frame), "imd_cell_series.csv", "text/csv")
    return frame


def cell_model_panel(frame):
    st.subheader("IMD historical model experiment")
    model_intro()
    if not st.checkbox("Evaluate Random Forest on this IMD historical grid cell", value=False):
        st.info("Enable evaluation to train on the selected historical cell. This is an optional historical experiment; no upcoming forecast is implied.")
        return
    st.warning(f"Historical endpoint experiment: model inputs end on {frame.date.max():%d %b %Y}. Predictions follow that date and are not a live city forecast. Evaluation targets are IMD interpolated grid cells, not independent station measurements.")
    try:
        with st.spinner("Training source-isolated IMD models and evaluating against persistence…"):
            result = imd_model(frame)
    except (ModelUnavailable, ValueError) as exc:
        st.info(f"IMD model unavailable: {exc}")
        return
    except Exception:
        LOGGER.exception("IMD model failed")
        st.error("IMD model training failed. No predictions were generated; inspect the application logs.")
        return
    figure = prediction_chart(frame, result.predictions)
    figure.data[0].name = "IMD grid-cell historical maximum"
    figure.update_layout(title="IMD historical grid-cell experiment · three horizons")
    st.plotly_chart(figure, width="stretch")
    shown = pd.concat([frame.tail(21)[["date"]], result.predictions[["date"]]], ignore_index=True)
    chart_story(shown, "The model estimates three dates immediately after this IMD history endpoint, separately from all Open-Meteo forecasts.",
                prediction_observation(result.predictions), "The solid line is the historical gridded target; dashed points are experimental predictions after its endpoint. Persistence repeats the latest maximum. These dates may already be in the past.", source=sources(frame))
    st.dataframe(result.predictions, hide_index=True, width="stretch")
    st.caption("Validation selects tree depth; chronological test results compare MAE/RMSE (°C) with persistence. Boundary-crossing labels are purged. Production refitting follows held-out scoring.")
    evaluation_story(result)
    evaluation = evaluation_chart(result.test_predictions)
    evaluation.data[0].name = "Held-out IMD gridded target"
    st.plotly_chart(evaluation, width="stretch")
    first = result.test_predictions.loc[result.test_predictions.horizon == 1]
    _, verdict = baseline_comparison(result.metrics.loc[result.metrics.horizon == 1])
    chart_story(first, "One-day-ahead model predictions are compared with IMD grid values held out from model selection.",
                verdict, "Only horizon one is drawn here; the table scores all three horizons. Agreement with interpolated grid targets does not independently validate accuracy at station locations.", source=sources(frame))
    with st.expander("IMD chronological evaluation partitions"):
        st.dataframe(result.splits, hide_index=True, width="stretch")


def render_imd():
    st.title("Climate Intelligence")
    st.caption("IMD · official station-derived temperature grids. Imported product dates are historical/static data, not a live observation stream. Open-Meteo remains a separate workspace.")
    store = IMDStore()
    message = st.session_state.pop("imd_import_message", None)
    if message:
        st.success(message)
    entries = store.catalog()
    if not entries:
        overview, historical, model, processing = st.tabs(SECTIONS)
        with overview:
            st.info("No IMD file has been imported. Maps and analytics appear only after a real GRD file passes validation.")
            st.write("Open Data Processing to import an official GRD file, or switch to Open-Meteo in the sidebar for city estimates.")
        with historical:
            st.info("Import an IMD dataset to explore dates and geographic temperature patterns.")
        with model:
            st.info("Historical model experiments require an imported annual 1° dataset and a cell with sufficient valid history.")
        with processing:
            import_form(store)
        return
    labels = {entry["id"]: f"{entry['grid']['product']} · {entry['start']} — {entry['end']} · {entry['original_filename']} · {entry['id'][:8]}" for entry in entries}
    identity = st.selectbox("Imported IMD dataset", list(labels), format_func=labels.get, key="imd_dataset")
    entry = next(item for item in entries if item["id"] == identity)
    overview, historical, model, processing = st.tabs(SECTIONS)
    with overview:
        st.subheader("What does this imported temperature dataset show?")
        st.caption(f"Source: IMD · {entry['original_filename']} · {entry['start']} – {entry['end']} · {entry['grid']['step']:g}° spatial resolution")
        columns = st.columns(3)
        columns[0].metric("Valid maximum-temperature readings", f"{entry['valid_cells']:,}")
        columns[1].metric("Missing readings", f"{entry['missing_cells']:,}")
        columns[2].metric("Highest available daily maximum", f"{entry['maximum_c']:.2f} °C")
        st.write(f"The imported file contains {entry['days']} daily grids. Valid daily maxima range from {entry['minimum_c']:.2f} to {entry['maximum_c']:.2f}°C across available cells.")
        st.caption("Missing cells do not mean zero temperature. IMD interpolates station information onto a regular grid; every coordinate is not an independent observation. These extremes describe the imported grid domain, not an area-weighted India average.")
    with historical:
        st.subheader("How do temperatures vary across the grid?")
        first, last = date.fromisoformat(entry["start"]), date.fromisoformat(entry["end"])
        st.caption(f"Imported product coverage: {first:%d %b %Y} through {last:%d %b %Y}. This imported file is not a live observation stream.")
        selected_date = st.date_input("Grid product date", value=first, min_value=first, max_value=last, key=f"imd_date_{identity}")
        try:
            daily = store.day(identity, selected_date)
            valid = daily.maximum_temperature_c.dropna()
            columns = st.columns(3)
            columns[0].metric("Available cells on selected date", f"{len(valid)} / {len(daily)}")
            columns[1].metric("Lowest available Tmax", f"{valid.min():.2f} °C" if len(valid) else "Unavailable")
            columns[2].metric("Highest available Tmax", f"{valid.max():.2f} °C" if len(valid) else "Unavailable")
            figure = grid_heatmap(daily, f"IMD {entry['grid']['step']:g}° Tmax · {selected_date:%d %b %Y}",
                                  entry["minimum_c"], entry["maximum_c"])
            st.plotly_chart(figure, width="stretch")
            chart_story(daily, "Colors show daily maximum temperature at each available grid coordinate for the selected product date.",
                        peak_observation(daily, "maximum_temperature_c"),
                        "Longitude runs west to east; latitude runs south to north. Blank cells have missing data. The color scale stays fixed to this dataset's range so dates can be compared. No extra interpolation or political boundary mask is applied; city dots are location references only.")
            st.download_button("Download selected IMD date CSV", csv_bytes(daily), f"imd_tmax_{selected_date}.csv", "text/csv")
            with st.expander("Daily, monthly and seasonal grid statistics"):
                st.caption("Unweighted summaries of available grid cell-days in this file, not an area-weighted India average. Minimum Tmax is the lowest daily maximum, not a nighttime minimum. Missing cells are excluded from temperatures and counted separately. Seasons are grouped within calendar year; December remains in its file year.")
                for name, table in store.statistics(identity).items():
                    st.markdown(f"**{name.title()}**")
                    st.dataframe(table, hide_index=True, width="stretch")
                    st.download_button(f"Download {name} IMD statistics CSV", csv_bytes(table),
                                       f"imd_{entry['start']}_{name}_statistics.csv", "text/csv")
        except (ValueError, OSError) as exc:
            st.warning(f"IMD date unavailable: {exc}")
        st.subheader("Explore a historical 1° grid cell")
        frame = cell_panel(store, entries)
    with model:
        if frame is not None:
            cell_model_panel(frame)
        else:
            st.info("Select a historical annual dataset and a grid cell with valid temperatures in Historical Analysis to evaluate the model.")
    with processing:
        st.subheader("Where did this data come from, and what changed?")
        processing_table(imd_processing_summary(entry))
        st.write("Processing: check binary size and calendar → validate declared grid and byte order → decode float32 values → replace IMD's missing sentinel with NaN → reject invalid files → map cells to date/latitude/longitude → preserve raw GRD and cache a separate Parquet table.")
        st.caption("Missing cell-days remain in the table; none are filled. The trend's rolling mean is a separate labeled visual summary. The source grid is already spatially interpolated by IMD; this app performs no additional spatial interpolation.")
        import_form(store)
        with st.expander("Detailed file metadata and validation"):
            st.json(entry)
        st.download_button("Download complete IMD dataset Parquet", store.parquet_path(identity).read_bytes(),
                           f"imd_{entry['grid']['product']}_{entry['start']}.parquet", "application/octet-stream")
        st.caption("Columns: date, latitude, longitude, maximum_temperature_c, data_source. Raw GRD bytes, SHA-256, declared byte order and import metadata are retained under data/imd/. These files are excluded from Git.")
    st.markdown(f"Source: India Meteorological Department, Climate Research & Services, Pune. [Historical documentation]({HISTORICAL_URL}) · [Recent data]({RECENT_URL}). Temperature dataset citation: Srivastava, Rajeevan & Kshirsagar (2009), Atmospheric Science Letters, [doi:10.1002/asl.232](https://doi.org/10.1002/asl.232).")
