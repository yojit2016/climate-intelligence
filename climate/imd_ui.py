"""Official IMD datasets, geographic exploration and isolated cell ML."""
from datetime import date
import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from climate.analytics import daily_trends, monthly_patterns, seasonal_patterns, summary
from climate.charts import evaluation_chart, finish, prediction_chart, trend_chart
from climate.evaluation import ModelUnavailable
from climate.export import csv_bytes
from climate.imd_charts import grid_heatmap, difference_heatmap
from climate.imd_analysis import hottest_cells, compare_dates, difference_findings
from climate.imd_format import HISTORICAL, HISTORICAL_URL, RECENT_URL
from climate.imd_parser import parse_grd
from climate.imd_store import IMDStore
from climate.locations import CITIES
from climate.model import train_and_predict
from climate.presentation import chart_story, processing_table, processing_overview, model_intro, evaluation_story
from climate.storytelling import SECTIONS, peak_observation, group_observation, prediction_observation, baseline_comparison, sources, imd_processing_summary, imd_before_after, forecast_table

LOGGER = logging.getLogger(__name__)


def _developer_imd_upload_enabled() -> bool:
    return os.environ.get("CLIMATE_ENABLE_IMD_UPLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}


@st.cache_resource(ttl=21600, max_entries=4, show_spinner=False)
def imd_model(frame: pd.DataFrame):
    return train_and_predict(frame)


def import_form(store: IMDStore):
    if not _developer_imd_upload_enabled():
        st.info("Developer-only IMD uploads are disabled in this deployment. Use the already-validated IMD catalog in the workspace instead.")
        return
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
    st.dataframe(forecast_table(frame, result.predictions), hide_index=True, width="stretch")
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


def date_comparison_panel(store, identity, entry, daily, selected_date, first, last):
    with st.expander("Compare two IMD dates"):
        if not st.checkbox("Enable date comparison", key=f"imd_compare_{identity}"):
            st.caption("Compare maximum temperatures on two product dates from this same dataset. No long-term climate trend is inferred.")
            return
        st.caption(f"Date A is the selected grid product date above: {selected_date:%d %b %Y}. Differences are Date B minus Date A.")
        second_date = st.date_input("Date B", value=min(selected_date + pd.Timedelta(days=1), last),
                                    min_value=first, max_value=last, key=f"imd_date_b_{identity}")
        try:
            second = store.day(identity, second_date)
            compared = compare_dates(daily, second)
            for column, frame, name, stamp in zip(st.columns(2), [daily, second], ["A", "B"], [selected_date, second_date]):
                with column:
                    if not frame.maximum_temperature_c.notna().any():
                        st.info(f"Date {name} has no available temperatures; its map is blank.")
                        continue
                    st.plotly_chart(grid_heatmap(frame, f"Date {name} · {stamp:%d %b %Y}",
                        entry["minimum_c"], entry["maximum_c"], height=500), width="stretch")
                    chart_story(frame, "Daily maximum at original IMD grid coordinates.",
                        peak_observation(frame, "maximum_temperature_c"),
                        "Both temperature maps use the same dataset-wide color limits. Grey/blank cells have no reading; no resampling or interpolation is applied.")
            finding = difference_findings(compared)
            if compared.difference_c.notna().any():
                st.plotly_chart(difference_heatmap(compared), width="stretch")
                chart_story(pd.DataFrame({"date": [pd.Timestamp(selected_date), pd.Timestamp(second_date)]}),
                    "Change in gridded daily maximum temperature: Date B minus Date A at cells valid on both dates.",
                    finding, "Red means a higher daily maximum on B; blue means lower; the scale is symmetric around zero. Grey/blank cells are unavailable on at least one date. A two-date change is not measured long-term climate change.", source=sources(daily))
            else:
                st.info(finding)
            st.download_button("Download IMD date differences CSV", csv_bytes(compared),
                f"imd_difference_{selected_date}_{second_date}.csv", "text/csv")
        except (ValueError, OSError) as exc:
            st.warning(f"Date comparison unavailable: {exc}")


def render_imd():
    st.title("Climate Intelligence")
    st.caption("IMD · official station-derived temperature grids. These annual datasets are automatically discovered and loaded for the selected year; no visitor uploads are required. Open-Meteo remains a separate workspace.")
    store = IMDStore()
    message = st.session_state.pop("imd_import_message", None)
    if message:
        st.success(message)
    entries = store.catalog()
    if not entries:
        overview, historical, model, processing = st.tabs(SECTIONS)
        with overview:
            st.warning("No IMD file has been imported. The public interface does not require any GRD uploads; the app keeps functioning in the Open-Meteo workspace, and developer-only uploads can be enabled with CLIMATE_ENABLE_IMD_UPLOAD=1.")
            st.write("The public interface does not require any GRD uploads. Switch to Open-Meteo in the sidebar for city estimates, or enable developer uploads for local validation workflows.")
        with historical:
            st.info("No IMD file has been imported. Historical temperature maps and cell analysis require a prevalidated annual MaxT dataset.")
        with model:
            st.info("Historical model experiments require a validated annual 1° dataset and a cell with sufficient valid history.")
        with processing:
            import_form(store)
        return
    labels = {entry["id"]: f"{entry['original_filename']} · {entry['start']} — {entry['end']} · {entry['grid']['step']:g}°" for entry in entries}
    identity = st.selectbox("Available IMD dataset", list(labels), format_func=labels.get, key="imd_dataset")
    entry = next(item for item in entries if item["id"] == identity)
    first, last = date.fromisoformat(entry["start"]), date.fromisoformat(entry["end"])
    selected_date = st.date_input("Grid product date", value=first, min_value=first, max_value=last, key=f"imd_date_{identity}")
    daily = None
    try:
        daily = store.day(identity, selected_date)
        rankings = hottest_cells(daily)
    except (ValueError, OSError) as exc:
        daily = None
        st.warning(f"IMD date unavailable: {exc}")
    overview, historical, model, processing = st.tabs(SECTIONS)
    with overview:
        st.subheader("What does this temperature dataset show?")
        st.caption(f"Source: IMD · {entry['original_filename']} · {entry['start']} – {entry['end']} · {entry['grid']['step']:g}° spatial resolution")
        st.caption(f"Selected product date: {selected_date:%d %b %Y} · IMD static annual dataset, not live observations")
        st.caption("historical/static")
        if daily is not None:
            valid = daily.maximum_temperature_c.dropna()
            columns = st.columns(3)
            columns[0].metric("Valid grid cells on selected date", f"{len(valid):,} / {len(daily):,}")
            columns[1].metric("Lowest grid-cell daily maximum", f"{valid.min():.2f} °C" if len(valid) else "Unavailable")
            columns[2].metric("Highest grid-cell daily maximum", f"{valid.max():.2f} °C" if len(valid) else "Unavailable")
            st.markdown("**Key Findings**")
            st.write(peak_observation(daily, "maximum_temperature_c"))
            if not rankings.empty:
                hottest = rankings.iloc[0]
                st.caption(f"Hottest valid grid coordinate: {hottest.latitude:g}°N, {hottest.longitude:g}°E. Coordinates describe a {entry['grid']['step']:g}° grid cell, not an exact city measurement; equal hottest values may occur at other cells.")
        else:
            st.info("Choose an available date to see grid-cell findings.")
        st.caption(f"Whole-file coverage: {entry['days']} daily grids · {entry['valid_cells']:,} valid cell-days · {entry['missing_cells']:,} missing cell-days.")
        st.caption("Missing cells do not mean zero temperature. IMD interpolates station information onto a regular grid; every coordinate is not an independent observation. These extremes describe the imported grid domain, not an area-weighted India average.")
    with historical:
        st.subheader("How do temperatures vary across the grid?")
        st.caption(f"Dataset coverage: {first:%d %b %Y} through {last:%d %b %Y}. This annual dataset is not a live observation stream.")
        try:
            if daily is None:
                raise ValueError("No grid table is available for the selected date.")
            figure = grid_heatmap(daily, f"IMD {entry['grid']['step']:g}° Tmax · {selected_date:%d %b %Y}",
                                  entry["minimum_c"], entry["maximum_c"])
            st.plotly_chart(figure, width="stretch")
            chart_story(daily, "Colors show daily maximum temperature at each available grid coordinate for the selected product date.",
                        peak_observation(daily, "maximum_temperature_c"),
                        "Longitude runs west to east; latitude runs south to north. Blank cells have missing data. The color scale stays fixed to this dataset's range so dates can be compared. No extra interpolation or political boundary mask is applied; city dots are location references only.")
            st.download_button("Download selected IMD date CSV", csv_bytes(daily), f"imd_tmax_{selected_date}.csv", "text/csv")
            st.markdown("**Hottest Locations · top 5 valid grid cells**")
            displayed = rankings.drop(columns="data_source").copy()
            displayed["date"] = displayed.date.dt.strftime("%d %b %Y")
            st.dataframe(displayed.rename(columns={"rank": "Rank", "latitude": "Latitude (°N)",
                "longitude": "Longitude (°E)", "maximum_temperature_c": "Maximum (°C)", "date": "Product date"}),
                hide_index=True, width="stretch")
            st.caption(f"Original {entry['grid']['step']:g}° grid cells, not station or exact city readings. Grey/blank map cells are missing; ties are ordered by latitude then longitude.")
            date_comparison_panel(store, identity, entry, daily, selected_date, first, last)
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
        details = imd_processing_summary(entry)
        processing_overview(details, imd_before_after(entry))
        st.write("Processing: validate the annual IMD product → retain source metadata and grid bounds → preserve the geographic coordinate grid → store a parsed Parquet table for web access. No visitor uploads are required for this prevalidated deployment catalogue.")
        st.caption("Missing cells can reflect areas without coverage inside the rectangular geographic domain. They remain missing, not zero. IMD's source grid is spatially interpolated; this app adds no spatial interpolation.")
        import_form(store)
        with st.expander("Advanced Details · file metadata and validation"):
            processing_table(details)
            st.json(entry)
        st.download_button("Download complete IMD dataset Parquet", store.parquet_path(identity).read_bytes(),
                           f"imd_{entry['grid']['product']}_{entry['start']}.parquet", "application/octet-stream")
        st.caption("Columns: date, latitude, longitude, maximum_temperature_c, data_source. Raw GRD bytes, SHA-256, declared byte order and import metadata are retained under data/imd/. These files are excluded from Git.")
    st.markdown(f"Source: India Meteorological Department, Climate Research & Services, Pune. [Historical documentation]({HISTORICAL_URL}) · [Recent data]({RECENT_URL}). Temperature dataset citation: Srivastava, Rajeevan & Kshirsagar (2009), Atmospheric Science Letters, [doi:10.1002/asl.232](https://doi.org/10.1002/asl.232).")
