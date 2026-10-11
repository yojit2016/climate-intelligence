"""Presentation-only facts calculated from displayed data and processing metadata."""
import numpy as np
import pandas as pd


SECTIONS = ["Overview", "Historical Analysis", "ML Forecasting", "Data Processing"]


def coverage(frame: pd.DataFrame) -> str:
    dates = frame["date"].dropna()
    return f"{dates.min():%d %b %Y} – {dates.max():%d %b %Y}" if len(dates) else "No available dates"


def sources(frame: pd.DataFrame) -> str:
    column = "data_source" if "data_source" in frame else "source"
    if column not in frame:
        return "Source unavailable"
    return " · ".join(str(value) for value in frame[column].dropna().unique()) or "Source unavailable"


def peak_observation(frame: pd.DataFrame, column: str = "tmax") -> str:
    valid = frame.loc[frame[column].notna() & frame.date.notna()]
    if valid.empty:
        return "No valid temperatures are available in this displayed period."
    row = valid.loc[valid[column].idxmax()]
    location = ""
    if "latitude" in row and "longitude" in row:
        location = f" at {row.latitude:g}°N, {row.longitude:g}°E"
    elif "city" in row:
        location = f" in {row.city}"
    return f"The highest displayed temperature was {row[column]:.2f}°C on {row.date:%d %b %Y}{location}. Based on {len(valid):,} available readings; missing readings are excluded."


def group_observation(table: pd.DataFrame, category: str, column: str = "mean_max") -> str:
    valid = table.dropna(subset=[column])
    if valid.empty:
        return "There are no valid values to summarize for these groups."
    row = valid.loc[valid[column].idxmax()]
    return f"{row[category]} had the highest displayed average daily maximum: {row[column]:.2f}°C. This summarizes available dates, not a long-term climate normal."


def prediction_observation(predictions: pd.DataFrame) -> str:
    valid = predictions.dropna(subset=["prediction"])
    if valid.empty:
        return "No model predictions are available."
    highest = valid.loc[valid.prediction.idxmax()]
    return f"Across these {len(valid)} prediction dates, the model estimates {valid.prediction.min():.2f}–{valid.prediction.max():.2f}°C, with the highest on {highest.date:%d %b %Y}. These are predictions, not measured temperatures."


def baseline_comparison(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compare both errors per held-out horizon, without averaging away failures."""
    test = metrics.loc[metrics.partition == "Test"]
    rows = []
    for horizon, group in test.groupby("horizon", sort=True):
        indexed = group.set_index("model")
        if not {"Random Forest", "Persistence"}.issubset(indexed.index):
            continue
        for metric in ("mae", "rmse"):
            rf, baseline = float(indexed.loc["Random Forest", metric]), float(indexed.loc["Persistence", metric])
            if not np.isfinite([rf, baseline]).all():
                continue
            verdict = "Random Forest lower" if rf < baseline else "Persistence lower" if rf > baseline else "Equal"
            rows.append({"Days ahead": int(horizon), "Error measure": metric.upper(),
                         "Random Forest (°C)": rf, "Persistence (°C)": baseline,
                         "Result": verdict})
    table = pd.DataFrame(rows)
    if table.empty:
        return table, "A complete held-out comparison with persistence is unavailable."
    statements = []
    for measure in ("MAE", "RMSE"):
        part = table.loc[table["Error measure"] == measure]
        wins = int(part.Result.eq("Random Forest lower").sum())
        ties = int(part.Result.eq("Equal").sum())
        statements.append(f"Random Forest had lower {measure} at {wins} of {len(part)} evaluated horizons ({ties} tied)")
    return table, "; ".join(statements) + ". Lower error on this test period does not guarantee better future forecasts."


def point_processing_summary(raw: pd.DataFrame, frame: pd.DataFrame, report: dict) -> dict:
    """Count distinct raw rows flagged by the existing QC rules, not mixed events.

    The original report's invalid_values_or_rows mixes value and row events;
    it must never be presented as a count of distinct invalid records.
    """
    values = raw[["tmax", "tmin", "tmean"]].apply(pd.to_numeric, errors="coerce")
    bad_values = raw[["tmax", "tmin", "tmean"]].notna() & (
        values.isna() | ~np.isfinite(values) | (values < -90) | (values > 65))
    inconsistent = (values.tmin > values.tmax) | (values.tmean < values.tmin) | (values.tmean > values.tmax)
    invalid = bad_values.any(axis=1) | inconsistent | pd.to_datetime(raw.date, errors="coerce").isna() | raw.city.isna()
    return {"Source dataset": sources(raw), "Data represents": "Modeled / reanalysis point estimates",
            "Records acquired (city-days)": int(report["raw_rows"]),
            "Raw records flagged invalid": int(invalid.sum()),
            "Duplicate records removed": int(report["duplicate_rows"]),
            "Processed calendar records": int(report["processed_rows"]),
            "Valid maximum-temperature records": int(frame.tmax.notna().sum()),
            "Missing maximum-temperature values": int(frame.tmax.isna().sum()),
            "Missing temperature values (max/min/mean)": int(report["missing_temperature_cells"]),
            "Date coverage": coverage(frame),
            "Interpolation by this application": "None; gaps remain missing",
            "Smoothing": "Optional chart-only filter; never applied to stored values or ML inputs"}


def imd_processing_summary(entry: dict) -> dict:
    total = int(entry["days"] * entry["grid"]["size"] ** 2)
    return {"Source dataset": f"IMD · {entry['original_filename']}",
            "Data represents": "Station-derived spatially interpolated grid estimates",
            "Records acquired (grid cell-days)": total,
            "Valid maximum-temperature records": int(entry["valid_cells"]),
            "Missing maximum-temperature values": int(entry["missing_cells"]),
            "Invalid records retained": "0 — imports with invalid values are rejected in full",
            "Date coverage": f"{entry['start']} – {entry['end']}",
            "Grid resolution": f"{entry['grid']['step']:g}°",
            "Interpolation by this application": "None; spatial interpolation belongs to the IMD source product",
            "Smoothing": "No signal smoothing; cell trend includes a labeled 7-day rolling mean"}
