"""Quality controls preserve missingness and never alter raw inputs."""
import numpy as np
import pandas as pd

TEMPERATURES = ["tmax", "tmin", "tmean"]


def normalize_time(values: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert(timezone)


def clean_daily(raw: pd.DataFrame, timezone: str = "Asia/Kolkata") -> tuple[pd.DataFrame, dict]:
    required = {"date", "city", *TEMPERATURES}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    frame = raw.copy(deep=True)
    frame["date"] = normalize_time(frame["date"], timezone).dt.normalize()
    bad_dates = int(frame["date"].isna().sum())
    frame = frame.dropna(subset=["date", "city"])
    duplicates = int(frame.duplicated(["city", "date"]).sum())
    frame = frame.drop_duplicates(["city", "date"], keep="last")
    invalid = 0
    for name in TEMPERATURES:
        values = pd.to_numeric(frame[name], errors="coerce")
        bad = values.notna() & (~np.isfinite(values) | ~values.between(-90, 65))
        invalid += int(bad.sum())
        frame[name] = values.mask(bad)
    inconsistent = frame["tmin"] > frame["tmax"]
    inconsistent |= (frame["tmean"] < frame["tmin"]) | (frame["tmean"] > frame["tmax"])
    invalid += int(inconsistent.sum())
    frame.loc[inconsistent, TEMPERATURES] = np.nan
    frame = frame.sort_values(["city", "date"]).reset_index(drop=True)
    report = {"raw_rows": len(raw), "processed_rows": len(frame), "duplicate_rows": duplicates,
              "invalid_dates": bad_dates, "invalid_values_or_rows": invalid,
              "missing_temperature_cells": int(frame[TEMPERATURES].isna().sum().sum())}
    return frame, report


def aggregate_hourly(raw: pd.DataFrame, timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    frame = raw.copy(deep=True)
    frame["time"] = normalize_time(frame["time"], timezone)
    frame = frame.dropna(subset=["time", "city"]).drop_duplicates(["city", "time"])
    values = pd.to_numeric(frame["temperature"], errors="coerce")
    frame["temperature"] = values.where(values.between(-90, 65))
    frame["date"] = frame["time"].dt.normalize()
    daily = frame.groupby(["city", "date"])["temperature"].agg(
        tmax="max", tmin="min", tmean="mean", valid_hours="count").reset_index()
    daily["partial_day"] = daily["valid_hours"] < 24
    return daily
