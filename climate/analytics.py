"""Descriptive temperature analytics; thresholds are exploratory."""
import pandas as pd


def summary(frame: pd.DataFrame, threshold: float) -> dict:
    valid = frame["tmax"].dropna()
    return {"mean_max": valid.mean(), "highest": valid.max(), "lowest": frame["tmin"].min(),
            "exceedance_days": int(valid.gt(threshold).sum()), "valid_days": len(valid),
            "missing_days": len(frame) - len(valid)}


def daily_trends(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.sort_values("date").copy()
    # Time-based windows do not mistake a missing day for a calendar day.
    result["rolling_7d"] = result.set_index("date")["tmax"].rolling("7D", min_periods=4).mean().to_numpy()
    result["exceeds_threshold"] = result["tmax"].gt(threshold)
    return result


def regional_summary(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.DataFrame([{"city": city, **summary(group, threshold)}
                         for city, group in frame.groupby("city")])
