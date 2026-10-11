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


SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
           5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon"}


def monthly_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["month"] = result["date"].dt.strftime("%Y-%m")
    return result.groupby("month", sort=True).agg(
        mean_max=("tmax", "mean"), highest=("tmax", "max"), mean_min=("tmin", "mean"),
        valid_days=("tmax", "count")).reset_index()


def seasonal_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["season"] = result["date"].dt.month.map(SEASONS)
    grouped = result.groupby("season").agg(mean_max=("tmax", "mean"),
        mean_min=("tmin", "mean"), highest=("tmax", "max"), valid_days=("tmax", "count"))
    return grouped.reindex(["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]).dropna(
        subset=["valid_days"]).reset_index()
