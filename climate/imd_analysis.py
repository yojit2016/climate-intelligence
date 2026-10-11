"""Coverage-aware gridded Tmax summaries and explicit geographic cell lookup."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from climate.analytics import SEASONS
from climate.imd_format import IMDFormatError


@dataclass(frozen=True)
class NearestCell:
    latitude: float
    longitude: float
    distance_km: float
    valid_days: int


def _single_grid(frame: pd.DataFrame) -> None:
    if frame.empty or frame.date.isna().any() or frame.date.nunique() != 1:
        raise IMDFormatError("Select exactly one nonempty IMD product date per grid.")
    if frame.duplicated(["latitude", "longitude"]).any() or frame.data_source.nunique() != 1:
        raise IMDFormatError("A daily grid must have unique coordinates and one source product.")


def hottest_cells(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Rank valid cells of one product date; ties use increasing coordinates."""
    _single_grid(frame)
    if limit < 1:
        raise ValueError("Ranking limit must be positive.")
    ranked = frame.dropna(subset=["maximum_temperature_c"]).sort_values(
        ["maximum_temperature_c", "latitude", "longitude"], ascending=[False, True, True]).head(limit)
    ranked = ranked[["latitude", "longitude", "maximum_temperature_c", "date", "data_source"]].reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def compare_dates(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Date B minus A at exact shared coordinates; missing on either date stays NaN."""
    _single_grid(first)
    _single_grid(second)
    keys = ["latitude", "longitude"]
    a_coords = pd.MultiIndex.from_frame(first[keys]).sort_values()
    b_coords = pd.MultiIndex.from_frame(second[keys]).sort_values()
    if not a_coords.equals(b_coords) or first.data_source.iloc[0] != second.data_source.iloc[0]:
        raise IMDFormatError("Date comparisons require the same grid coordinates and source product.")
    a = first[keys + ["date", "maximum_temperature_c"]].rename(
        columns={"date": "date_a", "maximum_temperature_c": "maximum_temperature_a_c"})
    b = second[keys + ["date", "maximum_temperature_c"]].rename(
        columns={"date": "date_b", "maximum_temperature_c": "maximum_temperature_b_c"})
    result = a.merge(b, on=keys, validate="one_to_one").sort_values(keys).reset_index(drop=True)
    result["difference_c"] = result.maximum_temperature_b_c - result.maximum_temperature_a_c
    result["data_source"] = first.data_source.iloc[0]
    return result


def difference_findings(frame: pd.DataFrame) -> str:
    valid = frame.dropna(subset=["difference_c"])
    if valid.empty:
        return "No grid cells have valid maximum temperatures on both dates; differences are unavailable."
    parts = [f"{len(valid):,} cells are valid on both dates (B minus A)."]
    for name, subset, largest in [("increase", valid.loc[valid.difference_c > 0], True),
                                  ("decrease", valid.loc[valid.difference_c < 0], False)]:
        if subset.empty:
            parts.append(f"No {name}s among jointly valid cells.")
        else:
            row = subset.loc[subset.difference_c.idxmax() if largest else subset.difference_c.idxmin()]
            parts.append(f"Largest {name}: {abs(row.difference_c):.2f}°C at {row.latitude:g}°N, {row.longitude:g}°E.")
    return " ".join(parts)


def nearest_available(frame: pd.DataFrame, latitude: float, longitude: float) -> NearestCell:
    """Nearest cell with any valid value in the supplied period, by great-circle distance.

    Availability does not imply complete coverage or a reading on every date.
    No data are interpolated, filled, or moved to the query coordinates.
    """
    if not np.isfinite([latitude, longitude]).all() or not (
            7.5 <= latitude <= 37.5 and 67.5 <= longitude <= 97.5):
        raise IMDFormatError("Location must lie within the documented IMD grid extent.")
    valid = frame.dropna(subset=["maximum_temperature_c"])
    cells = valid.groupby(["latitude", "longitude"]).date.nunique().reset_index(name="valid_days")
    if cells.empty:
        raise IMDFormatError("No available temperature grid cell in the selected period.")
    lat, lon = np.radians(latitude), np.radians(longitude)
    grid_lat, grid_lon = np.radians(cells.latitude), np.radians(cells.longitude)
    a = np.sin((grid_lat - lat) / 2) ** 2 + np.cos(lat) * np.cos(grid_lat) * np.sin((grid_lon - lon) / 2) ** 2
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    index = distances.idxmin()
    cell = cells.loc[index]
    return NearestCell(float(cell.latitude), float(cell.longitude), float(distances.loc[index]),
                       int(cell.valid_days))


def grid_statistics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Unweighted available cell-day summaries, not India area-average temperature.

    Missing cells remain missing; counts expose changing geographic coverage.
    Seasonal groups are within calendar year (December stays in that year).
    """
    frame = frame.copy()
    frame["day"] = frame.date.dt.strftime("%Y-%m-%d")
    frame["month"] = frame.date.dt.strftime("%Y-%m")
    frame["year"] = frame.date.dt.year
    frame["season"] = frame.date.dt.month.map(SEASONS)
    tables = {}
    for name, keys in {"daily": ["day"], "monthly": ["month"], "seasonal": ["year", "season"]}.items():
        grouped = frame.groupby(keys, sort=True).agg(
            sample_mean_tmax_c=("maximum_temperature_c", "mean"),
            minimum_tmax_c=("maximum_temperature_c", "min"),
            maximum_tmax_c=("maximum_temperature_c", "max"),
            valid_cell_days=("maximum_temperature_c", "count"),
            total_cell_days=("maximum_temperature_c", "size"))
        grouped["missing_cell_days"] = grouped.total_cell_days - grouped.valid_cell_days
        grouped["data_source"] = " | ".join(sorted(frame.data_source.unique()))
        tables[name] = grouped.reset_index()
    return tables
