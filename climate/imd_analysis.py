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
