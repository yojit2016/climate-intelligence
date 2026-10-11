"""Strict NumPy decoding of daily IMD Tmax records into a long table."""
import calendar
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import re

import numpy as np
import pandas as pd

from climate.imd_format import HISTORICAL, GridSpec, IMDFormatError, control_spec


@dataclass
class ParsedGrid:
    frame: pd.DataFrame
    metadata: dict


def validate_control_time(text: str, first: date, days: int) -> None:
    match = re.search(r"^\s*TDEF\s+(\d+)\s+LINEAR\s+(\S+)\s+(\S+)", text, re.I | re.M)
    if match is None:
        raise IMDFormatError("A matching CTL must declare TDEF with a daily LINEAR time axis.")
    count, origin, increment = match.groups()
    stamp = re.fullmatch(r"(?:\d{1,2}(?::\d{2})?Z)?(\d{1,2}[A-Z]{3}\d{4})", origin.upper())
    if not stamp or increment.upper() != "1DY":
        raise IMDFormatError("CTL must use one-day records and an explicit calendar origin.")
    try:
        declared = pd.to_datetime(stamp[1], format="%d%b%Y").date()
    except ValueError as exc:
        raise IMDFormatError("Invalid CTL time origin.") from exc
    if int(count) != days or declared != first:
        raise IMDFormatError("CTL TDEF dates/day count disagree with the selected file. Use the matching control file, not an unedited example.")


def parse_grd(payload: bytes, *, product: str, byte_order: str, year: int | None = None,
              day: date | None = None, control_text: str | None = None) -> ParsedGrid:
    if product not in ("historical", "recent") or byte_order not in ("<", ">"):
        raise IMDFormatError("Choose historical/recent and explicit little/big byte order.")
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    if product == "historical":
        if year is None or not 1951 <= year < today.year:
            raise IMDFormatError("Annual imports require a completed calendar year from 1951 onward.")
        first, days, spec = date(year, 1, 1), 366 if calendar.isleap(year) else 365, HISTORICAL
    else:
        if day is None or day > today:
            raise IMDFormatError("Daily imports require an explicit non-future IMD product date.")
        if not control_text:
            raise IMDFormatError("Recent 0.5° files require their matching official IMD CTL file.")
        first, days, spec = day, 1, None
    if control_text:
        spec = control_spec(control_text, product, byte_order)
        validate_control_time(control_text, first, days)
    assert isinstance(spec, GridSpec)
    # Longitude varies fastest, followed by south-to-north latitude, then days.
    expected = days * spec.size * spec.size * 4
    if len(payload) != expected:
        raise IMDFormatError(f"Expected exactly {expected:,} bytes ({days} days × {spec.size} × {spec.size} × 4); received {len(payload):,}. Truncated, sequential or wrong-resolution files are rejected.")
    values = np.frombuffer(payload, dtype=f"{byte_order}f4").reshape(days, spec.size, spec.size).astype(np.float32)
    missing = values == np.float32(spec.missing)
    valid = values[~missing]
    if not valid.size:
        raise IMDFormatError("The entire file is missing; there are no temperatures to analyze.")
    subnormal = (np.abs(valid) > 0) & (np.abs(valid) < np.finfo(np.float32).tiny)
    if not np.isfinite(valid).all() or subnormal.any() or (valid < -90).any() or (valid > 65).any():
        raise IMDFormatError("Invalid float values or temperatures outside −90…65 °C. Confirm byte order and product; no automatic byte swapping is performed.")
    if valid.min() == valid.max():
        raise IMDFormatError("Constant-value file failed the spatial/temporal variation sanity check.")
    values[missing] = np.nan
    lat, lon = spec.latitudes, spec.longitudes
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
            and lat[-1] == 37.5 and lon[-1] == 97.5):
        raise IMDFormatError("Grid coordinate axes/extent failed validation.")
    points = spec.size ** 2
    dates = pd.date_range(first, periods=days, freq="D", tz="Asia/Kolkata")
    frame = pd.DataFrame({"date": dates.repeat(points),
        "latitude": np.tile(np.repeat(lat, spec.size), days),
        "longitude": np.tile(lon, days * spec.size),
        "maximum_temperature_c": values.ravel(), "data_source": spec.source})
    metadata = {"schema": 1, "sha256": sha256(payload).hexdigest(), "grid": asdict(spec),
        "byte_order": "little" if byte_order == "<" else "big", "start": first.isoformat(),
        "end": dates[-1].date().isoformat(), "days": days, "bytes": expected,
        "valid_cells": int(valid.size), "missing_cells": int(missing.sum()),
        "minimum_c": float(valid.min()), "maximum_c": float(valid.max()),
        "orientation": "day, south-to-north latitude, west-to-east longitude (fastest)",
        "metadata_basis": "Official IMD grid contract and user-declared file date/byte order; headerless payload cannot independently verify these declarations"}
    return ParsedGrid(frame, metadata)
