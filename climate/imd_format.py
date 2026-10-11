"""IMD Tmax grid contracts, from the official historical page / uploaded CTL.

Headerless GRD files cannot prove their year, byte order or orientation. These
must be declared from the source; shape/values and CTL consistency are checked.
"""
from dataclasses import dataclass

import numpy as np

HISTORICAL_URL = "https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html"
RECENT_URL = "https://www.imdpune.gov.in/cmpg/Realtimedata/max/Max_Download.html"


class IMDFormatError(ValueError):
    """File does not satisfy the selected documented IMD grid contract."""


@dataclass(frozen=True)
class GridSpec:
    product: str
    size: int
    step: float
    longitude_start: float = 67.5
    latitude_start: float = 7.5
    missing: float = 99.9

    @property
    def latitudes(self) -> np.ndarray:
        return self.latitude_start + np.arange(self.size) * self.step

    @property
    def longitudes(self) -> np.ndarray:
        return self.longitude_start + np.arange(self.size) * self.step

    @property
    def source(self) -> str:
        return f"IMD {self.product} {self.step:g} degree station-derived gridded maximum temperature"


HISTORICAL = GridSpec("historical", 31, 1.)


def control_spec(text: str, product: str, byte_order: str) -> GridSpec:
    """Read geometry/UNDEF only; never follow DSET paths or execute CTL text."""
    if product not in ("historical", "recent") or byte_order not in ("<", ">"):
        raise IMDFormatError("Choose a supported product and explicit little/big byte order.")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("*")]
    fields = {}
    for line in lines:
        words = line.split()
        key = words[0].upper()
        if key in fields:
            raise IMDFormatError(f"Duplicate CTL directive: {key}")
        fields[key] = words[1:]
    try:
        x, y = fields["XDEF"], fields["YDEF"]
        if len(x) != 4 or len(y) != 4 or x[1].upper() != "LINEAR" or y[1].upper() != "LINEAR":
            raise IMDFormatError("Only documented LINEAR XDEF/YDEF grids are supported.")
        nx, ny = int(x[0]), int(y[0])
        lon, dx, lat, dy = float(x[2]), float(x[3]), float(y[2]), float(y[3])
        missing = float(fields["UNDEF"][0])
    except (KeyError, IndexError, ValueError) as exc:
        raise IMDFormatError("CTL must define numeric LINEAR XDEF, YDEF and UNDEF.") from exc
    expected_size, expected_step = (31, 1.) if product == "historical" else (61, .5)
    if (nx, ny) != (expected_size, expected_size) or (dx, dy) != (expected_step, expected_step):
        raise IMDFormatError("CTL dimensions/resolution do not match the selected IMD product.")
    if not np.isfinite([lon, lat, missing]).all() or (lon, lat) != (67.5, 7.5):
        raise IMDFormatError("Expected the India grid beginning at 67.5°E, 7.5°N.")
    if product == "historical" and missing != 99.9:
        raise IMDFormatError("Historical IMD Tmax uses UNDEF 99.9.")
    if -90 <= missing <= 65:
        raise IMDFormatError("UNDEF overlaps the accepted temperature range.")
    options = {value.lower() for value in fields.get("OPTIONS", [])}
    if options - {"little_endian", "big_endian"}:
        raise IMDFormatError("Reversed axes, sequential records, templates and other CTL layouts are unsupported.")
    if {"little_endian", "big_endian"} <= options:
        raise IMDFormatError("CTL specifies conflicting byte orders.")
    declared_order = "<" if "little_endian" in options else ">" if "big_endian" in options else None
    if declared_order and declared_order != byte_order:
        raise IMDFormatError("Selected byte order disagrees with the CTL OPTIONS.")
    if any(key in fields for key in ("FILEHEADER", "THEADER", "XYHEADER", "HEADERBYTES", "DTYPE")):
        raise IMDFormatError("Only headerless 4-byte float binary files are supported.")
    if "ZDEF" in fields and (not fields["ZDEF"] or fields["ZDEF"][0] != "1"):
        raise IMDFormatError("Only a single vertical level is supported.")
    if "VARS" in fields and fields["VARS"] != ["1"]:
        raise IMDFormatError("Only single-variable maximum-temperature files are supported.")
    return GridSpec(product, nx, dx, lon, lat, missing)
