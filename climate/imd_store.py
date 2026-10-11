"""Content-addressed IMD raw files, structured Parquet and import metadata."""
from datetime import date
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

from climate.imd_format import IMDFormatError
from climate.imd_analysis import grid_statistics, nearest_available
from climate.imd_parser import ParsedGrid

LOGGER = logging.getLogger(__name__)


class IMDStore:
    def __init__(self, root: str | Path = "data/imd"):
        self.root = Path(root)

    def _path(self, identity: str, folder: str, suffix: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise IMDFormatError("Invalid dataset identifier.")
        return self.root / folder / f"{identity}{suffix}"

    @staticmethod
    def _atomic(path: Path, writer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            writer(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, parsed: ParsedGrid, payload: bytes, filename: str) -> str:
        metadata = dict(parsed.metadata)
        if sha256(payload).hexdigest() != metadata["sha256"]:
            raise IMDFormatError("Raw bytes disagree with the validated payload hash.")
        identity = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        metadata.update({"id": identity, "original_filename": Path(filename).name,
                         "imported_at": pd.Timestamp.now(tz="UTC").isoformat()})
        self._atomic(self._path(metadata["sha256"], "raw", ".grd"), lambda path: path.write_bytes(payload))
        self._atomic(self.parquet_path(identity), lambda path: parsed.frame.to_parquet(path, index=False))
        # Metadata is the final commit marker; incomplete writes are not listed.
        self._atomic(self._path(identity, "metadata", ".json"),
                     lambda path: path.write_text(json.dumps(metadata, indent=2), encoding="utf-8"))
        return identity

    def parquet_path(self, identity: str) -> Path:
        return self._path(identity, "processed", ".parquet")

    def catalog(self) -> list[dict]:
        entries = []
        for path in (self.root / "metadata").glob("*.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if metadata["schema"] == 1 and self.parquet_path(metadata["id"]).is_file():
                    entries.append(metadata)
            except (ValueError, KeyError, OSError) as exc:
                LOGGER.warning("Ignoring unusable IMD metadata %s: %s", path, exc)
        return sorted(entries, key=lambda item: (item["grid"]["product"], item["start"], item["id"]))

    def day(self, identity: str, selected: date) -> pd.DataFrame:
        return pd.read_parquet(self.parquet_path(identity), filters=[
            ("date", "==", pd.Timestamp(selected, tz="Asia/Kolkata"))])

    def historical_cell(self, identities: list[str], latitude: float, longitude: float) -> pd.DataFrame:
        catalog = {entry["id"]: entry for entry in self.catalog()}
        if not identities or len(identities) != len(set(identities)):
            raise IMDFormatError("Select distinct annual historical datasets for analysis.")
        parts = []
        for identity in identities:
            entry = catalog.get(identity)
            if not entry or entry["grid"]["product"] != "historical":
                raise IMDFormatError("ML requires the historical 1° product only; recent 0.5° grids cannot be merged with it.")
            part = pd.read_parquet(self.parquet_path(identity), filters=[
                ("latitude", "==", latitude), ("longitude", "==", longitude)])
            if part.empty:
                raise IMDFormatError("The selected latitude/longitude is not an IMD grid coordinate.")
            parts.append(part)
        cell = pd.concat(parts, ignore_index=True).sort_values("date")
        if cell.date.duplicated().any():
            raise IMDFormatError("Selected files overlap in date. Choose one version of each year.")
        cell = cell.rename(columns={"maximum_temperature_c": "tmax", "data_source": "source"})
        cell["city"] = f"IMD grid cell {latitude:g}°N, {longitude:g}°E"
        # Tmax-only files contain no minimum or daily mean. Never manufacture them.
        cell["tmin"] = float("nan")
        cell["tmean"] = float("nan")
        return cell.reset_index(drop=True)

    def nearest_historical_cell(self, identities: list[str], latitude: float, longitude: float):
        catalog = {entry["id"]: entry for entry in self.catalog()}
        if not identities or len(identities) != len(set(identities)):
            raise IMDFormatError("Select distinct annual historical datasets for analysis.")
        parts = []
        for identity in identities:
            if identity not in catalog or catalog[identity]["grid"]["product"] != "historical":
                raise IMDFormatError("Nearest historical cell requires annual 1° datasets only.")
            parts.append(pd.read_parquet(self.parquet_path(identity), columns=[
                "date", "latitude", "longitude", "maximum_temperature_c"]))
        return nearest_available(pd.concat(parts, ignore_index=True), latitude, longitude)

    def statistics(self, identity: str) -> dict[str, pd.DataFrame]:
        return grid_statistics(pd.read_parquet(self.parquet_path(identity)))
