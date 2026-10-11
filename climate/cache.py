"""Atomic Parquet cache; expired data is never silently served."""
from hashlib import sha256
import json
import logging
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd

LOGGER = logging.getLogger(__name__)


class ParquetCache:
    def __init__(self, root: str | Path = "data/raw"):
        self.root = Path(root)

    def path(self, key: dict) -> Path:
        digest = sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.root / f"{digest}.parquet"

    def read(self, key: dict, ttl: float = 86400) -> pd.DataFrame | None:
        path = self.path(key)
        try:
            if not path.exists() or time.time() - path.stat().st_mtime > ttl:
                return None
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Cannot read cache %s: %s", path, exc)
            return None

    def write(self, key: dict, frame: pd.DataFrame) -> None:
        temp = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.path(key)
            temp = path.with_suffix(f".{uuid4().hex}.tmp")
            frame.to_parquet(temp, index=False)
            temp.replace(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Cannot write cache: %s", exc)
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError as exc:
                    LOGGER.warning("Cannot remove temporary cache file: %s", exc)
