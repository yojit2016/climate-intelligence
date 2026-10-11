"""Automatic multi-year acquisition, isolated from dashboard date filters."""
from dataclasses import dataclass, field
from datetime import date, timedelta
import logging

import pandas as pd

from climate.api import WeatherError
from climate.locations import CITIES
from climate.service import CURRENT_TTL, HISTORY_TTL, load_history
from climate.weather import local_today

LOGGER = logging.getLogger(__name__)
TRAINING_YEARS = 3


@dataclass
class AcquisitionResult:
    start: date
    end: date
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def training_period(as_of: date, years: int = TRAINING_YEARS) -> tuple[date, date]:
    if years not in (3, 4, 5):
        raise ValueError("Automatic training history must cover three, four or five years.")
    return (pd.Timestamp(as_of) - pd.DateOffset(years=years)).date(), as_of - timedelta(days=1)


def acquire_supported(as_of: date, years: int = TRAINING_YEARS,
                      force_refresh: bool = False) -> AcquisitionResult:
    if as_of > local_today(CITIES["Delhi"]):
        raise ValueError("Acquisition date cannot be in the future.")
    start, end = training_period(as_of, years)
    result = AcquisitionResult(start, end)
    for name in CITIES:
        try:
            result.datasets[name] = load_history(name, start, end, force_refresh=force_refresh)
        except (WeatherError, ValueError, OSError) as exc:
            LOGGER.warning("Automatic history acquisition failed for %s: %s", name, exc)
            result.errors[name] = str(exc)
    return result
