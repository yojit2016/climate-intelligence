"""Provider contract: adapters return raw daily tables with provenance.

Future IMD/NASA POWER adapters can implement this contract without changing
preprocessing or forecasting. Observational providers must identify observations
explicitly in the source column; Open-Meteo supplies model/reanalysis values.
"""
from datetime import date
from typing import Protocol

import pandas as pd

from climate.api import WeatherClient
from climate.locations import City
from climate.weather import current_and_forecast, historical


class WeatherSource(Protocol):
    cache_id: str

    def history(self, city: City, start: date, end: date) -> pd.DataFrame: ...
    def current(self, city: City) -> tuple[dict, pd.DataFrame]: ...
    def close(self) -> None: ...


class OpenMeteoSource:
    cache_id = "open-meteo-archive-only-v2"

    def __init__(self):
        self.client = WeatherClient()

    def history(self, city: City, start: date, end: date) -> pd.DataFrame:
        return historical(self.client, city, start, end)

    def current(self, city: City) -> tuple[dict, pd.DataFrame]:
        current, forecast = current_and_forecast(self.client, city)
        return {**current, "source": "Open-Meteo current model estimate"}, forecast

    def close(self) -> None:
        self.client.close()
