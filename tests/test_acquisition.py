from datetime import date
from unittest.mock import Mock

import pytest

from climate.acquisition import acquire_supported, training_period
from climate.api import WeatherError
from climate.locations import CITIES


def test_training_period_covers_calendar_years_and_completed_days():
    assert training_period(date(2024, 2, 29)) == (date(2021, 2, 28), date(2024, 2, 28))
    assert training_period(date(2025, 3, 1), 5) == (date(2020, 3, 1), date(2025, 2, 28))
    with pytest.raises(ValueError):
        training_period(date(2025, 1, 1), 2)


def test_automatic_acquisition_attempts_every_city_and_retains_partial_success(monkeypatch):
    sentinel = ("raw test data", "processed test data", {"valid_days": 1000})
    def load(name, start, end, *, force_refresh):
        assert start == date(2022, 1, 10) and end == date(2025, 1, 9)
        assert force_refresh
        if name == "Delhi":
            raise WeatherError("offline")
        return sentinel
    loader = Mock(side_effect=load)
    monkeypatch.setattr("climate.acquisition.load_history", loader)
    monkeypatch.setattr("climate.acquisition.local_today", lambda city: date(2025, 1, 10))
    result = acquire_supported(date(2025, 1, 10), force_refresh=True)
    assert loader.call_count == len(CITIES)
    assert set(result.datasets) == set(CITIES) - {"Delhi"}
    assert result.errors == {"Delhi": "offline"}
