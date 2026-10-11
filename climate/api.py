"""Bounded, retrying Open-Meteo HTTP transport. No synthetic fallback."""
import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)


class WeatherError(RuntimeError):
    """An upstream weather request or response could not be used."""


class WeatherClient:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        retries = Retry(total=3, backoff_factor=0.7, status_forcelist=(429, 500, 502, 503, 504),
                        allowed_methods=("GET",), respect_retry_after_header=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get(self, url: str, params: dict[str, Any]) -> dict:
        try:
            response = self.session.get(url, params=params, timeout=(10, 45))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if payload.get("error"):
                raise WeatherError(str(payload.get("reason", "Open-Meteo error")))
            return payload
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Open-Meteo request failed: %s", exc)
            raise WeatherError("Weather service unavailable or returned invalid data. Retry later.") from exc

    def close(self) -> None:
        self.session.close()
