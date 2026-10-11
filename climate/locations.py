"""Supported city centres; all timestamps use IANA location timezones."""
from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    timezone: str = "Asia/Kolkata"


CITIES = {city.name: city for city in (
    City("Mumbai", 19.0760, 72.8777),
    City("Delhi", 28.6139, 77.2090),
    City("Pune", 18.5204, 73.8567),
    City("Ahmedabad", 23.0225, 72.5714),
    City("Jaipur", 26.9124, 75.7873),
    City("Nagpur", 21.1458, 79.0882),
)}
