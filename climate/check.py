"""Real-network diagnostic: python -m climate.check --city Delhi --days 365."""
import argparse
from datetime import timedelta
import logging

from climate.api import WeatherClient, WeatherError
from climate.evaluation import ModelUnavailable
from climate.locations import CITIES
from climate.model import train_and_predict
from climate.service import load_history
from climate.weather import current_and_forecast, local_today


def main() -> int:
    parser = argparse.ArgumentParser(description="Check live Open-Meteo ingestion and local ML.")
    parser.add_argument("--city", choices=list(CITIES), default="Delhi")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    if not 150 <= args.days <= 3653:
        parser.error("--days must be between 150 and 3653")
    client = WeatherClient()
    try:
        today = local_today(CITIES[args.city])
        current, forecast = current_and_forecast(client, CITIES[args.city])
        print("Current model estimate:", current)
        print("Provider forecast:\n", forecast[["date", "tmax", "source"]].to_string(index=False))
        _, history, quality = load_history(args.city, today - timedelta(days=args.days), today - timedelta(days=1))
        print("Data quality:", quality)
        result = train_and_predict(history)
        print("Local ML predictions:\n", result.predictions.to_string(index=False))
        print("Evaluation:\n", result.metrics.to_string(index=False))
        return 0
    except (WeatherError, ModelUnavailable, ValueError, OSError) as exc:
        print(f"Check failed: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
