"""Live verification: python -m climate.check --all-cities --refresh."""
import argparse
from datetime import timedelta
import logging

from climate.acquisition import acquire_supported, training_period
from climate.api import WeatherError
from climate.evaluation import ModelUnavailable
from climate.locations import CITIES
from climate.model import train_and_predict
from climate.service import load_current, load_history
from climate.weather import local_today


def check_city(name, start, end, *, dataset=None, force_refresh=False) -> int:
    print(f"\n=== {name}: {start} through {end} ===", flush=True)
    try:
        current, forecast = load_current(name, force_refresh=force_refresh)
        print("Current model estimate:", current, flush=True)
        print("Provider forecast:\n", forecast[["date", "tmax", "source"]].to_string(index=False), flush=True)
        _, history, quality = dataset if dataset is not None else load_history(name, start, end, force_refresh=force_refresh)
        print("Data quality:", quality, flush=True)
        print("Historical sources:", history.source.unique().tolist(), flush=True)
        print("Latest history:\n", history[["date", "tmax"]].tail(3).to_string(index=False), flush=True)
        result = train_and_predict(history)
        print("Local ML predictions:\n", result.predictions.to_string(index=False), flush=True)
        print("Evaluation:\n", result.metrics.to_string(index=False), flush=True)
        return 0
    except (WeatherError, ModelUnavailable, ValueError, OSError) as exc:
        print(f"Check failed for {name}: {exc}", flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Download real historical/current weather and verify local ML.")
    parser.add_argument("--city", choices=list(CITIES), default="Delhi")
    parser.add_argument("--years", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--days", type=int, help="Optional single-city custom diagnostic range")
    parser.add_argument("--all-cities", action="store_true", help="Acquire and verify every supported city")
    parser.add_argument("--refresh", action="store_true", help="Bypass disk caches and make fresh API requests")
    args = parser.parse_args()
    if args.days is not None and (not 150 <= args.days <= 3653 or args.all_cities):
        parser.error("--days must be 150–3653 and can only be used for a single city")
    today = local_today(CITIES[args.city])
    start, end = training_period(today, args.years)
    if args.days is not None:
        start = today - timedelta(days=args.days)
    if args.all_cities:
        acquired = acquire_supported(today, args.years, force_refresh=args.refresh)
        for name, error in acquired.errors.items():
            print(f"Acquisition failed for {name}: {error}", flush=True)
        failed = bool(acquired.errors)
        for name, dataset in acquired.datasets.items():
            failed |= bool(check_city(name, acquired.start, acquired.end,
                                      dataset=dataset, force_refresh=args.refresh))
        return int(failed)
    return check_city(args.city, start, end, force_refresh=args.refresh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
