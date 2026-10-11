"""Explicit local annual GRD import: python -m climate.imd_import --help."""
import argparse
import json
from pathlib import Path

from climate.imd_parser import parse_grd
from climate.imd_store import IMDStore


def import_annual(path: Path, year: int, byte_order: str, store: IMDStore) -> dict:
    """Validate before writing; retain the original file and source provenance."""
    if path.stat().st_size > 2_000_000:
        raise ValueError("The annual 31x31 Tmax product must be smaller than 2 MB.")
    payload = path.read_bytes()
    parsed = parse_grd(payload, product="historical", year=year, byte_order=byte_order)
    identity = store.save(parsed, payload, path.name)
    return {**parsed.metadata, "id": identity,
            "parquet_path": str(store.parquet_path(identity)),
            "original_file": str(path), "live_weather": False}


def main():
    parser = argparse.ArgumentParser(description="Validate and cache an official annual IMD Tmax GRD")
    parser.add_argument("file", type=Path)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--byte-order", choices=["little", "big"], required=True)
    parser.add_argument("--store", type=Path, default=Path("data/imd"))
    args = parser.parse_args()
    try:
        report = import_annual(args.file, args.year, "<" if args.byte_order == "little" else ">",
                               IMDStore(args.store))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"IMD import failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
