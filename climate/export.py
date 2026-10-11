"""CSV serialization retains timestamps, city names and data provenance."""
import pandas as pd


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S%z").encode("utf-8-sig")
