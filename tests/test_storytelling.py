import numpy as np
import pandas as pd

from climate.processing import clean_daily
from climate.storytelling import baseline_comparison, group_observation, peak_observation, point_processing_summary, imd_processing_summary, prediction_observation


def test_observations_use_available_values_and_handle_empty_groups():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3), "tmax": [np.nan, 42., 30.], "source": "Actual test source"})
    text = peak_observation(frame)
    assert "42.00°C" in text and "02 Jan 2025" in text and "2 available" in text
    assert "No valid" in peak_observation(frame.assign(tmax=np.nan))
    assert "no valid" in group_observation(pd.DataFrame({"month": ["Jan"], "mean_max": [np.nan]}), "month")


def test_baseline_verdict_handles_mixed_results_and_ties():
    metrics = pd.DataFrame([
        {"partition": "Test", "horizon": 1, "model": "Random Forest", "mae": 1., "rmse": 3.},
        {"partition": "Test", "horizon": 1, "model": "Persistence", "mae": 2., "rmse": 2.},
        {"partition": "Test", "horizon": 2, "model": "Random Forest", "mae": 2., "rmse": 2.},
        {"partition": "Test", "horizon": 2, "model": "Persistence", "mae": 2., "rmse": 3.},
        {"partition": "Validation", "horizon": 1, "model": "Random Forest", "mae": 0., "rmse": 0.}])
    table, text = baseline_comparison(metrics)
    assert len(table) == 4 and "lower MAE at 1 of 2" in text and "1 tied" in text
    assert "lower RMSE at 1 of 2" in text
    assert table.Result.eq("Equal").sum() == 1
    assert "unavailable" in baseline_comparison(metrics.iloc[:1])[1]


def test_processing_summary_does_not_label_invalid_events_as_invalid_rows():
    raw = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3), "city": "Delhi",
                        "tmax": [100., 30., np.nan], "tmin": [-100., 20., 10.],
                        "tmean": [25., 25., 15.], "source": "Test source"})
    frame, report = clean_daily(raw)
    result = point_processing_summary(raw, frame, report)
    assert report["invalid_values_or_rows"] == 2
    assert result["Raw records flagged invalid"] == 1
    assert result["Valid maximum-temperature records"] == 1
    assert result["Missing maximum-temperature values"] == 2
    assert result["Source dataset"] == "Test source"
    assert "filter off" in result["Smoothing"]
    assert "11-day" in point_processing_summary(raw, frame, report, smoothing=True, window=11)["Smoothing"]


def test_imd_summary_counts_cell_days_and_labels_source_interpolation():
    result = imd_processing_summary({"days": 365, "grid": {"size": 31, "step": 1},
        "original_filename": "MAX.GRD", "valid_cells": 129552, "missing_cells": 221213,
        "start": "2025-01-01", "end": "2025-12-31"})
    assert result["Records acquired (grid cell-days)"] == 350765
    assert "IMD source product" in result["Interpolation by this application"]


def test_prediction_observation_reports_real_dates_and_does_not_claim_observation():
    predictions = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3),
                                "prediction": [30., 32.5, np.nan]})
    text = prediction_observation(predictions)
    assert "30.00–32.50°C" in text and "02 Jan 2026" in text
    assert "2 prediction dates" in text and "not measured" in text
    assert "No model predictions" in prediction_observation(predictions.assign(prediction=np.nan))
