import numpy as np
import pandas as pd
import pytest

from climate.evaluation import ModelUnavailable
from climate.model import train_and_predict


def fixture_history(days=240):
    """Synthetic test fixture only; never used as application weather."""
    x = np.arange(days)
    temperatures = 30 + 5 * np.sin(x / 35) + np.sin(x / 3)
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=days, tz="Asia/Kolkata"),
                         "city": "Delhi", "tmax": temperatures})


def test_three_direct_forecasts_and_real_test_metrics():
    frame = fixture_history()
    result = train_and_predict(frame)
    assert result.predictions.horizon.tolist() == [1, 2, 3]
    assert result.predictions.date.min() == frame.date.max() + pd.Timedelta(days=1)
    assert np.isfinite(result.predictions.prediction).all()
    assert len(result.metrics) == 12 and set(result.models) == {1, 2, 3}
    for horizon in (1, 2, 3):
        test = result.test_predictions.query("horizon == @horizon")
        metric = result.metrics.query("horizon == @horizon and partition == 'Test' and model == 'Random Forest'")
        assert metric.iloc[0].mae == pytest.approx(abs(test.actual - test.prediction).mean())
    pd.testing.assert_frame_equal(frame, fixture_history())


def test_insufficient_history_and_recent_gaps_fail_clearly():
    with pytest.raises(ModelUnavailable, match="120"):
        train_and_predict(fixture_history(40))
    frame = fixture_history()
    frame.loc[len(frame) - 3, "tmax"] = np.nan
    with pytest.raises(ModelUnavailable, match="14 days"):
        train_and_predict(frame)
