import numpy as np
import pandas as pd
import pytest

from climate.evaluation import ModelUnavailable, chronological_split, persistence, scores
from climate.features import FEATURES, daily_series, make_features


def series():
    return pd.Series(np.arange(240, dtype=float),
                     index=pd.date_range("2025-01-01", periods=240, tz="Asia/Kolkata"))


def test_features_are_causal_at_each_horizon():
    values = series()
    for horizon in (1, 2, 3):
        before = make_features(values, horizon)
        changed = values.copy()
        changed.iloc[101:] = -1234
        after = make_features(changed, horizon)
        pd.testing.assert_frame_equal(before.loc[:values.index[100], FEATURES],
                                      after.loc[:values.index[100], FEATURES])
        row = before.iloc[100]
        assert row.latest == 100 and row.lag_1 == 99 and row.mean_7 == 97
        assert row.target == 100 + horizon
        assert row.target_date == values.index[100 + horizon]


def test_calendar_gaps_remain_missing():
    values = series()
    frame = pd.DataFrame({"date": values.index, "city": "Delhi", "tmax": values.values})
    full = daily_series(frame.drop(index=50))
    assert pd.isna(full.iloc[50])
    assert pd.isna(make_features(full, 1).iloc[51].lag_1)


def test_splits_exclude_labels_crossing_boundaries():
    features = make_features(series(), 3).dropna()
    train, validation, test = chronological_split(features)
    assert train.target_date.max() < validation.index.min()
    assert validation.target_date.max() < test.index.min()
    assert train.index.intersection(test.index).empty
    np.testing.assert_array_equal(persistence(test), test.latest)
    assert scores([1, 3], [2, 2]) == {"mae": 1., "rmse": 1.}
    with pytest.raises(ModelUnavailable):
        chronological_split(features.iloc[:30])
