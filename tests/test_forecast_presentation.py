import pandas as pd

from climate.charts import prediction_chart
from climate.storytelling import forecast_table, mae_summary


def test_forecast_boundary_symbols_endpoint_and_dates_come_from_inputs():
    history = pd.DataFrame({"date": pd.date_range("2025-12-10", periods=22, tz="Asia/Kolkata"), "tmax": 30.})
    predictions = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3, tz="Asia/Kolkata"),
        "horizon": [1, 2, 3], "prediction": [31., 32., 33.], "baseline": 30.})
    chart = prediction_chart(history, predictions)
    assert chart.layout.shapes[0].x0 == history.date.max()
    assert chart.data[0].line.dash is None
    assert chart.data[1].line.dash == "dash" and chart.data[1].marker.symbol == "diamond"
    assert chart.data[2].line.dash == "dot" and chart.data[2].marker.symbol == "circle-open"
    table = forecast_table(history, predictions)
    assert table["Historical input endpoint"].eq(history.date.max()).all()
    assert table["Forecast date"].equals(predictions.date)
    assert table["Days after endpoint"].tolist() == [1, 2, 3]


def test_mae_summary_reports_improvement_decline_and_tie_without_validation_leakage():
    metrics = pd.DataFrame([{"partition": "Test", "horizon": h, "model": model, "mae": error, "rmse": error}
        for h, forest in [(1, 1.), (2, 3.), (3, 2.)]
        for model, error in [("Random Forest", forest), ("Persistence", 2.)]])
    table = mae_summary(metrics)
    assert table["MAE change: RF − persistence (°C)"].tolist() == [-1., 1., 0.]
    assert table.Result.tolist() == ["Random Forest improved", "Random Forest worse", "Equal performance"]
    assert mae_summary(metrics.assign(partition="Validation")).empty
