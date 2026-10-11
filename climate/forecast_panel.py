"""Explicit separation between locally trained ML and provider forecasts."""
import logging

import pandas as pd
import streamlit as st

from climate.charts import evaluation_chart, prediction_chart
from climate.evaluation import ModelUnavailable
from climate.locations import CITIES
from climate.model import train_and_predict
from climate.weather import local_today
from climate.presentation import chart_story, model_intro, evaluation_story
from climate.storytelling import prediction_observation, baseline_comparison, sources

LOGGER = logging.getLogger(__name__)
cached_model = st.cache_resource(ttl=21600, max_entries=8, show_spinner=False)(train_and_predict)


def forecast_panel(frame: pd.DataFrame):
    st.subheader("Local Random Forest model")
    model_intro()
    origin = frame.date.max()
    city = frame.city.iloc[0]
    today = local_today(CITIES[city])
    if origin.date() < today - pd.Timedelta(days=1):
        st.warning("The training dataset ends before yesterday. Predictions follow that historical endpoint and are not an upcoming forecast. Refresh weather data to retry acquisition.")
    st.caption(f"Forecast origin: {origin:%d %b %Y} after the completed local day. Direct horizons 1–3 use only temperatures available at this origin. Results are experimental, without calibrated uncertainty intervals.")
    try:
        with st.spinner("Training and evaluating three forecasting models…"):
            result = cached_model(frame)
    except ModelUnavailable as exc:
        st.info(f"Local ML predictions unavailable: {exc}")
    except Exception:
        LOGGER.exception("Model training failed")
        st.error("Local ML training failed. No predictions were generated. Check application logs and retry with a different date range.")
    else:
        st.plotly_chart(prediction_chart(frame, result.predictions), width="stretch")
        shown = pd.concat([frame.tail(21)[["date"]], result.predictions[["date"]]], ignore_index=True)
        chart_story(shown, "Recent history provides context for the next three model prediction dates. The dotted baseline repeats the latest daily maximum.",
                    prediction_observation(result.predictions),
                    "Solid values are historical inputs. Dashed estimates are local Random Forest outputs; the dotted line is persistence. Prediction dates follow the history endpoint, so stale history produces historical experiments rather than upcoming forecasts.", source=sources(frame))
        st.dataframe(result.predictions[["date", "horizon", "prediction", "baseline"]],
                     hide_index=True, width="stretch")
        evaluation_story(result)
        st.caption("Test targets are Open-Meteo reanalysis/model estimates, not independent station measurements.")
        st.plotly_chart(evaluation_chart(result.test_predictions), width="stretch")
        first = result.test_predictions.loc[result.test_predictions.horizon == 1]
        _, verdict = baseline_comparison(result.metrics.loc[result.metrics.horizon == 1])
        chart_story(first, "This compares one-day-ahead predictions with the actual held-out source targets. These dates were reserved for evaluation.",
                    verdict, "Look for agreement with the held-out target line. This chart shows only the one-day horizon; the error table evaluates all three. Test scoring uses past values available at each origin, not future temperatures.", source=sources(frame))
        test = result.metrics.query("partition == 'Test'")
        pivot = test.pivot(index="horizon", columns="model", values="mae")
        worse = pivot.index[pivot["Random Forest"] >= pivot["Persistence"]].tolist()
        if worse:
            st.warning(f"Random Forest did not improve on persistence MAE at horizons {worse}. Treat those predictions cautiously.")
        with st.expander("Training and evaluation method"):
            st.write("Chronological 70% train / 15% validation / 15% test split of complete feature rows. Boundary-crossing targets are purged. Two tree-depth options are selected by validation MAE; test predictions use a refit on pre-test labels only. Test scoring uses rolling origins with observed past values and no test-period retraining. Production models are then separately refit on all available labels.")
            st.dataframe(result.splits, hide_index=True, width="stretch")
            st.caption("No smoothed values or provider future forecasts enter model features. Models are cached in memory by dataset for 6 hours; changing history invalidates the cache.")
