"""Small reusable Streamlit explanations; no ingestion or model changes."""
import pandas as pd
import streamlit as st

from climate.storytelling import baseline_comparison, coverage, sources, mae_summary


def chart_story(frame, explanation, observation, how_to_read, *, source=None):
    st.caption(f"Source: {source or sources(frame)} · {coverage(frame)} · Temperature in °C")
    st.write(explanation)
    st.markdown(f"**What the displayed data shows:** {observation}")
    with st.expander("How to read this chart"):
        st.write(how_to_read)


def processing_table(details):
    st.dataframe(pd.DataFrame([{"Item": key, "Actual dataset summary": str(value)}
                              for key, value in details.items()]), hide_index=True, width="stretch")


def processing_overview(details, before_after):
    st.markdown("**Acquisition → Validation → Cleaning → Aggregation → Analytics → Visualization**")
    st.caption(f"Source: {details['Source dataset']} · {details['Date coverage']} · {details['Data represents']}")
    st.dataframe(before_after, hide_index=True, width="stretch")
    st.caption("Before/after counts need not decrease: calendar alignment can add missing records, and IMD conversion preserves missing cells.")
    invalid_key = "Raw records flagged invalid" if "Raw records flagged invalid" in details else "Invalid records retained"
    duplicates_key = "Duplicate records removed" if "Duplicate records removed" in details else "Duplicate removal"
    st.write(f"{invalid_key}: {details[invalid_key]} · {duplicates_key}: {details[duplicates_key]}")
    st.caption(f"Smoothing: {details['Smoothing']}. Exports: {details['Exports available']}.")


def model_intro():
    st.write("Random Forest predicts the daily maximum temperature 1, 2 and 3 days after the last completed day in its input history. Each horizon has its own model.")
    with st.expander("What goes into the model?"):
        st.write("Inputs are the latest daily maximum; maxima 1, 2, 3 and 7 days earlier; trailing 7- and 14-day averages; and the target date's month and annual seasonal cycle. All temperature inputs are available at the forecast origin. Future weather and chart smoothing are excluded.")
        st.write("Persistence is the simple comparison: predict that the latest available daily maximum repeats for each future date. A complex model is useful only if its measured performance improves on this baseline.")
    st.caption("Experimental estimates: no calibrated uncertainty interval, no official heatwave warning, and no guarantee that unusual future weather is represented by the training history.")


def evaluation_story(result):
    st.subheader("Held-out model evaluation")
    st.write("MAE is the average size of a prediction error, in °C. RMSE is also in °C, but gives larger errors more weight. Lower values are better for both.")
    table, conclusion = baseline_comparison(result.metrics)
    st.markdown(f"**Did the model beat the simple baseline?** {conclusion}")
    st.markdown("**Model Performance Summary · test MAE**")
    summary = mae_summary(result.metrics)
    if not summary.empty:
        st.dataframe(summary.round(3), hide_index=True, width="stretch")
        st.caption("MAE change = Random Forest minus persistence: negative is an improvement; positive means Random Forest performed worse. Results are separate for each number of days ahead.")
    if not table.empty:
        with st.expander("Compare MAE and RMSE at every horizon"):
            st.dataframe(table.round(3), hide_index=True, width="stretch")
    with st.expander("Full validation and test metrics"):
        st.dataframe(result.metrics, hide_index=True, width="stretch")
