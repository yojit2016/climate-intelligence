"""Plotly figures with explicit units, legends and source-neutral styling."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = ["#34495e", "#b35b45", "#467a77", "#827191", "#a0884b", "#687d94"]


def finish(fig: go.Figure, title: str, y: str = "Temperature (°C)") -> go.Figure:
    fig.update_layout(template="plotly_white", title=title, height=350,
        font=dict(family="Arial, sans-serif", color="#30363d", size=12),
        margin=dict(l=16, r=16, t=50, b=16), paper_bgcolor="white",
        legend=dict(orientation="h", y=-.22), hovermode="x unified")
    fig.update_yaxes(title=y, gridcolor="#edf0f2")
    fig.update_xaxes(title=None, showgrid=False)
    fig.update_traces(connectgaps=False, selector=dict(type="scatter"))
    return fig


def trend_chart(frame: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=frame.date, y=frame.tmax, name="Daily maximum", line=dict(color=COLORS[0], width=1.5))
    fig.add_scatter(x=frame.date, y=frame.rolling_7d, name="7-day rolling mean", line=dict(color=COLORS[2], width=2))
    exceeds = frame.loc[frame.exceeds_threshold]
    fig.add_scatter(x=exceeds.date, y=exceeds.tmax, mode="markers", name="Exploratory exceedance",
                    marker=dict(color=COLORS[1], size=5))
    fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS[1],
                  annotation_text=f"Exploratory threshold: {threshold:g} °C")
    return finish(fig, "Daily maximum temperature")


def range_chart(frame: pd.DataFrame) -> go.Figure:
    fig = px.line(frame, x="date", y=["tmax", "tmin", "tmean"], color_discrete_sequence=COLORS,
                  labels={"variable": "Daily statistic", "value": "Temperature (°C)"})
    names = {"tmax": "Maximum", "tmin": "Minimum", "tmean": "Mean"}
    for trace in fig.data:
        trace.name = names[trace.name]
    return finish(fig, "Daily maximum, minimum and mean")


def smoothing_chart(frame: pd.DataFrame) -> go.Figure:
    fig = px.line(frame, x="date", y=["tmax", "smoothed"], color_discrete_sequence=["#b8bec5", COLORS[0]])
    for trace in fig.data:
        trace.name = {"tmax": "Raw maximum", "smoothed": "Smoothed maximum"}[trace.name]
    return finish(fig, "Raw and smoothed daily maximum")


def pattern_chart(frame: pd.DataFrame, category: str, title: str) -> go.Figure:
    fig = px.bar(frame, x=category, y=["mean_max", "mean_min"], barmode="group",
                 color_discrete_sequence=[COLORS[0], COLORS[2]])
    for trace in fig.data:
        trace.name = {"mean_max": "Mean maximum", "mean_min": "Mean minimum"}[trace.name]
    return finish(fig, title)


def comparison_chart(frame: pd.DataFrame) -> go.Figure:
    fig = px.line(frame.sort_values(["city", "date"]), x="date", y="tmax", color="city",
                  color_discrete_sequence=COLORS, labels={"city": "City"})
    return finish(fig, "Regional daily maxima · shared date range")


def prediction_chart(history: pd.DataFrame, predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    recent = history.tail(21)
    fig.add_scatter(x=recent.date, y=recent.tmax, name="Historical model estimates", line=dict(color=COLORS[0]))
    fig.add_scatter(x=predictions.date, y=predictions.prediction, name="Local Random Forest prediction",
                    mode="lines+markers", marker=dict(symbol="diamond", size=9), line=dict(color=COLORS[2], width=2, dash="dash"))
    fig.add_scatter(x=predictions.date, y=predictions.baseline, name="Persistence baseline",
                    mode="lines+markers", marker=dict(symbol="circle-open", size=9), line=dict(color="#747b84", width=2, dash="dot"))
    origin = history.date.max()
    if pd.notna(origin) and not predictions.empty:
        fig.add_shape(type="line", x0=origin, x1=origin, xref="x", y0=0, y1=1,
                      yref="paper", line=dict(color="#68717d", width=1.5, dash="dash"))
        fig.add_annotation(x=origin, xref="x", y=1, yref="paper", text=f"History ends {origin:%d %b %Y}",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=10))
        fig.add_vrect(x0=origin + pd.Timedelta(hours=12), x1=predictions.date.max() + pd.Timedelta(hours=12),
                      fillcolor="#edf2f2", opacity=.6, line_width=0, layer="below")
    return finish(fig, "Predicted daily maximum for three dates")


def evaluation_chart(test: pd.DataFrame) -> go.Figure:
    first = test.loc[test.horizon == 1]
    fig = px.line(first, x="date", y=["actual", "prediction", "baseline"], color_discrete_sequence=COLORS)
    for trace in fig.data:
        trace.name = {"actual": "Held-out reanalysis/model target", "prediction": "Random Forest",
                      "baseline": "Persistence"}[trace.name]
    return finish(fig, "One-day predictions compared with held-out temperatures")
