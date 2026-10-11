"""Geographic heatmaps of real imported grid cells, without interpolation."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from climate.locations import CITIES


def grid_heatmap(frame: pd.DataFrame, title: str, minimum: float, maximum: float, *, height=720) -> go.Figure:
    if frame.empty or frame.maximum_temperature_c.notna().sum() == 0:
        raise ValueError("No valid temperatures are available for the selected date.")
    grid = frame.pivot(index="latitude", columns="longitude", values="maximum_temperature_c").sort_index()
    grid = grid.reindex(sorted(grid.columns), axis=1)
    if frame.date.nunique() != 1:
        raise ValueError("A temperature map requires one product date.")
    product_date = frame.date.iloc[0].strftime("%d %b %Y")
    figure = go.Figure(go.Heatmap(x=grid.columns, y=grid.index, z=grid.to_numpy(),
        zmin=minimum, zmax=maximum, colorscale="YlOrRd", zsmooth=False, hoverongaps=False,
        colorbar=dict(title="Tmax (°C)"),
        hovertemplate=f"Date {product_date}<br>Longitude %{{x:.1f}}°E<br>Latitude %{{y:.1f}}°N<br>Maximum %{{z:.2f}} °C<extra>IMD grid</extra>"))
    cities = list(CITIES.values())
    figure.add_scatter(x=[city.longitude for city in cities], y=[city.latitude for city in cities],
        text=[city.name for city in cities], mode="markers+text", textposition="top center",
        marker=dict(size=5, color="#30363d"), textfont=dict(size=10),
        hovertemplate="%{text}<extra>Location reference only</extra>", name="City locations")
    figure.update_layout(title=title, template="plotly_white", height=height, plot_bgcolor="#f0f1f3",
        margin=dict(l=20, r=20, t=55, b=30), xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        yaxis=dict(scaleanchor="x", scaleratio=1), legend=dict(orientation="h", y=-.15))
    return figure


def difference_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Symmetric zero-centered differences; leave unavailable cell pairs blank."""
    values = frame.difference_c.dropna()
    if values.empty:
        raise ValueError("No cells are valid on both dates; a difference map cannot be drawn.")
    date_a, date_b = frame.date_a.iloc[0], frame.date_b.iloc[0]
    # A zero-only field still needs a nondegenerate color range; values stay zero.
    limit = max(float(np.abs(values).max()), 1e-6)
    mapped = frame.assign(date=frame.date_b, maximum_temperature_c=frame.difference_c)
    figure = grid_heatmap(mapped, f"Change in daily maximum · {date_b:%d %b} minus {date_a:%d %b %Y}", -limit, limit)
    figure.data[0].update(colorscale="RdBu_r", zmid=0, colorbar=dict(title="B − A (°C)"),
        hovertemplate=f"A: {date_a:%d %b %Y}<br>B: {date_b:%d %b %Y}<br>Longitude %{{x:.1f}}°E<br>Latitude %{{y:.1f}}°N<br>Change %{{z:+.2f}} °C<extra>IMD daily Tmax difference</extra>")
    return figure
