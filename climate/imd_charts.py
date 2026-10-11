"""Geographic heatmaps of real imported grid cells, without interpolation."""
import pandas as pd
import plotly.graph_objects as go

from climate.locations import CITIES


def grid_heatmap(frame: pd.DataFrame, title: str, minimum: float, maximum: float) -> go.Figure:
    if frame.empty or frame.maximum_temperature_c.notna().sum() == 0:
        raise ValueError("No valid temperatures are available for the selected date.")
    grid = frame.pivot(index="latitude", columns="longitude", values="maximum_temperature_c").sort_index()
    grid = grid.reindex(sorted(grid.columns), axis=1)
    figure = go.Figure(go.Heatmap(x=grid.columns, y=grid.index, z=grid.to_numpy(),
        zmin=minimum, zmax=maximum, colorscale="YlOrRd", zsmooth=False, hoverongaps=False,
        colorbar=dict(title="Tmax (°C)"),
        hovertemplate="Longitude %{x:.1f}°E<br>Latitude %{y:.1f}°N<br>Maximum %{z:.2f} °C<extra>IMD grid</extra>"))
    cities = list(CITIES.values())
    figure.add_scatter(x=[city.longitude for city in cities], y=[city.latitude for city in cities],
        text=[city.name for city in cities], mode="markers+text", textposition="top center",
        marker=dict(size=5, color="#30363d"), textfont=dict(size=10),
        hovertemplate="%{text}<extra>Location reference only</extra>", name="City locations")
    figure.update_layout(title=title, template="plotly_white", height=650,
        margin=dict(l=20, r=20, t=55, b=30), xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        yaxis=dict(scaleanchor="x", scaleratio=1), legend=dict(orientation="h", y=-.15))
    return figure
