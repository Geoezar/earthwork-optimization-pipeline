from __future__ import annotations

from pathlib import Path
from typing import Optional
import webbrowser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.optimization import OptimizationResult
from core.reporting import DEFAULT_JURY_RESULTS_URL, create_word_report


def save_html(figure: go.Figure, output_path: Path, open_in_browser: bool = False) -> None:
    figure.write_html(str(output_path), auto_open=False, config={"responsive": True, "scrollZoom": True})
    if open_in_browser:
        webbrowser.open(output_path.resolve().as_uri())
    print(f"  Saved: {output_path.name}")


def create_3d_model(result: OptimizationResult, frame: pd.DataFrame) -> go.Figure:
    z_min = float(frame["Y"].min()) - 1.0
    z_max = float(frame["Y"].max()) + 1.0
    max_deviation = float(np.abs(result.deviation_grid).max()) or 0.01
    figure = go.Figure()

    original_surface = (
        result.original_intercept
        + result.original_x1_coefficient * result.grid_x1
        + result.original_x2_coefficient * result.grid_x2
    )
    figure.add_trace(
        go.Surface(
            x=result.grid_x1,
            y=result.grid_x2,
            z=original_surface,
            colorscale=[[0, "#C0392B"], [1, "#C0392B"]],
            opacity=0.28,
            showscale=False,
            name="Natural Trend Plane",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    figure.add_trace(
        go.Surface(
            x=result.grid_x1,
            y=result.grid_x2,
            z=result.design_surface,
            colorscale=[[0, "#2563EB"], [1, "#2563EB"]],
            opacity=0.5,
            showscale=False,
            showlegend=True,
            name="Optimized Design Plane",
            hoverinfo="skip",
            contours=dict(
                x=dict(show=True, color="black", width=2, start=0, end=100, size=10),
                y=dict(show=True, color="black", width=2, start=0, end=100, size=10),
            ),
        )
    )

    figure.add_trace(
        go.Surface(
            x=result.grid_x1,
            y=result.grid_x2,
            z=result.natural_surface,
            surfacecolor=result.deviation_grid,
            colorscale="RdBu_r",
            cmin=-max_deviation,
            cmax=max_deviation,
            colorbar=dict(
                title="<b>Deviation (m)</b><br>Red = Cut<br>Blue = Fill",
                thickness=20,
                len=0.75,
                x=1.05,
                bgcolor="rgba(245,247,250,0.85)",
            ),
            opacity=0.85,
            showlegend=True,
            name="RBF Terrain Surface",
            hovertemplate="<b>X1:</b> %{x:.2f} m<br><b>X2:</b> %{y:.2f} m<br><b>Elevation:</b> %{z:.2f} m<extra></extra>",
        )
    )

    point_numbers = np.arange(1, len(frame) + 1)
    figure.add_trace(
        go.Scatter3d(
            x=frame["X1"].values,
            y=frame["X2"].values,
            z=frame["Y"].values,
            mode="markers",
            name="Survey Points",
            marker=dict(size=3, color="#DC2626", line=dict(color="white", width=1)),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=frame["X1"].values,
            y=frame["X2"].values,
            z=frame["Y"].values,
            mode="markers",
            showlegend=False,
            marker=dict(size=35, color="rgba(0,0,0,0)"),
            customdata=np.column_stack((point_numbers, frame["X1"].values, frame["X2"].values, frame["Y"].values)),
            hovertemplate=(
                "<b>Point %{customdata[0]:.0f}</b><br>"
                "X1: %{customdata[1]:.2f} m<br>"
                "X2: %{customdata[2]:.2f} m<br>"
                "Elevation: %{customdata[3]:.2f} m<extra></extra>"
            ),
        )
    )

    centroid_x1 = float(frame["X1"].mean())
    centroid_x2 = float(frame["X2"].mean())
    centroid_y = float(frame["Y"].mean())
    figure.add_trace(
        go.Scatter3d(
            x=[centroid_x1],
            y=[centroid_x2],
            z=[centroid_y],
            mode="markers",
            name="Centroid Pivot",
            marker=dict(size=5, color="#FACC15", line=dict(color="black", width=2), symbol="diamond"),
            hovertemplate="<b>Centroid Pivot</b><br>X1: %{x:.2f} m<br>X2: %{y:.2f} m<br>Elevation: %{z:.2f} m<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter3d(
            x=[3, 3],
            y=[8, 15],
            z=[z_max, z_max],
            mode="lines+text",
            name="North Arrow",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color="black", width=6),
            text=["", "N"],
            textposition="top center",
            textfont=dict(size=14, color="black"),
        )
    )
    figure.add_trace(
        go.Cone(
            x=[3],
            y=[15],
            z=[z_max],
            u=[0],
            v=[6],
            w=[0],
            sizemode="absolute",
            sizeref=4,
            anchor="tail",
            showscale=False,
            colorscale=[[0, "#DC2626"], [1, "#DC2626"]],
            hoverinfo="skip",
        )
    )

    title = (
        f"<b>{result.area_name} - 3D Topographic Model</b><br>"
        f"<sup><b>Total earthwork:</b> {result.total_earthwork_m3:,.2f} m3 | "
        f"<b>Cut:</b> {result.cut_volume_m3:,.2f} m3 | <b>Fill:</b> {result.fill_volume_m3:,.2f} m3<br>"
        f"<b>Natural orientation:</b> {result.original_azimuth} | "
        f"<b>Design orientation:</b> {result.design_azimuth}<br>"
        f"<b>Status:</b> {result.geotechnical_warning}</sup>"
    )

    figure.update_layout(
        title=dict(text=title, x=0.5, y=0.95),
        paper_bgcolor="#F5F7FA",
        font=dict(color="black"),
        hoverlabel=dict(bgcolor="#111827", font_color="white", bordercolor="white"),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.2),
            zaxis=dict(range=[z_min, z_max]),
            xaxis_title="X1 East-West (m)",
            yaxis_title="X2 North-South (m)",
            zaxis_title="Elevation (m)",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8)),
        ),
        hoverdistance=200,
        margin=dict(l=10, r=10, b=10, t=160),
        legend=dict(
            title="<b>Layers</b>",
            x=0.02,
            y=0.95,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            itemsizing="constant",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0,
                y=1.05,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="rgba(0,0,0,0)",
                buttons=[
                    dict(label="Light", method="relayout", args=[{"paper_bgcolor": "#F5F7FA", "font.color": "black"}]),
                    dict(label="Dark", method="relayout", args=[{"paper_bgcolor": "#111827", "font.color": "white"}]),
                ],
            )
        ],
    )
    return figure


def create_cut_fill_heatmap(result: OptimizationResult, frame: pd.DataFrame) -> go.Figure:
    max_deviation = float(np.abs(result.deviation_grid).max()) or 0.01
    point_numbers = np.arange(1, len(frame) + 1)
    figure = go.Figure()

    figure.add_trace(
        go.Contour(
            x=result.grid_x1[0, :],
            y=result.grid_x2[:, 0],
            z=result.deviation_grid,
            colorscale="RdBu_r",
            zmin=-max_deviation,
            zmax=max_deviation,
            colorbar=dict(title="<b>Design Difference (m)</b>", tickformat=".2f", thickness=20),
            contours=dict(coloring="heatmap", showlabels=False),
            line=dict(width=0),
            hovertemplate="<b>X1:</b> %{x:.1f} m<br><b>X2:</b> %{y:.1f} m<br><b>Deviation:</b> %{z:.3f} m<extra></extra>",
            name="Cut/Fill Surface",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Contour(
            x=result.grid_x1[0, :],
            y=result.grid_x2[:, 0],
            z=result.deviation_grid,
            showscale=False,
            contours=dict(coloring="none", showlabels=True, labelfont=dict(size=11, color="black")),
            line=dict(color="rgba(0,0,0,0.5)", width=0.8, smoothing=1.3),
            hovertemplate="<b>Deviation contour:</b> %{z:.2f} m<extra></extra>",
            name="Cut/Fill Contours",
            showlegend=True,
        )
    )
    figure.add_trace(
        go.Contour(
            x=result.grid_x1[0, :],
            y=result.grid_x2[:, 0],
            z=result.natural_surface,
            showscale=False,
            contours=dict(coloring="none", showlabels=True, labelfont=dict(size=12, color="black")),
            line=dict(color="black", width=1.5, dash="dash", smoothing=1.3),
            hovertemplate="<b>Natural elevation:</b> %{z:.2f} m<extra></extra>",
            name="Natural Elevation Contours",
            showlegend=True,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["X1"].values,
            y=frame["X2"].values,
            mode="markers",
            marker=dict(size=6, color="#DC2626"),
            customdata=np.column_stack((point_numbers, frame["Y"].values)),
            hovertemplate="<b>Point %{customdata[0]:.0f}</b><br>X1: %{x:.2f} m<br>X2: %{y:.2f} m<br>Elevation: %{customdata[1]:.2f} m<extra></extra>",
            name="Survey Points",
            showlegend=False,
        )
    )

    title = (
        f"<b>{result.area_name} - 2D Cut/Fill Heatmap</b><br>"
        f"<sup>Red = Cut | Blue = Fill | Natural orientation: {result.original_azimuth} | "
        f"Design orientation: {result.design_azimuth}</sup>"
    )
    figure.update_layout(
        title=dict(text=title, x=0.5, y=0.95),
        xaxis=dict(title="X1 East-West (m)"),
        yaxis=dict(title="X2 North-South (m)", scaleanchor="x", scaleratio=1),
        autosize=True,
        margin=dict(l=40, r=40, t=160, b=70),
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#F5F7FA",
        hoverlabel=dict(bgcolor="#111827", font_color="white", bordercolor="white"),
        legend=dict(
            title="<b>Layers</b>",
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    return figure


def create_volume_histogram(result: OptimizationResult) -> go.Figure:
    deviations = result.deviation_grid.flatten()
    resolution = float(abs(result.grid_x1[0, 1] - result.grid_x1[0, 0]))
    cell_area = resolution**2
    lower = min(float(deviations.min()), -0.01)
    upper = max(float(deviations.max()), 0.01)
    negative_bins = np.linspace(lower, 0, 35)
    positive_bins = np.linspace(0, upper, 35)
    fill_volume = np.array(
        [
            np.sum(np.abs(deviations[(deviations >= negative_bins[i]) & (deviations < negative_bins[i + 1])]))
            * cell_area
            for i in range(len(negative_bins) - 1)
        ]
    )
    cut_volume = np.array(
        [
            np.sum(deviations[(deviations > positive_bins[i]) & (deviations <= positive_bins[i + 1])]) * cell_area
            for i in range(len(positive_bins) - 1)
        ]
    )
    negative_centers = (negative_bins[:-1] + negative_bins[1:]) / 2
    positive_centers = (positive_bins[:-1] + positive_bins[1:]) / 2
    negative_width = float(negative_bins[1] - negative_bins[0])
    positive_width = float(positive_bins[1] - positive_bins[0])

    total = result.cut_volume_m3 + result.fill_volume_m3
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]], column_widths=[0.72, 0.28])
    figure.add_trace(
        go.Bar(
            x=negative_centers,
            y=fill_volume,
            name="Fill",
            marker=dict(color="#2563EB", line=dict(color="white", width=0.2)),
            width=negative_width,
            hovertemplate="<b>Deviation:</b> %{x:.2f} m<br><b>Fill volume:</b> %{y:.2f} m3<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=positive_centers,
            y=cut_volume,
            name="Cut",
            marker=dict(color="#DC2626", line=dict(color="white", width=0.2)),
            width=positive_width,
            hovertemplate="<b>Deviation:</b> %{x:.2f} m<br><b>Cut volume:</b> %{y:.2f} m3<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_vline(x=0, line_dash="solid", line_color="#111827", line_width=2, row=1, col=1)
    figure.add_trace(
        go.Pie(
            labels=["Cut", "Fill"],
            values=[result.cut_volume_m3, result.fill_volume_m3],
            marker=dict(colors=["#DC2626", "#2563EB"]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.2f} m3<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        title=dict(
            text=f"<b>{result.area_name} - Volume Distribution by Depth</b><br><sup>Total: {total:,.2f} m3</sup>",
            x=0.5,
            y=0.95,
        ),
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#F5F7FA",
        xaxis_title="Design deviation (m)",
        yaxis_title="Volume (m3)",
        margin=dict(l=40, r=40, t=140, b=70),
        bargap=0.0,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return figure


def create_cost_waterfall_chart(result: OptimizationResult) -> go.Figure:
    cost = result.cost_breakdown
    figure = go.Figure(
        go.Waterfall(
            name="Cost buildup",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Excavation", "Fill", "Loading", "Transport", "Total"],
            y=[
                cost.excavation_cost,
                cost.fill_cost,
                cost.loading_cost,
                cost.transport_cost,
                cost.total_cost,
            ],
            connector={"line": {"color": "rgba(17,24,39,0.55)"}},
            increasing={"marker": {"color": "#2563EB"}},
            totals={"marker": {"color": "#111827"}},
            text=[
                f"TL {cost.excavation_cost:,.0f}",
                f"TL {cost.fill_cost:,.0f}",
                f"TL {cost.loading_cost:,.0f}",
                f"TL {cost.transport_cost:,.0f}",
                f"TL {cost.total_cost:,.0f}",
            ],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>TL %{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=dict(
            text=(
                f"<b>{result.area_name} - Official Cost Waterfall</b><br>"
                f"<sup>Components accumulate to the total site budget. External handling is based on net imbalance.</sup>"
            ),
            x=0.5,
        ),
        yaxis_title="Cost (TL)",
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#F5F7FA",
        margin=dict(l=60, r=40, t=120, b=70),
        showlegend=False,
    )
    return figure


def create_cost_sensitivity_chart(result: OptimizationResult) -> go.Figure:
    sensitivity = result.sensitivity
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=sensitivity.slopes,
            y=sensitivity.fill_volumes,
            name="Fill volume",
            marker_color="#2563EB",
            opacity=0.3,
            hovertemplate="<b>Fill volume:</b> %{y:,.2f} m3<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=sensitivity.slopes,
            y=sensitivity.cut_volumes,
            name="Cut volume",
            marker_color="#DC2626",
            opacity=0.3,
            hovertemplate="<b>Cut volume:</b> %{y:,.2f} m3<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=sensitivity.slopes,
            y=sensitivity.excavation_costs,
            mode="lines",
            stackgroup="cost",
            name="Excavation cost",
            line=dict(width=0),
            fillcolor="rgba(220,38,38,0.65)",
            hovertemplate="<b>Excavation:</b> TL %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=sensitivity.slopes,
            y=sensitivity.fill_costs,
            mode="lines",
            stackgroup="cost",
            name="Fill cost",
            line=dict(width=0),
            fillcolor="rgba(37,99,235,0.65)",
            hovertemplate="<b>Fill:</b> TL %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=sensitivity.slopes,
            y=sensitivity.loading_costs,
            mode="lines",
            stackgroup="cost",
            name="Loading cost",
            line=dict(width=0),
            fillcolor="rgba(245,158,11,0.65)",
            hovertemplate="<b>Loading:</b> TL %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=sensitivity.slopes,
            y=sensitivity.transport_costs,
            mode="lines",
            stackgroup="cost",
            name="Transport cost",
            line=dict(width=0),
            fillcolor="rgba(16,185,129,0.65)",
            hovertemplate="<b>Transport:</b> TL %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    smooth_slopes = np.linspace(0.5, 2.0, 100)
    poly_a, poly_b, poly_c = sensitivity.polynomial_coefficients
    smooth_costs = poly_a * smooth_slopes**2 + poly_b * smooth_slopes + poly_c
    figure.add_trace(
        go.Scatter(
            x=smooth_slopes,
            y=smooth_costs,
            mode="lines",
            name="Quadratic ML cost curve",
            line=dict(color="black", width=3, dash="dash"),
            hovertemplate="<b>Modeled cost:</b> TL %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=[sensitivity.theoretical_optimum_slope],
            y=[sensitivity.theoretical_minimum_cost],
            mode="markers+text",
            name="Theoretical optimum",
            marker=dict(color="#FACC15", size=15, line=dict(color="black", width=2), symbol="star"),
            text=[f"Optimum: {sensitivity.theoretical_optimum_slope:.2f}%"],
            textposition="top center",
            hovertemplate="<b>Optimum slope:</b> %{x:.2f}%<br><b>Minimum cost:</b> TL %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title=dict(
            text=f"<b>{result.area_name} - ML Cost Sensitivity</b><br><sup>Coarse-to-fine slope sweep with quadratic regression.</sup>",
            x=0.5,
        ),
        barmode="stack",
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#F5F7FA",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=130, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    figure.update_yaxes(title_text="Volume (m3)", secondary_y=False)
    figure.update_yaxes(title_text="Cost (TL)", secondary_y=True, showgrid=False)
    figure.update_xaxes(title_text="Design slope (%)")
    return figure


def create_mass_haul_flow_chart(result: OptimizationResult) -> go.Figure:
    max_deviation = float(np.abs(result.deviation_grid).max()) or 0.01
    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=result.grid_x1[0, :],
            y=result.grid_x2[:, 0],
            z=result.deviation_grid,
            colorscale="RdBu_r",
            zmin=-max_deviation,
            zmax=max_deviation,
            colorbar=dict(title="<b>Deviation (m)</b>", tickformat=".2f"),
            contours=dict(coloring="heatmap", showlabels=False),
            line=dict(width=0),
            hovertemplate="<b>X1:</b> %{x:.1f} m<br><b>X2:</b> %{y:.1f} m<br><b>Deviation:</b> %{z:.3f} m<extra></extra>",
            name="Cut/Fill Base",
        )
    )

    mass_haul = result.mass_haul
    if mass_haul and mass_haul.transfers:
        top_transfers = sorted(mass_haul.transfers, key=lambda transfer: transfer.volume_m3, reverse=True)[:60]
        annotations = []
        midpoint_x = []
        midpoint_y = []
        hover_text = []
        max_volume = max(transfer.volume_m3 for transfer in top_transfers)

        for transfer in top_transfers:
            width = 1 + 5 * (transfer.volume_m3 / max_volume)
            annotations.append(
                dict(
                    x=transfer.target_x,
                    y=transfer.target_y,
                    ax=transfer.source_x,
                    ay=transfer.source_y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=width,
                    arrowcolor="rgba(17,24,39,0.78)",
                    opacity=0.75,
                )
            )
            midpoint_x.append((transfer.source_x + transfer.target_x) / 2)
            midpoint_y.append((transfer.source_y + transfer.target_y) / 2)
            hover_text.append(
                f"Volume: {transfer.volume_m3:,.2f} m3<br>"
                f"Distance: {transfer.distance_m:.2f} m<br>"
                f"From ({transfer.source_x:.1f}, {transfer.source_y:.1f}) to "
                f"({transfer.target_x:.1f}, {transfer.target_y:.1f})"
            )

        figure.add_trace(
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers",
                marker=dict(size=8, color="#111827", opacity=0.2),
                text=hover_text,
                hovertemplate="%{text}<extra>Internal transfer</extra>",
                name="Internal Transfer",
            )
        )
    else:
        annotations = []

    title = "<b>{area} - Greedy Internal Mass-Haul Flow</b><br><sup>{summary}</sup>".format(
        area=result.area_name,
        summary=(
            f"Matched {mass_haul.total_internal_volume_m3:,.2f} m3 at "
            f"{mass_haul.average_distance_m:.2f} m average distance"
            if mass_haul
            else "No internal transfer result available"
        ),
    )
    figure.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="X1 East-West (m)", range=[0, 100]),
        yaxis=dict(title="X2 North-South (m)", scaleanchor="x", scaleratio=1, range=[0, 100]),
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#F5F7FA",
        margin=dict(l=50, r=40, t=120, b=70),
        annotations=annotations,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return figure


def create_summary_png(result: OptimizationResult, frame: pd.DataFrame, output_directory: Path) -> Path:
    z_min = float(frame["Y"].min()) - 1.0
    z_max = float(frame["Y"].max()) + 1.0
    max_deviation = float(np.abs(result.deviation_grid).max()) or 0.01
    norm = matplotlib.colors.Normalize(vmin=-max_deviation, vmax=max_deviation)
    figure = plt.figure(figsize=(21, 6), dpi=130, facecolor="#F5F7FA")
    figure.suptitle(
        f"{result.area_name} - Total Earthwork: {result.total_earthwork_m3:,.0f} m3 "
        f"(Cut: {result.cut_volume_m3:,.0f} | Fill: {result.fill_volume_m3:,.0f})\n"
        f"Natural orientation: {result.original_azimuth} | Design: {result.design_azimuth}",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        x=0.5,
        y=0.98,
        ha="center",
    )

    axis_3d = figure.add_subplot(131, projection="3d")
    axis_3d.set_box_aspect((1.2, 1.0, 0.2))
    axis_3d.set_zlim(z_min, z_max)
    axis_3d.set_facecolor("#F5F7FA")
    axis_3d.zaxis.set_major_locator(plt.MaxNLocator(4))
    axis_3d.plot_surface(
        result.grid_x1,
        result.grid_x2,
        result.natural_surface,
        facecolors=plt.cm.RdBu_r(norm(result.deviation_grid)),
        alpha=0.9,
        rstride=1,
        cstride=1,
        shade=True,
    )
    axis_3d.scatter(frame["X1"].values, frame["X2"].values, frame["Y"].values, color="#DC2626", s=15, edgecolor="white")
    axis_3d.set_title("RBF Terrain Surface")
    axis_3d.view_init(elev=30, azim=-135)

    axis_hist = figure.add_subplot(132)
    axis_hist.set_facecolor("#F5F7FA")
    deviations = result.deviation_grid.flatten()
    resolution = float(abs(result.grid_x1[0, 1] - result.grid_x1[0, 0]))
    cell_area = resolution**2
    negative_bins = np.linspace(deviations.min(), 0, 35)
    positive_bins = np.linspace(0, deviations.max(), 35)
    fill_volume = np.array(
        [
            np.sum(np.abs(deviations[(deviations >= negative_bins[i]) & (deviations < negative_bins[i + 1])]))
            * cell_area
            for i in range(len(negative_bins) - 1)
        ]
    )
    cut_volume = np.array(
        [
            np.sum(deviations[(deviations > positive_bins[i]) & (deviations <= positive_bins[i + 1])]) * cell_area
            for i in range(len(positive_bins) - 1)
        ]
    )
    axis_hist.bar(
        (negative_bins[:-1] + negative_bins[1:]) / 2,
        fill_volume,
        width=float(negative_bins[1] - negative_bins[0]),
        color="#2563EB",
        edgecolor="white",
        linewidth=0.3,
        label="Fill",
    )
    axis_hist.bar(
        (positive_bins[:-1] + positive_bins[1:]) / 2,
        cut_volume,
        width=float(positive_bins[1] - positive_bins[0]),
        color="#DC2626",
        edgecolor="white",
        linewidth=0.3,
        label="Cut",
    )
    axis_hist.axvline(0, color="#111827", linestyle="-", linewidth=2)
    axis_hist.set_title("Volume Distribution by Depth")
    axis_hist.set_xlabel("Design deviation (m)")
    axis_hist.set_ylabel("Volume (m3)")
    axis_hist.legend()
    axis_hist.grid(axis="y", alpha=0.3)

    axis_map = figure.add_subplot(133)
    axis_map.set_facecolor("#F5F7FA")
    filled = axis_map.contourf(
        result.grid_x1,
        result.grid_x2,
        result.deviation_grid,
        levels=30,
        cmap="RdBu_r",
        vmin=-max_deviation,
        vmax=max_deviation,
    )
    plt.colorbar(filled, ax=axis_map, label="Design difference (m)")
    contours = axis_map.contour(result.grid_x1, result.grid_x2, result.natural_surface, levels=10, colors="black", linewidths=0.6)
    axis_map.clabel(contours, inline=True, fmt="%.1f m", fontsize=8, colors="black")
    axis_map.scatter(frame["X1"].values, frame["X2"].values, c="#DC2626", s=15, edgecolors="white", linewidths=0.5)
    axis_map.set_title("Cut/Fill Map with Natural Contours")

    plt.tight_layout(pad=3.0)
    output_path = output_directory / f"{result.area_name.replace(' ', '_')}_Summary.png"
    figure.savefig(str(output_path), bbox_inches="tight", facecolor=figure.get_facecolor(), edgecolor="none")
    plt.close(figure)
    return output_path


def generate_all_outputs(
    result: OptimizationResult,
    frame: pd.DataFrame,
    output_directory: Optional[Path] = None,
    open_in_browser: bool = False,
    all_results: dict[str, OptimizationResult] | None = None,
    jury_results_url: str = DEFAULT_JURY_RESULTS_URL,
) -> dict[str, Path]:
    if output_directory is None:
        output_directory = Path.cwd() / "outputs"

    label = result.area_name.replace(" ", "_")
    target_directory = Path(output_directory) / label
    target_directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    paths_and_figures = {
        "3D_Model": (target_directory / f"{label}_3D_Model.html", create_3d_model(result, frame)),
        "Cut_Fill_Heatmap": (target_directory / f"{label}_Cut_Fill_Heatmap.html", create_cut_fill_heatmap(result, frame)),
        "Volume_Histogram": (target_directory / f"{label}_Volume_Histogram.html", create_volume_histogram(result)),
        "Cost_Waterfall": (target_directory / f"{label}_Cost_Waterfall.html", create_cost_waterfall_chart(result)),
        "Cost_Sensitivity": (target_directory / f"{label}_Cost_Sensitivity.html", create_cost_sensitivity_chart(result)),
        "Mass_Haul_Flow": (target_directory / f"{label}_Mass_Haul_Flow.html", create_mass_haul_flow_chart(result)),
    }

    for key, (path, figure) in paths_and_figures.items():
        save_html(figure, path, open_in_browser=open_in_browser and key == "3D_Model")
        files[key] = path

    summary_path = create_summary_png(result, frame, target_directory)
    files["Summary"] = summary_path

    report_path = create_word_report(
        result,
        frame,
        target_directory,
        all_results=all_results,
        summary_png_path=summary_path,
        jury_results_url=jury_results_url,
    )
    files["APA7_Report"] = report_path

    print("  All HTML figures, summary image, QR code, and Word report were saved.")
    return files
