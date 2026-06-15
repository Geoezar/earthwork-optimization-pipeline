from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.data_loader import read_and_clean_file
from core.optimization import (
    CSB_EXCAVATION,
    CSB_FILL,
    CSB_LOADING,
    TARGET_SLOPE_PERCENT,
    calculate_cost_breakdown,
    calculate_mass_haul_flow,
    calculate_transport_unit_cost,
    optimize_slope,
)
from core.regression import calculate_slope, fit_plane


def test_optimize_slope_respects_limit_and_centroid() -> None:
    frame = pd.DataFrame(
        {
            "X1": [0.0, 50.0, 100.0, 0.0, 50.0, 100.0, 0.0, 50.0, 100.0, 25.0, 75.0],
            "X2": [0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 25.0, 75.0],
        }
    )
    frame["Y"] = 10.0 + 0.05 * frame["X1"] + 0.03 * frame["X2"]
    regression = fit_plane(frame)

    intercept, x1_coef, x2_coef, design_slope = optimize_slope(regression, frame)

    assert design_slope <= TARGET_SLOPE_PERCENT + 1e-9
    assert math.isclose(calculate_slope(x1_coef, x2_coef), design_slope)

    centroid_design_y = intercept + x1_coef * frame["X1"].mean() + x2_coef * frame["X2"].mean()
    assert math.isclose(centroid_design_y, frame["Y"].mean(), rel_tol=0, abs_tol=1e-9)


def test_transport_formula_switches_at_ten_kilometers() -> None:
    below = calculate_transport_unit_cost(10.0)
    above = calculate_transport_unit_cost(10.1)

    assert below > 0
    assert above > 0
    assert not math.isclose(below, above)


def test_cost_breakdown_totals_components() -> None:
    cut = 100.0
    fill = 60.0
    cost = calculate_cost_breakdown(cut, fill, transport_distance_km=5.0)
    expected_loading = abs(cut - fill) * CSB_LOADING

    assert math.isclose(cost.excavation_cost, cut * CSB_EXCAVATION)
    assert math.isclose(cost.fill_cost, fill * CSB_FILL)
    assert math.isclose(cost.loading_cost, expected_loading)
    assert math.isclose(
        cost.total_cost,
        cost.excavation_cost + cost.fill_cost + cost.loading_cost + cost.transport_cost,
    )


def test_mass_haul_conserves_available_internal_volume() -> None:
    grid_x1, grid_x2 = np.meshgrid(np.array([0.0, 10.0]), np.array([0.0, 10.0]))
    deviation = np.array([[2.0, -1.0], [1.0, -3.0]])

    result = calculate_mass_haul_flow(
        grid_x1,
        grid_x2,
        deviation,
        resolution=1.0,
        segment_size_m=5.0,
        minimum_cell_volume_m3=0.0,
        minimum_transfer_volume_m3=0.1,
    )

    total_cut = 3.0
    total_fill = 4.0
    assert math.isclose(result.total_internal_volume_m3, min(total_cut, total_fill))
    assert math.isclose(result.unmatched_cut_m3, 0.0)
    assert math.isclose(result.unmatched_fill_m3, 1.0)
    assert all(transfer.volume_m3 > 0 for transfer in result.transfers)


def test_csv_loader_applies_quality_control(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "X1": list(range(11)) + [1000],
            "X2": list(range(11)) + [1000],
            "Y": [10 + value * 0.1 for value in range(11)] + [5000],
        }
    )
    csv_path = tmp_path / "survey.csv"
    frame.to_csv(csv_path, index=False)

    cleaned = read_and_clean_file(csv_path, "survey", is_excel=False)

    assert len(cleaned) == 11
    assert cleaned["Y"].max() < 5000
