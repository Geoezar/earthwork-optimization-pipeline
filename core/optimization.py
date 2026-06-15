from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import LeaveOneOut

from core.regression import RegressionResult, calculate_slope, fit_plane


AREA_X1_METERS = 100.0
AREA_X2_METERS = 100.0
TARGET_SLOPE_PERCENT = 2.0

CSB_EXCAVATION = 187.94
CSB_FILL = 257.78
CSB_LOADING = 39.66

KGM_DIFFICULTY_COEFFICIENT = 1.75
KGM_TRUCK_TRANSPORT_COEFFICIENT = 2365.0
DEFAULT_SOIL_DENSITY = 1.6
DEFAULT_TRANSPORT_DISTANCE_KM = 15.0


@dataclass
class CostBreakdown:
    excavation_cost: float
    fill_cost: float
    loading_cost: float
    transport_cost: float
    total_cost: float


@dataclass
class SensitivityResult:
    slopes: np.ndarray
    cut_volumes: np.ndarray
    fill_volumes: np.ndarray
    excavation_costs: np.ndarray
    fill_costs: np.ndarray
    loading_costs: np.ndarray
    transport_costs: np.ndarray
    total_costs: np.ndarray
    polynomial_coefficients: np.ndarray
    theoretical_optimum_slope: float
    theoretical_minimum_cost: float


@dataclass
class MassHaulTransfer:
    source_x: float
    source_y: float
    target_x: float
    target_y: float
    volume_m3: float
    distance_m: float


@dataclass
class MassHaulResult:
    segment_size_m: float
    transfers: list[MassHaulTransfer]
    total_internal_volume_m3: float
    weighted_distance_m3_m: float
    average_distance_m: float
    unmatched_cut_m3: float
    unmatched_fill_m3: float


@dataclass
class OptimizationResult:
    area_name: str
    original_intercept: float
    original_x1_coefficient: float
    original_x2_coefficient: float
    original_slope_percent: float
    design_intercept: float
    design_x1_coefficient: float
    design_x2_coefficient: float
    design_slope_percent: float
    original_azimuth: str
    original_quadrant: str
    design_azimuth: str
    design_quadrant: str
    geotechnical_warning: str
    cut_volume_m3: float
    fill_volume_m3: float
    total_earthwork_m3: float
    grid_x1: np.ndarray = field(repr=False)
    grid_x2: np.ndarray = field(repr=False)
    natural_surface: np.ndarray = field(repr=False)
    design_surface: np.ndarray = field(repr=False)
    deviation_grid: np.ndarray = field(repr=False)
    regression: RegressionResult = field(repr=False)
    sensitivity: SensitivityResult | None = field(default=None, repr=False)
    balance_warning: str = field(default="", repr=False)
    cost_breakdown: CostBreakdown | None = field(default=None, repr=False)
    rbf_model: str = field(default="")
    rbf_rmse: float = field(default=0.0)
    mass_haul: MassHaulResult | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return (
            f"  Design slope      : {self.design_slope_percent:.4f}%\n"
            f"  Cut volume        : {self.cut_volume_m3:,.2f} m3\n"
            f"  Fill volume       : {self.fill_volume_m3:,.2f} m3\n"
            f"  Total earthwork   : {self.total_earthwork_m3:,.2f} m3"
        )


def calculate_transport_unit_cost(
    distance_km: float = DEFAULT_TRANSPORT_DISTANCE_KM,
    density: float = DEFAULT_SOIL_DENSITY,
) -> float:
    if distance_km > 10.0:
        return (
            1.25
            * KGM_DIFFICULTY_COEFFICIENT
            * KGM_TRUCK_TRANSPORT_COEFFICIENT
            * (0.0007 * distance_km + 0.01)
            * density
        )

    distance_m = distance_km * 1000.0
    return (
        1.25
        * KGM_DIFFICULTY_COEFFICIENT
        * 0.00017
        * KGM_TRUCK_TRANSPORT_COEFFICIENT
        * math.sqrt(distance_m)
        * density
    )


def optimize_slope(
    regression: RegressionResult,
    frame: pd.DataFrame,
    target_percent: float = TARGET_SLOPE_PERCENT,
) -> Tuple[float, float, float, float]:
    if regression.slope_percent <= target_percent:
        return (
            regression.intercept,
            regression.x1_coefficient,
            regression.x2_coefficient,
            regression.slope_percent,
        )

    scale_factor = target_percent / regression.slope_percent
    design_x1 = regression.x1_coefficient * scale_factor
    design_x2 = regression.x2_coefficient * scale_factor

    mean_x1 = float(frame["X1"].mean())
    mean_x2 = float(frame["X2"].mean())
    mean_y = float(frame["Y"].mean())
    design_intercept = mean_y - design_x1 * mean_x1 - design_x2 * mean_x2
    design_slope = calculate_slope(design_x1, design_x2)

    return design_intercept, design_x1, design_x2, design_slope


def calculate_geological_orientation(x1_coefficient: float, x2_coefficient: float) -> tuple[str, str]:
    if abs(x1_coefficient) < 1e-6 and abs(x2_coefficient) < 1e-6:
        return "Horizontal", "Horizontal"

    dip_direction = math.degrees(math.atan2(-x1_coefficient, -x2_coefficient))
    if dip_direction < 0:
        dip_direction += 360.0

    dip_amount = math.degrees(math.atan(math.sqrt(x1_coefficient**2 + x2_coefficient**2)))
    compass_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    compass_index = int(round((dip_direction % 360) / 45.0))
    dip_label = compass_labels[compass_index]

    azimuth = f"{int(round(dip_direction)):03d}/{int(round(dip_amount))}{dip_label}"
    strike = (dip_direction - 90) % 360

    if strike <= 90:
        quadrant_strike = f"N{int(round(strike))}E"
    elif strike <= 180:
        quadrant_strike = f"S{int(round(180 - strike))}E"
    elif strike <= 270:
        quadrant_strike = f"S{int(round(strike - 180))}W"
    else:
        quadrant_strike = f"N{int(round(360 - strike))}W"

    quadrant = f"{quadrant_strike}/{int(round(dip_amount))}{dip_label}"
    return azimuth, quadrant


def find_best_rbf_surface(
    frame: pd.DataFrame,
    resolution: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    points = frame[["X1", "X2"]].values.astype(float)
    values = frame["Y"].values.astype(float)

    model_candidates = [
        ("thin_plate_spline", None),
        ("gaussian", 0.01),
        ("gaussian", 0.05),
        ("gaussian", 0.1),
        ("multiquadric", 0.01),
        ("multiquadric", 0.05),
        ("multiquadric", 0.1),
    ]

    best_mse = float("inf")
    best_kernel = "thin_plate_spline"
    best_epsilon: float | None = None
    leave_one_out = LeaveOneOut()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kernel, epsilon in model_candidates:
            squared_error = 0.0
            successful_tests = 0

            for train_index, test_index in leave_one_out.split(points):
                train_points = points[train_index]
                test_points = points[test_index]
                train_values = values[train_index]
                test_values = values[test_index]

                try:
                    if epsilon is None:
                        model = RBFInterpolator(train_points, train_values, kernel=kernel)
                    else:
                        model = RBFInterpolator(train_points, train_values, kernel=kernel, epsilon=epsilon)
                    prediction = model(test_points)
                    squared_error += float((test_values[0] - prediction[0]) ** 2)
                    successful_tests += 1
                except Exception:
                    continue

            if successful_tests:
                mse = squared_error / successful_tests
                if mse < best_mse:
                    best_mse = mse
                    best_kernel = kernel
                    best_epsilon = epsilon

    if best_epsilon is None:
        final_model = RBFInterpolator(points, values, kernel=best_kernel)
        model_name = best_kernel
    else:
        final_model = RBFInterpolator(points, values, kernel=best_kernel, epsilon=best_epsilon)
        model_name = f"{best_kernel} (epsilon={best_epsilon})"

    x1_values = np.arange(0.0, AREA_X1_METERS + resolution, resolution)
    x2_values = np.arange(0.0, AREA_X2_METERS + resolution, resolution)
    grid_x1, grid_x2 = np.meshgrid(x1_values, x2_values)
    grid_points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    natural_surface = final_model(grid_points).reshape(grid_x1.shape)

    return grid_x1, grid_x2, natural_surface, model_name, float(np.sqrt(best_mse))


def calculate_volume(
    design_intercept: float,
    design_x1: float,
    design_x2: float,
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    natural_surface: np.ndarray,
    resolution: float = 0.25,
) -> Tuple[float, float, np.ndarray]:
    cell_area = resolution**2
    design_surface = design_intercept + design_x1 * grid_x1 + design_x2 * grid_x2
    deviation = natural_surface - design_surface
    cut_volume = float(np.sum(deviation[deviation > 0]) * cell_area)
    fill_volume = float(np.sum(np.abs(deviation[deviation < 0])) * cell_area)
    return cut_volume, fill_volume, design_surface


def calculate_cost_breakdown(
    cut_volume_m3: float,
    fill_volume_m3: float,
    transport_distance_km: float = DEFAULT_TRANSPORT_DISTANCE_KM,
) -> CostBreakdown:
    excavation_cost = cut_volume_m3 * CSB_EXCAVATION
    fill_cost = fill_volume_m3 * CSB_FILL
    net_external_volume = abs(cut_volume_m3 - fill_volume_m3)
    loading_cost = net_external_volume * CSB_LOADING
    transport_cost = net_external_volume * calculate_transport_unit_cost(transport_distance_km)
    total_cost = excavation_cost + fill_cost + loading_cost + transport_cost
    return CostBreakdown(excavation_cost, fill_cost, loading_cost, transport_cost, total_cost)


def run_sensitivity_analysis(
    regression: RegressionResult,
    frame: pd.DataFrame,
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    natural_surface: np.ndarray,
    resolution: float = 0.25,
) -> SensitivityResult:
    cache: dict[float, tuple[float, float, CostBreakdown]] = {}

    def calculate_for_slope(slope_value: float) -> float:
        rounded_slope = round(float(slope_value), 3)
        if rounded_slope not in cache:
            intercept, x1_coef, x2_coef, _ = optimize_slope(regression, frame, target_percent=rounded_slope)
            cut, fill, _ = calculate_volume(intercept, x1_coef, x2_coef, grid_x1, grid_x2, natural_surface, resolution)
            cache[rounded_slope] = (cut, fill, calculate_cost_breakdown(cut, fill))
        return cache[rounded_slope][2].total_cost

    coarse_slopes = np.arange(0.5, 2.05, 0.1)
    for slope in coarse_slopes:
        calculate_for_slope(slope)

    best_coarse_slope = min(coarse_slopes, key=lambda slope: cache[round(float(slope), 3)][2].total_cost)
    lower_bound = max(0.5, best_coarse_slope - 0.1)
    upper_bound = min(2.0, best_coarse_slope + 0.1)
    fine_slopes = np.arange(lower_bound, upper_bound + 0.005, 0.01)

    for slope in fine_slopes:
        calculate_for_slope(slope)

    slopes = np.array(sorted(cache.keys()))
    cut_values: list[float] = []
    fill_values: list[float] = []
    excavation_costs: list[float] = []
    fill_costs: list[float] = []
    loading_costs: list[float] = []
    transport_costs: list[float] = []
    total_costs: list[float] = []

    for slope in slopes:
        cut, fill, cost = cache[slope]
        cut_values.append(cut)
        fill_values.append(fill)
        excavation_costs.append(cost.excavation_cost)
        fill_costs.append(cost.fill_cost)
        loading_costs.append(cost.loading_cost)
        transport_costs.append(cost.transport_cost)
        total_costs.append(cost.total_cost)

    polynomial = np.polyfit(slopes, total_costs, 2)
    poly_a, poly_b, poly_c = polynomial
    if poly_a > 0:
        optimum_slope = -poly_b / (2 * poly_a)
    else:
        optimum_slope = float(slopes[np.argmin(total_costs)])
    optimum_slope = float(max(0.5, min(optimum_slope, 2.0)))
    minimum_cost = float(poly_a * optimum_slope**2 + poly_b * optimum_slope + poly_c)

    return SensitivityResult(
        slopes=slopes,
        cut_volumes=np.array(cut_values),
        fill_volumes=np.array(fill_values),
        excavation_costs=np.array(excavation_costs),
        fill_costs=np.array(fill_costs),
        loading_costs=np.array(loading_costs),
        transport_costs=np.array(transport_costs),
        total_costs=np.array(total_costs),
        polynomial_coefficients=polynomial,
        theoretical_optimum_slope=optimum_slope,
        theoretical_minimum_cost=minimum_cost,
    )


def _aggregate_mass_cells(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    deviation_grid: np.ndarray,
    resolution: float,
    segment_size_m: float,
    positive: bool,
    minimum_cell_volume_m3: float,
) -> list[dict[str, float]]:
    cell_area = resolution**2
    segment_map: dict[tuple[int, int], dict[str, float]] = {}
    mask = deviation_grid > 0 if positive else deviation_grid < 0

    x_values = grid_x1[mask].ravel()
    y_values = grid_x2[mask].ravel()
    deviations = deviation_grid[mask].ravel()
    volumes = np.abs(deviations) * cell_area

    for x_value, y_value, volume in zip(x_values, y_values, volumes):
        if volume < minimum_cell_volume_m3:
            continue
        key = (int(x_value // segment_size_m), int(y_value // segment_size_m))
        entry = segment_map.setdefault(key, {"volume": 0.0, "weighted_x": 0.0, "weighted_y": 0.0})
        entry["volume"] += float(volume)
        entry["weighted_x"] += float(x_value * volume)
        entry["weighted_y"] += float(y_value * volume)

    segments: list[dict[str, float]] = []
    for entry in segment_map.values():
        volume = entry["volume"]
        if volume <= 0:
            continue
        segments.append(
            {
                "x": entry["weighted_x"] / volume,
                "y": entry["weighted_y"] / volume,
                "remaining": volume,
            }
        )

    segments.sort(key=lambda item: item["remaining"], reverse=True)
    return segments


def calculate_mass_haul_flow(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    deviation_grid: np.ndarray,
    resolution: float,
    segment_size_m: float = 10.0,
    minimum_cell_volume_m3: float = 0.01,
    minimum_transfer_volume_m3: float = 1.0,
) -> MassHaulResult:
    cut_segments = _aggregate_mass_cells(
        grid_x1, grid_x2, deviation_grid, resolution, segment_size_m, True, minimum_cell_volume_m3
    )
    fill_segments = _aggregate_mass_cells(
        grid_x1, grid_x2, deviation_grid, resolution, segment_size_m, False, minimum_cell_volume_m3
    )

    transfers: list[MassHaulTransfer] = []

    for source in cut_segments:
        while source["remaining"] >= minimum_transfer_volume_m3:
            candidates = [target for target in fill_segments if target["remaining"] >= minimum_transfer_volume_m3]
            if not candidates:
                break

            target = min(
                candidates,
                key=lambda item: math.hypot(source["x"] - item["x"], source["y"] - item["y"]),
            )
            transfer_volume = min(source["remaining"], target["remaining"])
            if transfer_volume < minimum_transfer_volume_m3:
                break

            distance = math.hypot(source["x"] - target["x"], source["y"] - target["y"])
            transfers.append(
                MassHaulTransfer(
                    source_x=source["x"],
                    source_y=source["y"],
                    target_x=target["x"],
                    target_y=target["y"],
                    volume_m3=transfer_volume,
                    distance_m=distance,
                )
            )
            source["remaining"] -= transfer_volume
            target["remaining"] -= transfer_volume

    total_internal_volume = float(sum(transfer.volume_m3 for transfer in transfers))
    weighted_distance = float(sum(transfer.volume_m3 * transfer.distance_m for transfer in transfers))
    average_distance = weighted_distance / total_internal_volume if total_internal_volume > 0 else 0.0
    unmatched_cut = float(sum(segment["remaining"] for segment in cut_segments))
    unmatched_fill = float(sum(segment["remaining"] for segment in fill_segments))

    return MassHaulResult(
        segment_size_m=segment_size_m,
        transfers=transfers,
        total_internal_volume_m3=total_internal_volume,
        weighted_distance_m3_m=weighted_distance,
        average_distance_m=average_distance,
        unmatched_cut_m3=unmatched_cut,
        unmatched_fill_m3=unmatched_fill,
    )


def analyze_site_area(
    frame: pd.DataFrame,
    area_name: str,
    resolution: float = 0.25,
    mass_haul_segment_size_m: float = 10.0,
) -> OptimizationResult:
    regression = fit_plane(frame)
    design_intercept, design_x1, design_x2, design_slope = optimize_slope(regression, frame)
    grid_x1, grid_x2, natural_surface, rbf_model, rbf_rmse = find_best_rbf_surface(frame, resolution)
    cut, fill, design_surface = calculate_volume(
        design_intercept,
        design_x1,
        design_x2,
        grid_x1,
        grid_x2,
        natural_surface,
        resolution,
    )

    original_azimuth, original_quadrant = calculate_geological_orientation(
        regression.x1_coefficient,
        regression.x2_coefficient,
    )
    design_azimuth, design_quadrant = calculate_geological_orientation(design_x1, design_x2)
    cost_breakdown = calculate_cost_breakdown(cut, fill)
    sensitivity = run_sensitivity_analysis(regression, frame, grid_x1, grid_x2, natural_surface, resolution)

    deviation_grid = natural_surface - design_surface
    deviation_std = float(np.std(deviation_grid))
    max_abs_deviation = float(np.max(np.abs(deviation_grid)))

    if deviation_std > 1.5 or max_abs_deviation > 4.0:
        geotechnical_warning = (
            f"Geological risk flag (std: {deviation_std:.2f} m, max deviation: {max_abs_deviation:.2f} m): "
            "the terrain is not homogeneous. Field verification for faults, abrupt layer contacts, "
            "or rock blocks is required."
        )
    else:
        geotechnical_warning = (
            f"Geotechnical approval (std: {deviation_std:.2f} m): no major terrain anomaly was detected."
        )

    if fill > cut * 1.5:
        balance_warning = "Warning: fill demand is high; imported borrow material may be required."
    elif cut > fill * 1.5:
        balance_warning = "Warning: cut volume is high; surplus material disposal may be required."
    else:
        balance_warning = "Earthwork balance is acceptable; cut and fill volumes broadly offset each other."

    mass_haul = calculate_mass_haul_flow(
        grid_x1,
        grid_x2,
        deviation_grid,
        resolution,
        segment_size_m=mass_haul_segment_size_m,
    )

    return OptimizationResult(
        area_name=area_name,
        original_intercept=regression.intercept,
        original_x1_coefficient=regression.x1_coefficient,
        original_x2_coefficient=regression.x2_coefficient,
        original_slope_percent=regression.slope_percent,
        design_intercept=design_intercept,
        design_x1_coefficient=design_x1,
        design_x2_coefficient=design_x2,
        design_slope_percent=design_slope,
        original_azimuth=original_azimuth,
        original_quadrant=original_quadrant,
        design_azimuth=design_azimuth,
        design_quadrant=design_quadrant,
        geotechnical_warning=geotechnical_warning,
        cut_volume_m3=cut,
        fill_volume_m3=fill,
        total_earthwork_m3=cut + fill,
        cost_breakdown=cost_breakdown,
        grid_x1=grid_x1,
        grid_x2=grid_x2,
        natural_surface=natural_surface,
        design_surface=design_surface,
        deviation_grid=deviation_grid,
        regression=regression,
        sensitivity=sensitivity,
        balance_warning=balance_warning,
        rbf_model=rbf_model,
        rbf_rmse=rbf_rmse,
        mass_haul=mass_haul,
    )
