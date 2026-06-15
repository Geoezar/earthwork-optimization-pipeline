from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegressionResult:

    intercept: float
    x1_coefficient: float
    x2_coefficient: float
    mse: float
    rmse: float
    r2: float
    sse: float
    correlation: float
    slope_percent: float
    n: int
    sum_x1: float
    sum_x2: float
    sum_y: float
    sum_x1_sq: float
    sum_x2_sq: float
    sum_x1_x2: float
    sum_x1_y: float
    sum_x2_y: float
    observed_y: np.ndarray = field(repr=False)
    predicted_y: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    model: LinearRegression | None = field(repr=False, default=None)

    def __str__(self) -> str:
        return (
            f"  Coefficients : intercept = {self.intercept:.5f} | "
            f"x1 = {self.x1_coefficient:.5f} | x2 = {self.x2_coefficient:.5f}\n"
            f"  R2           : {self.r2:.4f}\n"
            f"  RMSE         : {self.rmse:.4f} m\n"
            f"  Natural slope: {self.slope_percent:.4f}%"
        )


def calculate_slope(x1_coefficient: float, x2_coefficient: float) -> float:
    return float(np.sqrt(x1_coefficient**2 + x2_coefficient**2) * 100)


def fit_plane(frame: pd.DataFrame) -> RegressionResult:
    features = frame[["X1", "X2"]].values.astype(float)
    observed_y = frame["Y"].values.astype(float)

    model = LinearRegression()
    model.fit(features, observed_y)
    sklearn_intercept = float(model.intercept_)
    sklearn_x1 = float(model.coef_[0])
    sklearn_x2 = float(model.coef_[1])

    design_matrix = np.column_stack((np.ones(len(observed_y)), features))
    normal_matrix = design_matrix.T @ design_matrix
    rhs = design_matrix.T @ observed_y

    try:
        numpy_coefficients = np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        numpy_coefficients = np.linalg.lstsq(design_matrix, observed_y, rcond=None)[0]

    numpy_intercept, numpy_x1, numpy_x2 = [float(value) for value in numpy_coefficients]

    if abs(sklearn_intercept - numpy_intercept) > 0.01 or abs(sklearn_x1 - numpy_x1) > 1e-4:
        print("  [WARNING] NumPy and scikit-learn plane coefficients diverged. Using the NumPy solution.")
        intercept, x1_coefficient, x2_coefficient = numpy_intercept, numpy_x1, numpy_x2
    else:
        intercept, x1_coefficient, x2_coefficient = sklearn_intercept, sklearn_x1, sklearn_x2

    predicted_y = intercept + x1_coefficient * features[:, 0] + x2_coefficient * features[:, 1]
    residuals = observed_y - predicted_y

    mse = float(mean_squared_error(observed_y, predicted_y))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(observed_y, predicted_y))
    sse = float(np.sum(residuals**2))
    correlation = float(np.corrcoef(observed_y, predicted_y)[0, 1])
    slope_percent = calculate_slope(x1_coefficient, x2_coefficient)

    return RegressionResult(
        intercept=intercept,
        x1_coefficient=x1_coefficient,
        x2_coefficient=x2_coefficient,
        mse=mse,
        rmse=rmse,
        r2=r2,
        sse=sse,
        correlation=correlation,
        slope_percent=slope_percent,
        n=len(frame),
        sum_x1=float(frame["X1"].sum()),
        sum_x2=float(frame["X2"].sum()),
        sum_y=float(frame["Y"].sum()),
        sum_x1_sq=float((frame["X1"] ** 2).sum()),
        sum_x2_sq=float((frame["X2"] ** 2).sum()),
        sum_x1_x2=float((frame["X1"] * frame["X2"]).sum()),
        sum_x1_y=float((frame["X1"] * frame["Y"]).sum()),
        sum_x2_y=float((frame["X2"] * frame["Y"]).sum()),
        observed_y=observed_y,
        predicted_y=predicted_y,
        residuals=residuals,
        model=model,
    )


def interpret_slope(slope_percent: float, limit: float = 2.0) -> Tuple[str, str]:
    if slope_percent <= limit:
        return "COMPLIANT", f"Slope {slope_percent:.2f}% <= {limit:.1f}%."
    return "EXCEEDS_LIMIT", f"Slope {slope_percent:.2f}% > {limit:.1f}%; optimization is required."
