# Architecture

Earthwork Optimization Pipeline is a local command-line system for geological engineering design. The project separates data loading, regression, optimization, visualization, and report generation into small Python modules.

## System Flow

```text
Input workbook or CSV
  -> data quality control
  -> regression trend plane
  -> centroid-locked slope optimization
  -> RBF terrain modeling by leave-one-out cross-validation
  -> cut and fill volume integration
  -> cost calculation
  -> cost sensitivity analysis
  -> internal mass-haul routing
  -> local output files
```

## Modules

- `core/data_loader.py` reads Excel or CSV files, validates `X1`, `X2`, and `Y`, coerces numeric values, removes IQR x 3.0 outliers, and returns clean DataFrames.
- `core/regression.py` fits the natural trend plane with scikit-learn and a NumPy matrix cross-check.
- `core/optimization.py` applies slope control, RBF model selection, volume integration, cost modeling, sensitivity analysis, geological orientation, and mass-haul routing.
- `core/visualization.py` creates interactive Plotly HTML files and static PNG summaries in the local output directory.
- `core/reporting.py` creates local Word reports and QR codes.
- `main.py` provides the command-line interface and parallel execution control.

## Design Rules

- The system runs locally and offline.
- The public repository contains source code, tests, configuration, and documentation only.
- Input datasets and generated outputs stay outside version control.
- The computational modules use functions and dataclasses so that parallel execution remains simple.
- The maximum design slope is 2.0 percent unless the engineering requirement changes.
