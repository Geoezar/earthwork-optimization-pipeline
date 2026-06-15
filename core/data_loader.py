from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["X1", "X2", "Y"]
MIN_VALID_POINTS = 10
IQR_MULTIPLIER = 3.0


def _validate_required_columns(frame: pd.DataFrame, source_name: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def read_and_clean_file(file_path: str | Path, sheet_name: str | None = None, is_excel: bool = True) -> pd.DataFrame:
    path = Path(file_path)
    source_name = sheet_name or path.stem

    try:
        if is_excel:
            raw_frame = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        else:
            raw_frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {source_name}. Check the input file. Details: {exc}") from exc

    _validate_required_columns(raw_frame, source_name)

    frame = raw_frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna().reset_index(drop=True)
    initial_count = len(frame)

    for column in REQUIRED_COLUMNS:
        q1 = frame[column].quantile(0.25)
        q3 = frame[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        frame = frame[(frame[column] >= lower_bound) & (frame[column] <= upper_bound)]
        frame = frame.reset_index(drop=True)

    final_count = len(frame)
    removed_count = initial_count - final_count

    if final_count < MIN_VALID_POINTS:
        raise ValueError(
            f"{source_name}: only {final_count} valid survey points remain after QC. "
            f"At least {MIN_VALID_POINTS} points are required."
        )

    x1_min, x1_max = frame["X1"].min(), frame["X1"].max()
    x2_min, x2_max = frame["X2"].min(), frame["X2"].max()
    y_min, y_max = frame["Y"].min(), frame["Y"].max()

    print(f"\n  [{source_name}] Data Quality Control Summary:")
    print(f"  - File type          : {'Excel workbook' if is_excel else 'CSV file'}")
    print(f"  - Numeric rows       : {initial_count}")
    print(f"  - Removed outliers   : {removed_count} using IQR x {IQR_MULTIPLIER}")
    print(f"  - Valid points       : {final_count}")
    print(f"  - X1 range           : {x1_min:.2f} m to {x1_max:.2f} m")
    print(f"  - X2 range           : {x2_min:.2f} m to {x2_max:.2f} m")
    print(f"  - Elevation range    : {y_min:.2f} m to {y_max:.2f} m")

    if (frame["X1"] < 0).any() or (frame["X2"] < 0).any():
        print(f"  [WARNING] {source_name}: negative coordinates were detected.")

    return frame


def read_all_areas(file_path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    extension = path.suffix.lower()
    if extension == ".csv":
        return {path.stem: read_and_clean_file(path, path.stem, is_excel=False)}

    if extension not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported input format: {extension}. Use .xlsx, .xls, or .csv.")

    workbook = pd.ExcelFile(path, engine="openpyxl")
    areas: dict[str, pd.DataFrame] = {}

    for sheet_name in workbook.sheet_names:
        preview = pd.read_excel(path, sheet_name=sheet_name, nrows=1, engine="openpyxl")
        if all(column in preview.columns for column in REQUIRED_COLUMNS):
            areas[sheet_name] = read_and_clean_file(path, sheet_name, is_excel=True)

    if not areas:
        raise ValueError(f"No workbook sheets contain the required columns: {REQUIRED_COLUMNS}")

    return areas
