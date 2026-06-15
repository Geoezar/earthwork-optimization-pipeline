# Earthwork Optimization Pipeline

Earthwork Optimization Pipeline is a local Python project for geological engineering design. It reads survey points, checks data quality, models terrain, designs a slope-limited grading plane, calculates cut and fill volumes, estimates earthwork cost, and creates local engineering outputs.

The project is built for offline use. It does not include a web application, public API, or hosted execution system. Input datasets and generated reports are private local materials and are not part of the public repository.

## Main Value

This repository shows a reproducible engineering workflow for terrain and earthwork analysis. It connects survey data, numerical modeling, optimization, cost logic, and visual output generation in one command-line pipeline.

The pipeline can help with:

- checking survey tables before analysis,
- fitting a natural terrain trend plane,
- selecting a radial basis function terrain model,
- limiting the final design plane to a maximum 2.0 percent slope,
- calculating cut and fill quantities,
- estimating cost with unit-price logic,
- creating local HTML, PNG, Word, and QR outputs.

## Repository Structure

```text
.
|-- main.py
|-- core/
|   |-- data_loader.py
|   |-- regression.py
|   |-- optimization.py
|   |-- visualization.py
|   `-- reporting.py
|-- tests/
|-- .github/workflows/ci.yml
|-- pyproject.toml
|-- requirements.txt
|-- uv.lock
|-- ARCHITECTURE.md
|-- LICENSE
`-- README.md
```

## Processing Flow

```text
Excel or CSV input
  -> data quality control
  -> terrain trend regression
  -> slope-constrained design plane
  -> RBF terrain surface selection
  -> cut and fill volume calculation
  -> cost and mass-haul analysis
  -> local output generation
```

## Requirements

- Python 3.12 or newer
- `uv` for dependency and lockfile management
- A local Excel or CSV file with survey data

The main Python libraries are:

- pandas
- NumPy
- SciPy
- scikit-learn
- Plotly
- Matplotlib
- python-docx
- qrcode
- Pillow
- openpyxl

## Input Data Format

The repository does not publish a dataset. Provide your own local workbook or CSV file.

Each valid table must include these columns:

| Column | Meaning |
| --- | --- |
| `X1` | East-west or local X coordinate in meters |
| `X2` | North-south or local Y coordinate in meters |
| `Y` | Elevation in meters |

For Excel workbooks, each sheet with these three columns is treated as one candidate area. For CSV files, the file is treated as one area.

## Setup

Install dependencies with `uv`:

```powershell
uv sync
```

You can also install from `requirements.txt` if you do not use `uv`:

```powershell
python -m pip install -r requirements.txt
```

## Usage

Run a fast comparison for all valid workbook sheets:

```powershell
uv run python main.py compare --file path\to\survey.xlsx --resolution 1.0
```

Analyze one area:

```powershell
uv run python main.py analyze --file path\to\survey.xlsx --area "Area A" --resolution 1.0
```

Generate local outputs for one area:

```powershell
uv run python main.py visualize --file path\to\survey.xlsx --area "Area A" --resolution 0.25
```

Run the full local pipeline for every area:

```powershell
uv run python main.py auto --file path\to\survey.xlsx --resolution 0.25
```

## Generated Local Outputs

The program writes generated files to `outputs/<AreaName>/`. This folder is ignored by Git.

Typical output files include:

- interactive terrain HTML files,
- cut and fill heatmaps,
- volume histograms,
- cost waterfall charts,
- cost sensitivity charts,
- mass-haul flow maps,
- static summary PNG files,
- local Word reports,
- QR code images.

These files are execution results. They should be reviewed locally and should not be committed to the public repository.

## Testing

Run the test suite with:

```powershell
uv run --with pytest pytest -q
```

The tests check core engineering behavior, including slope control, transport cost switching, cost totals, mass-haul volume conservation, and CSV data quality control.

## Customization

You can adjust execution behavior through CLI arguments:

- `--file` sets the input workbook or CSV path.
- `--area` selects one area by sheet name.
- `--resolution` controls grid spacing for the terrain and volume calculation.
- `--segment-size` controls mass-haul segment aggregation.
- `--jury-url` sets the QR target for locally generated reports.

Use lower resolution values only when you need final local outputs. Fine grids increase memory use and runtime.

## Security and Publication Rules

This repository is designed for public source code only. Do not commit:

- private survey files,
- generated outputs,
- reports or report source files,
- presentation files,
- runtime logs,
- browser profile exports,
- local environment files.

Keep real project data and generated deliverables outside the repository.

## License

This project is released under the MIT License.
