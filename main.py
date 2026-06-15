from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
from pathlib import Path
import sys

from core.data_loader import read_all_areas
from core.optimization import TARGET_SLOPE_PERCENT, OptimizationResult, analyze_site_area
from core.reporting import DEFAULT_JURY_RESULTS_URL
from core.visualization import generate_all_outputs


class LoggerWriter:

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.write(f"\n[{timestamp}] NEW SITE ANALYSIS SESSION STARTED\n")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def print_header(text: str) -> None:
    print(f"\n{'=' * 72}\n {text}\n{'=' * 72}\n")


def print_result_summary(result: OptimizationResult) -> None:
    cost = result.cost_breakdown
    mass_haul = result.mass_haul
    print(f"[{result.area_name}] Optimization Results")
    print(
        f"  - Natural slope       : {result.original_slope_percent:.2f}% "
        f"({'requires optimization' if result.original_slope_percent > TARGET_SLOPE_PERCENT else 'already compliant'})"
    )
    print(f"  - Design slope        : {result.design_slope_percent:.2f}%")
    print(f"  - Cut volume          : {result.cut_volume_m3:,.2f} m3")
    print(f"  - Fill volume         : {result.fill_volume_m3:,.2f} m3")
    print(f"  - Total earthwork     : {result.total_earthwork_m3:,.2f} m3")
    print(f"  - Natural orientation : {result.original_azimuth} ({result.original_quadrant})")
    print(f"  - Design orientation  : {result.design_azimuth} ({result.design_quadrant})")
    print(f"  - RBF model           : {result.rbf_model.upper()} (LOO-CV RMSE: {result.rbf_rmse:.3f} m)")
    print(f"  - Trend coefficients  : a={result.regression.intercept:.5f}, "
          f"b={result.regression.x1_coefficient:.5f}, c={result.regression.x2_coefficient:.5f}")
    print("  - Official 2026 Cost Breakdown")
    print(f"      Excavation        : TL {cost.excavation_cost:,.2f}")
    print(f"      Fill              : TL {cost.fill_cost:,.2f}")
    print(f"      Loading           : TL {cost.loading_cost:,.2f}")
    print(f"      Transport         : TL {cost.transport_cost:,.2f}")
    print(f"      Total             : TL {cost.total_cost:,.2f}")
    print(
        f"  - ML optimum slope    : {result.sensitivity.theoretical_optimum_slope:.2f}% "
        f"(estimated minimum: TL {result.sensitivity.theoretical_minimum_cost:,.2f})"
    )
    if mass_haul:
        print(
            f"  - Internal mass haul  : {mass_haul.total_internal_volume_m3:,.2f} m3 matched "
            f"at {mass_haul.average_distance_m:.2f} m average distance"
        )
    print(f"  - Balance status      : {result.balance_warning}")
    print(f"  - Geotechnical status : {result.geotechnical_warning}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topographic optimization and geological engineering design pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one named area")
    analyze_parser.add_argument("--file", default="area_optimization_automation.xlsx")
    analyze_parser.add_argument("--area", default="Area A")
    analyze_parser.add_argument("--resolution", type=float, default=0.25)
    analyze_parser.add_argument("--segment-size", type=float, default=10.0)

    compare_parser = subparsers.add_parser("compare", help="Compare all areas by total earthwork")
    compare_parser.add_argument("--file", default="area_optimization_automation.xlsx")
    compare_parser.add_argument("--resolution", type=float, default=1.0)
    compare_parser.add_argument("--segment-size", type=float, default=10.0)

    visualize_parser = subparsers.add_parser("visualize", help="Generate outputs for one named area")
    visualize_parser.add_argument("--file", default="area_optimization_automation.xlsx")
    visualize_parser.add_argument("--area", default="Area A")
    visualize_parser.add_argument("--resolution", type=float, default=0.25)
    visualize_parser.add_argument("--segment-size", type=float, default=10.0)
    visualize_parser.add_argument("--jury-url", default=DEFAULT_JURY_RESULTS_URL)

    auto_parser = subparsers.add_parser("auto", help="Analyze all areas and generate final outputs")
    auto_parser.add_argument("--file", default="area_optimization_automation.xlsx")
    auto_parser.add_argument("--resolution", type=float, default=0.25)
    auto_parser.add_argument("--segment-size", type=float, default=10.0)
    auto_parser.add_argument("--jury-url", default=DEFAULT_JURY_RESULTS_URL)

    return parser


def analyze_all_areas(
    areas,
    resolution: float,
    segment_size: float,
) -> dict[str, OptimizationResult]:
    print("  [SYSTEM] Multiprocessing is active. Site areas are being evaluated in parallel.\n")
    results: dict[str, OptimizationResult] = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            name: executor.submit(analyze_site_area, frame, name, resolution, segment_size)
            for name, frame in areas.items()
        }
        for name in areas.keys():
            result = futures[name].result()
            results[name] = result
            print_result_summary(result)
    return results


def main() -> None:
    sys.stdout = LoggerWriter("optimization.log")
    parser = build_parser()
    args = parser.parse_args()
    output_directory = Path("outputs")

    try:
        print("Reading input data...")
        areas = read_all_areas(args.file)

        if args.command == "analyze":
            print_header(f"SITE AREA ANALYSIS - {args.area}")
            result = analyze_site_area(areas[args.area], args.area, args.resolution, args.segment_size)
            print_result_summary(result)

        elif args.command == "compare":
            print_header("SITE AREA COMPARISON")
            results = analyze_all_areas(areas, args.resolution, args.segment_size)
            best_area = min(results, key=lambda key: results[key].total_earthwork_m3)
            print(f"Best area by total earthwork: {best_area}")

        elif args.command == "visualize":
            print_header(f"OUTPUT GENERATION - {args.area}")
            result = analyze_site_area(areas[args.area], args.area, args.resolution, args.segment_size)
            print_result_summary(result)
            generate_all_outputs(
                result,
                areas[args.area],
                output_directory=output_directory,
                open_in_browser=False,
                all_results=None,
                jury_results_url=args.jury_url,
            )

        elif args.command == "auto":
            print_header("AUTOMATIC OPTIMIZATION AND OUTPUT GENERATION")
            results = analyze_all_areas(areas, args.resolution, args.segment_size)
            best_area = min(results, key=lambda key: results[key].total_earthwork_m3)
            best_result = results[best_area]
            print("=" * 72)
            print("ENGINEERING DECISION")
            print(
                "After all areas are constrained to the maximum 2.0% design slope, "
                "the optimum area is the one with the lowest total earthwork volume."
            )
            print(f"OPTIMUM AREA: {best_area} ({best_result.total_earthwork_m3:,.2f} m3 total earthwork)")
            print("=" * 72)
            print("Generating detailed outputs for every analyzed area...")

            for area_name, result in results.items():
                generate_all_outputs(
                    result,
                    areas[area_name],
                    output_directory=output_directory,
                    open_in_browser=False,
                    all_results=results,
                    jury_results_url=args.jury_url,
                )

    except KeyError as exc:
        print(f"\n[CRITICAL ERROR] The requested area was not found in the input workbook: {exc}")
        print("Check the --area argument and the workbook sheet names.")
    except PermissionError as exc:
        print(f"\n[CRITICAL ERROR] File access was denied: {exc}")
        print("Close any open Excel, Word, or HTML output files and retry.")
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
