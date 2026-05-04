from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from src.astr502.data.paths import K2_AGES, KEPLER_AGES, OUTPUT_FIGS_DIR, OUTPUT_RESULTS_DIR
from src.astr502.data.utils import LoggingUtils


def _extract_first_float(raw: str | None) -> float | None:
    """Extract the first float from values like '[4.07 4.07]'."""
    if raw is None:
        return None

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _latest_candidate_results_file(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("interpolate_*_candidate_fits.csv"))
    if not candidates:
        raise FileNotFoundError(f"No candidate fits CSV found in {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_results_ages(results_csv: Path) -> dict[str, float]:
    """Parse interpolated ages (Gyr) from candidate results keyed by TIC ID."""
    results_age_by_tic: dict[str, float] = {}

    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = (row.get("tic_id") or row.get("tic_ids") or "").strip()
            age_yr = _extract_first_float(row.get("age_yr"))
            if not tic_id or age_yr is None or not math.isfinite(age_yr):
                continue

            results_age_by_tic[tic_id] = age_yr / 1e9

    return results_age_by_tic


def _parse_kepler_comparison_ages(kepler_catalog_csv: Path = KEPLER_AGES) -> tuple[dict[str, float], int]:
    """Parse Kepler catalog ages (Gyr) keyed by TIC ID without the TIC prefix."""
    tic_to_age_gyr: dict[str, float] = {}
    invalid_or_nan_rows = 0

    with kepler_catalog_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = re.sub(r"^TIC\s*", "", (row.get("tic_ids") or "").strip(), flags=re.IGNORECASE)
            age_gyr = _extract_first_float(row.get("st_age"))
            if not tic_id or age_gyr is None or not math.isfinite(age_gyr):
                invalid_or_nan_rows += 1
                continue

            tic_to_age_gyr[tic_id] = age_gyr

    return tic_to_age_gyr, invalid_or_nan_rows


def _parse_k2_comparison_ages(k2_catalog_csv: Path | None = None) -> tuple[dict[str, float], int]:
    """Parse K2 catalog ages (Gyr) keyed by TIC ID without the TIC prefix.

    Prefers the explicitly provided path. If not provided, uses `k2_star_ages.csv` and
    falls back to `k2_star_ages.cvsv` to support legacy naming.
    """
    if k2_catalog_csv is None:
        default_csv = K2_AGES
        legacy_cvsv = default_csv.with_suffix(".cvsv")
        k2_catalog_csv = default_csv if default_csv.exists() else legacy_cvsv

    if not k2_catalog_csv.exists():
        raise FileNotFoundError(f"K2 catalog file not found: {k2_catalog_csv}")

    tic_to_age_gyr: dict[str, float] = {}
    invalid_or_nan_rows = 0
    with k2_catalog_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = re.sub(r"^TIC\s*", "", (row.get("tic_ids") or "").strip(), flags=re.IGNORECASE)
            age_gyr = _extract_first_float(row.get("st_age"))
            if not tic_id or age_gyr is None or not math.isfinite(age_gyr):
                invalid_or_nan_rows += 1
                continue

            tic_to_age_gyr[tic_id] = age_gyr

    return tic_to_age_gyr, invalid_or_nan_rows


def _parse_external_comparison_ages(
    comparison_csv: Path,
    tic_column: str,
    age_column: str,
    age_unit: str,
) -> tuple[dict[str, float], int]:
    """Parse external comparison ages (Gyr) keyed by TIC ID from user-specified columns."""
    tic_to_age_gyr: dict[str, float] = {}
    invalid_or_nan_rows = 0

    unit_scale = 1e9 if age_unit == "yr" else 1.0

    with comparison_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = re.sub(r"^TIC\s*", "", (row.get(tic_column) or "").strip(), flags=re.IGNORECASE)
            age_value = _extract_first_float(row.get(age_column))
            if not tic_id or age_value is None or not math.isfinite(age_value):
                invalid_or_nan_rows += 1
                continue

            tic_to_age_gyr[tic_id] = age_value / unit_scale

    return tic_to_age_gyr, invalid_or_nan_rows


def _fit_linear_regression(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)

    sxx = sum((xi - mean_x) ** 2 for xi in x_values)
    if sxx == 0:
        raise ValueError("Comparison ages have zero variance; linear regression is undefined.")

    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_values, y_values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    y_hat = [slope * xi + intercept for xi in x_values]
    residuals = [yi - yhi for yi, yhi in zip(y_values, y_hat)]

    ss_res = sum(res * res for res in residuals)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_values)
    r_squared = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    rmse = math.sqrt(ss_res / len(residuals))

    return {
        "n_overlap": float(len(x_values)),
        "slope": float(slope),
        "intercept_gyr": float(intercept),
        "r_squared": float(r_squared),
        "rmse_gyr": rmse,
    }


def _plot_regression(
    x_comparison_age_gyr: list[float],
    y_results_age_gyr: list[float],
    regression: dict[str, float],
    *,
    comparison_label: str,
    output_path: Path,
) -> Path:
    """Plot fit line and extracted-age dots (red=results, blue=comparison)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_min = min(x_comparison_age_gyr)
    x_max = max(x_comparison_age_gyr)
    x_line = [x_min, x_max]
    y_fit_line = [
        regression["slope"] * x_min + regression["intercept_gyr"],
        regression["slope"] * x_max + regression["intercept_gyr"],
    ]

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate the regression plot.") from exc

    fig, ax = plt.subplots(figsize=(8, 6))
    # Comparison ages shown on y=x as blue reference points.
    ax.scatter(x_comparison_age_gyr, x_comparison_age_gyr, c="blue", s=26, alpha=0.7, label=f"{comparison_label} ages")
    # Results ages at y=interpolated_age as red points.
    ax.scatter(x_comparison_age_gyr, y_results_age_gyr, c="red", s=28, alpha=0.5, label="Results ages")
    ax.plot(x_line, y_fit_line, c="black", linewidth=2.0, label="Best fit line")

    ax.set_xlabel(f"{comparison_label} age (Gyr)")
    ax.set_ylabel("Interpolated age (Gyr)")
    ax.set_title(f"Interpolated ages vs {comparison_label} ages")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def regress_interpolated_ages(
    *,
    results_csv: Path | None = None,
    comparison_source: str = "kepler",
    comparison_csv: Path | None = None,
    comparison_tic_column: str = "tic_id",
    comparison_age_column: str = "age_gyr",
    comparison_age_unit: str = "gyr",
    plot_output: Path | None = None,
    create_plot: bool = True,
) -> dict[str, float | str]:
    """Fit y = m*x + b where x=comparison age (Gyr), y=interpolated age (Gyr)."""
    results_path = results_csv or _latest_candidate_results_file(OUTPUT_RESULTS_DIR)
    results_age_by_tic = _parse_results_ages(results_path)

    if comparison_source == "kepler":
        comparison_age_by_tic, comparison_invalid_or_nan_rows = _parse_kepler_comparison_ages()
        comparison_label = "Kepler"
    elif comparison_source == "k2":
        comparison_age_by_tic, comparison_invalid_or_nan_rows = _parse_k2_comparison_ages()
        comparison_label = "K2"
    elif comparison_source == "external":
        if comparison_csv is None:
            raise ValueError("comparison_csv must be provided when comparison_source='external'.")
        comparison_age_by_tic, comparison_invalid_or_nan_rows = _parse_external_comparison_ages(
            comparison_csv=comparison_csv,
            tic_column=comparison_tic_column,
            age_column=comparison_age_column,
            age_unit=comparison_age_unit,
        )
        comparison_label = "External"
    else:
        raise ValueError("comparison_source must be one of: kepler, k2, external")

    x_comparison_age_gyr: list[float] = []
    y_results_age_gyr: list[float] = []
    for tic_id, result_age_gyr in results_age_by_tic.items():
        comparison_age_gyr = comparison_age_by_tic.get(tic_id)
        if comparison_age_gyr is None or not math.isfinite(comparison_age_gyr):
            continue

        x_comparison_age_gyr.append(comparison_age_gyr)
        y_results_age_gyr.append(result_age_gyr)

    if len(x_comparison_age_gyr) < 2:
        raise ValueError("Need at least two overlapping stars between results and selected comparison ages.")

    regression = _fit_linear_regression(x_comparison_age_gyr, y_results_age_gyr)

    plot_path = "not_generated"
    if create_plot:
        if plot_output is None:
            plot_output = LoggingUtils.timestamped_output_path(
                output_dir=OUTPUT_FIGS_DIR,
                suffix=f"ages_regression_{comparison_source}.png",
                prefix="interpolate",
            )
        saved_plot = _plot_regression(
            x_comparison_age_gyr=x_comparison_age_gyr,
            y_results_age_gyr=y_results_age_gyr,
            regression=regression,
            comparison_label=comparison_label,
            output_path=plot_output,
        )
        plot_path = str(saved_plot)

    return {
        **regression,
        "comparison_source": comparison_source,
        "comparison_invalid_or_nan_skipped": float(comparison_invalid_or_nan_rows),
        "results_csv": str(results_path),
        "plot_path": plot_path,
    }


def _format_result(result: dict[str, float | str]) -> str:
    source = str(result["comparison_source"])
    return (
        f"n={int(float(result['n_overlap']))}\n"
        f"comparison invalid/NaN rows skipped={int(float(result['comparison_invalid_or_nan_skipped']))}\n"
        f"fit: interpolated_age_gyr = ({float(result['slope']):.4f}) * {source}_age_gyr + ({float(result['intercept_gyr']):.4f})\n"
        f"R^2={float(result['r_squared']):.4f}, RMSE={float(result['rmse_gyr']):.4f} Gyr\n"
        f"plot={result['plot_path']}"
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regress interpolated ages against Kepler, K2, or external ages.")
    parser.add_argument("--results-csv", type=Path, default=None, help="Candidate fits CSV. Defaults to latest.")
    parser.add_argument(
        "--comparison-source",
        choices=("kepler", "k2", "external"),
        default="external",
        help="Comparison age source.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default="Users/archon",
        help="External comparison CSV path (required for --comparison-source external).",
    )
    parser.add_argument(
        "--comparison-tic-column",
        default="tic_id",
        help="TIC column name in external comparison CSV.",
    )
    parser.add_argument(
        "--comparison-age-column",
        default="age_gyr",
        help="Age column name in external comparison CSV.",
    )
    parser.add_argument(
        "--comparison-age-unit",
        choices=("gyr", "yr"),
        default="gyr",
        help="Unit used by --comparison-age-column in the external CSV.",
    )
    parser.add_argument("--plot-output", type=Path, default=None, help="Optional output path for the regression plot.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip writing the regression plot.")
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    regression = regress_interpolated_ages(
        results_csv=args.results_csv,
        comparison_source=args.comparison_source,
        comparison_csv=args.comparison_csv,
        comparison_tic_column=args.comparison_tic_column,
        comparison_age_column=args.comparison_age_column,
        comparison_age_unit=args.comparison_age_unit,
        plot_output=args.plot_output,
        create_plot=not args.skip_plot,
    )
    print(f"Linear regression between interpolated ages and {args.comparison_source} ages")
    print(_format_result(regression))
