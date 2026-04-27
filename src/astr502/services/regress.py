from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from src.astr502.data.paths import K2_AGES, KEPLER_AGES, OUTPUT_RESULTS_DIR


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


def _load_comparison_ages(comparison_catalog_csv: Path) -> dict[str, float]:
    """Load comparison st_age (Gyr) keyed by TIC ID without the TIC prefix."""
    tic_to_age_gyr: dict[str, float] = {}

    with comparison_catalog_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = re.sub(r"^TIC\s*", "", (row.get("tic_ids") or "").strip(), flags=re.IGNORECASE)
            age_gyr = _extract_first_float(row.get("st_age"))
            if not tic_id or age_gyr is None or not math.isfinite(age_gyr):
                continue

            tic_to_age_gyr[tic_id] = age_gyr

    return tic_to_age_gyr


def regress_interpolated_vs_catalog_ages(
    results_csv: Path | None = None,
    comparison_csv: Path = KEPLER_AGES,
    comparison_label: str = "Kepler",
) -> dict[str, float]:
    """Fit y = m*x + b where x=catalog st_age (Gyr), y=interpolated age (Gyr)."""
    results_path = results_csv or _latest_candidate_results_file(OUTPUT_RESULTS_DIR)
    comparison_age_by_tic = _load_comparison_ages(comparison_csv)

    x_catalog_age_gyr: list[float] = []
    y_interp_age_gyr: list[float] = []

    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = (row.get("tic_id") or row.get("tic_ids") or "").strip()
            age_yr = _extract_first_float(row.get("age_yr"))
            if not tic_id or age_yr is None or not math.isfinite(age_yr):
                continue

            catalog_age_gyr = comparison_age_by_tic.get(tic_id)
            if catalog_age_gyr is None or not math.isfinite(catalog_age_gyr):
                continue

            interp_age_gyr = age_yr / 1e9
            x_catalog_age_gyr.append(catalog_age_gyr)
            y_interp_age_gyr.append(interp_age_gyr)

    if len(x_catalog_age_gyr) < 2:
        raise ValueError(
            f"Need at least two overlapping stars between latest interpolation results and {comparison_csv.name}."
        )

    x = x_catalog_age_gyr
    y = y_interp_age_gyr

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError(f"{comparison_label} ages have zero variance; linear regression is undefined.")

    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    y_hat = [slope * xi + intercept for xi in x]
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]

    ss_res = sum(res * res for res in residuals)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    rmse = math.sqrt(ss_res / len(residuals))

    return {
        "n_overlap": float(len(x)),
        "slope": float(slope),
        "intercept_gyr": float(intercept),
        "r_squared": float(r_squared),
        "rmse_gyr": rmse,
        "comparison_csv": str(comparison_csv),
        "comparison_label": comparison_label,
    }


def regress_interpolated_vs_kepler_ages(results_csv: Path | None = None) -> dict[str, float]:
    """Backward-compatible wrapper using kepler_star_ages.csv for comparison."""
    return regress_interpolated_vs_catalog_ages(
        results_csv=results_csv,
        comparison_csv=KEPLER_AGES,
        comparison_label="Kepler",
    )


def _format_result(result: dict[str, float]) -> str:
    comparison_label = str(result.get("comparison_label", "Catalog"))
    return (
        f"n={int(result['n_overlap'])}\n"
        f"fit: interpolated_age_gyr = ({result['slope']:.4f}) * {comparison_label.lower()}_st_age_gyr + ({result['intercept_gyr']:.4f})\n"
        f"R^2={result['r_squared']:.4f}, RMSE={result['rmse_gyr']:.4f} Gyr"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Linear regression between interpolated ages and a comparison age catalog "
            "(Kepler, K2, or user-provided CSV)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional path to interpolate_*_candidate_fits.csv. Defaults to latest in outputs/results.",
    )
    parser.add_argument(
        "--comparison-source",
        choices=["kepler", "k2", "external"],
        default="kepler",
        help="Which comparison catalog to use for st_age values.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="Path to external comparison CSV when --comparison-source=external.",
    )
    return parser


def _resolve_comparison_csv(comparison_source: str, external_csv: Path | None) -> tuple[Path, str]:
    if comparison_source == "kepler":
        return KEPLER_AGES, "Kepler"
    if comparison_source == "k2":
        return K2_AGES, "K2"
    if external_csv is None:
        raise ValueError("When --comparison-source=external, you must supply --comparison-csv.")
    return external_csv, "External"


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    comparison_csv, comparison_label = _resolve_comparison_csv(args.comparison_source, args.comparison_csv)
    regression = regress_interpolated_vs_catalog_ages(
        results_csv=args.results_csv,
        comparison_csv=comparison_csv,
        comparison_label=comparison_label,
    )
    print(f"Linear regression between latest interpolated ages and {comparison_csv.name} ({comparison_label})")
    print(_format_result(regression))
