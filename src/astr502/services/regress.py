from __future__ import annotations

import csv
import math
import re
from pathlib import Path




REPO_ROOT = Path(__file__).resolve().parents[3]
KEPLER_AGES = REPO_ROOT / "data" / "raw" / "catalogs" / "kepler_star_ages.csv"
OUTPUT_RESULTS_DIR = REPO_ROOT / "outputs" / "results"


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


def _load_kepler_ages(kepler_catalog_csv: Path = KEPLER_AGES) -> dict[str, float]:
    """Load Kepler catalog st_age (Gyr) keyed by TIC ID without the TIC prefix."""
    tic_to_age_gyr: dict[str, float] = {}

    with kepler_catalog_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = re.sub(r"^TIC\s*", "", (row.get("tic_ids") or "").strip(), flags=re.IGNORECASE)
            age_gyr = _extract_first_float(row.get("st_age"))
            if not tic_id or age_gyr is None or not math.isfinite(age_gyr):
                continue

            tic_to_age_gyr[tic_id] = age_gyr

    return tic_to_age_gyr


def regress_interpolated_vs_kepler_ages(results_csv: Path | None = None) -> dict[str, float]:
    """Fit y = m*x + b where x=Kepler st_age (Gyr), y=interpolated age (Gyr)."""
    results_path = results_csv or _latest_candidate_results_file(OUTPUT_RESULTS_DIR)
    kepler_age_by_tic = _load_kepler_ages()

    x_kepler_age_gyr: list[float] = []
    y_interp_age_gyr: list[float] = []

    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = (row.get("tic_id") or row.get("tic_ids") or "").strip()
            age_yr = _extract_first_float(row.get("age_yr"))
            if not tic_id or age_yr is None or not math.isfinite(age_yr):
                continue

            kepler_age_gyr = kepler_age_by_tic.get(tic_id)
            if kepler_age_gyr is None or not math.isfinite(kepler_age_gyr):
                continue

            interp_age_gyr = age_yr / 1e9
            x_kepler_age_gyr.append(kepler_age_gyr)
            y_interp_age_gyr.append(interp_age_gyr)

    if len(x_kepler_age_gyr) < 2:
        raise ValueError(
            "Need at least two overlapping stars between latest interpolation results and kepler_star_ages.csv."
        )

    x = x_kepler_age_gyr
    y = y_interp_age_gyr

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("Kepler ages have zero variance; linear regression is undefined.")

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
    }


def _format_result(result: dict[str, float]) -> str:
    return (
        f"n={int(result['n_overlap'])}\n"
        f"fit: interpolated_age_gyr = ({result['slope']:.4f}) * kepler_st_age_gyr + ({result['intercept_gyr']:.4f})\n"
        f"R^2={result['r_squared']:.4f}, RMSE={result['rmse_gyr']:.4f} Gyr"
    )


if __name__ == "__main__":
    regression = regress_interpolated_vs_kepler_ages()
    print("Linear regression between latest interpolated ages and kepler_star_ages.csv")
    print(_format_result(regression))
