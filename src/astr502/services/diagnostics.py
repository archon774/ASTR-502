from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, median



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _latest_candidate_results_file(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("interpolate_*_candidate_fits.csv"))
    if not candidates:
        raise FileNotFoundError(f"No candidate fits CSV found in {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _hostname_family(hostname: str) -> str:
    """Collapse hostnames into families such as Kepler, TOI, HD, etc."""
    cleaned = re.sub(r"\s+", " ", str(hostname or "").strip())
    if not cleaned:
        return "UNKNOWN"

    match = re.match(r"^([A-Za-z]+)", cleaned)
    if match:
        return match.group(1).upper()

    token = re.split(r"[-_\s]", cleaned, maxsplit=1)[0]
    return token.upper() if token else "UNKNOWN"


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def _cohens_d(group: list[float], remainder: list[float]) -> float | None:
    """Compute Cohen's d effect size for one group vs all others."""
    if len(group) < 2 or len(remainder) < 2:
        return None

    g_var = _sample_variance(group)
    r_var = _sample_variance(remainder)
    if g_var is None or r_var is None:
        return None

    pooled_var = ((len(group) - 1) * g_var + (len(remainder) - 1) * r_var) / (len(group) + len(remainder) - 2)
    if pooled_var <= 0:
        return None

    return (mean(group) - mean(remainder)) / math.sqrt(pooled_var)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty list")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _winsorized(values: list[float], upper_cap: float) -> list[float]:
    return [min(value, upper_cap) for value in values]


def run_hostname_fit_diagnostics(
    *,
    results_csv: Path | None = None,
    min_group_size: int = 5,
    good_fit_threshold: float = 1.5,
) -> list[dict[str, int | float | str | None]]:
    """Check whether fit quality appears to vary by hostname family.

    Lower `chi2_reduced` means a better fit.
    """
    default_results_dir = _repo_root() / "outputs" / "results"
    results_path = results_csv or _latest_candidate_results_file(default_results_dir)

    family_to_chi2: dict[str, list[float]] = {}
    all_chi2: list[float] = []

    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Results file {results_path} has no header row")

        required_cols = {"hostname", "chi2_reduced"}
        missing = required_cols.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Results file {results_path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            hostname = (row.get("hostname") or "").strip()
            chi2 = _to_float(row.get("chi2_reduced"))
            if chi2 is None:
                continue

            family = _hostname_family(hostname)
            family_to_chi2.setdefault(family, []).append(chi2)
            all_chi2.append(chi2)

    if not all_chi2:
        raise ValueError(f"No valid rows with numeric chi2_reduced were found in {results_path}")

    overall_median = median(all_chi2)
    overall_good_rate = sum(value <= good_fit_threshold for value in all_chi2) / len(all_chi2)

    upper_cap = _quantile(all_chi2, 0.99)
    all_chi2_capped = _winsorized(all_chi2, upper_cap)
    overall_capped_mean = mean(all_chi2_capped)

    summary: list[dict[str, int | float | str | None]] = []
    for family, chi2_values in family_to_chi2.items():
        if len(chi2_values) < min_group_size:
            continue

        remainder = [value for other_family, values in family_to_chi2.items() if other_family != family for value in values]
        family_capped_mean = mean(_winsorized(chi2_values, upper_cap))
        family_median = median(chi2_values)
        family_good_rate = sum(value <= good_fit_threshold for value in chi2_values) / len(chi2_values)

        # Cohen's d on log1p-compressed chi2 to avoid one extreme outlier dominating.
        group_log = [math.log10(1.0 + value) for value in chi2_values]
        remainder_log = [math.log10(1.0 + value) for value in remainder]

        summary.append(
            {
                "family": family,
                "n": len(chi2_values),
                "capped_mean_chi2_reduced": family_capped_mean,
                "median_chi2_reduced": family_median,
                "good_fit_rate": family_good_rate,
                "delta_capped_mean_vs_overall": family_capped_mean - overall_capped_mean,
                "delta_median_vs_overall": family_median - overall_median,
                "cohens_d_logchi2_vs_other_families": _cohens_d(group_log, remainder_log),
            }
        )

    if not summary:
        print(f"Results file: {results_path}")
        print(
            f"No hostname family had at least {min_group_size} rows. "
            "Try lowering min_group_size."
        )
        return []

    summary.sort(key=lambda row: float(row["median_chi2_reduced"]))

    print(f"Results file: {results_path}")
    print(f"Rows analyzed: {len(all_chi2)}")
    print(f"Unique hostname families: {len(family_to_chi2)}")
    print(f"Overall median reduced chi2: {overall_median:.3f}")
    print(f"Overall capped-mean reduced chi2 (99th pct cap={upper_cap:.3f}): {overall_capped_mean:.3f}")
    print(
        f"Overall good-fit rate (chi2_reduced <= {good_fit_threshold:.2f}): "
        f"{overall_good_rate:.1%}"
    )

    header = (
        f"{'family':<10} {'n':>5} {'cap_mean':>9} {'median':>8} {'good_rate':>10} "
        f"{'Δcap_mean':>10} {'Δmedian':>9} {'cohens_d':>9}"
    )
    print("\nHostname-family summary (lower chi2_reduced = better):")
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{str(row['family']):<10} {int(row['n']):>5d} "
            f"{float(row['capped_mean_chi2_reduced']):>9.3f} {float(row['median_chi2_reduced']):>8.3f} "
            f"{float(row['good_fit_rate']):>9.1%} {float(row['delta_capped_mean_vs_overall']):>+10.3f} "
            f"{float(row['delta_median_vs_overall']):>+9.3f} {_format_float(row['cohens_d_logchi2_vs_other_families']):>9}"
        )

    best = summary[0]
    worst = summary[-1]
    print(
        "\nQuick readout: "
        f"best median family={best['family']} ({float(best['median_chi2_reduced']):.3f}), "
        f"worst median family={worst['family']} ({float(worst['median_chi2_reduced']):.3f})."
    )

    return summary


if __name__ == "__main__":
    run_hostname_fit_diagnostics()
