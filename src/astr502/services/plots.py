from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.astr502.domain.stats import reduced_chi2_from_csv


def plot_observed_vs_table_age_scatter(
    catalog_csv: str | Path = "../../../data/raw/catalogs/ASTR502_Mega_Target_List.csv",
    observed_csv: str | Path = "../../../outputs/results/interpolate_best_fit_results.csv",
    output_path: str | Path = "../../../outputs/figs/age_obs_vs_table_scatter.png",
) -> Path:
    """Plot fractional age residuals vs. table age for targets with both age values.

    The x-axis is (Age_obs - Age_table) / Age_table, where:
      - Age_obs is read from `age_yr` in the observed results file and converted to Gyr.
      - Age_table is read from `st_age` in the mega target list (already in Gyr).

    The y-axis is Age_table.
    """

    catalog_csv = Path(catalog_csv)
    observed_csv = Path(observed_csv)
    output_path = Path(output_path)

    table_age_by_host: dict[str, float] = {}
    with catalog_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hostname = (row.get("hostname") or "").strip()
            st_age_raw = (row.get("st_age") or "").strip()
            if not hostname or not st_age_raw:
                continue
            try:
                st_age = float(st_age_raw)
            except ValueError:
                continue
            if st_age <= 0:
                continue
            table_age_by_host[hostname] = st_age

    x_values: list[float] = []
    y_values: list[float] = []
    reduced_chi2_by_host = reduced_chi2_from_csv(observed_csv)
    reduced_chi2_values: list[float] = []

    with observed_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hostname = (row.get("hostname") or "").strip()
            age_obs_raw = (row.get("age_yr") or "").strip()

            if not hostname or not age_obs_raw or hostname not in table_age_by_host:
                continue

            try:
                age_obs_yr = float(age_obs_raw)
            except ValueError:
                continue
            if age_obs_yr <= 0:
                continue

            age_table_gyr = table_age_by_host[hostname]
            age_obs_gyr = age_obs_yr / 1e9
            fractional_residual = (age_obs_gyr - age_table_gyr) / age_table_gyr

            x_values.append(fractional_residual)
            y_values.append(age_table_gyr)
            reduced_chi2_values.append(reduced_chi2_by_host.get(hostname, np.nan))

    if not x_values:
        raise ValueError(
            "No overlapping valid targets found between catalog 'st_age' and results 'age_yr' columns."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    reduced = np.asarray(reduced_chi2_values, dtype=float)
    finite = reduced[np.isfinite(reduced) & (reduced > 0)]
    if finite.size:
        # Visual scaling around reduced-chi2=1.
        vmin = min(float(np.nanmin(finite)), 1.0)
        vmax = max(float(np.nanmax(finite)), 1.0)
        if vmin == vmax:
            vmin, vmax = 0.5, 2.0
    else:
        vmin, vmax = 0.5, 2.0

    chi2_cmap = LinearSegmentedColormap.from_list(
        "chi2_blue_green_red",
        [(0.0, "#1f77b4"), (0.5, "#2ca02c"), (1.0, "#d62728")],
    )
    chi2_cmap.set_bad(color="0.65")

    scatter = ax.scatter(
        x_values,
        y_values,
        c=reduced,
        cmap=chi2_cmap,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax),
        s=24,
        alpha=0.85,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Reduced $\chi^2$")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-1,1)
    ax.set_xlabel(r"$(Age_{obs} - Age_{table}) / Age_{table}$")
    ax.set_ylabel("Age_table (Gyr)")
    ax.set_title("Observed vs Table Age (point color = reduced $\\chi^2$)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    saved_path = plot_observed_vs_table_age_scatter()
    print(f"Saved plot to: {saved_path}")
