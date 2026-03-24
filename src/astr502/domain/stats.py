from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ChiSquareSummary:
    """Container for photometric and prior chi-square terms."""

    chi2_phot: float
    chi2_prior: float

    @property
    def chi2_total(self) -> float:
        return float(self.chi2_phot + self.chi2_prior)


def chi2_photometric(
    model_mags: Mapping[str, float],
    observed_abs_mags: Mapping[str, float],
    sigma_phot: float,
) -> float:
    """Compute photometric chi-square using all overlapping finite bands."""
    chi2 = 0.0
    n_used = 0

    for band, observed in observed_abs_mags.items():
        predicted = model_mags.get(band, np.nan)
        if not np.isfinite(predicted):
            continue
        chi2 += ((observed - predicted) / sigma_phot) ** 2
        n_used += 1

    if n_used == 0:
        return 1e30
    return float(chi2)


def chi2_prior(
    mass: float,
    log10_age: float,
    feh: float,
    prior: Mapping[str, float],
) -> float:
    """Gaussian priors for mass/feh and asymmetric age prior (in Gyr)."""
    chi2 = 0.0

    if np.isfinite(prior["m0"]):
        chi2 += ((mass - prior["m0"]) / prior["sig_m"]) ** 2

    if np.isfinite(prior["feh0"]):
        chi2 += ((feh - prior["feh0"]) / prior["sig_feh"]) ** 2

    if np.isfinite(prior["a0_gyr"]):
        age_gyr = (10.0 ** log10_age) / 1e9
        age_sigma = prior["sig_age_hi"] if age_gyr >= prior["a0_gyr"] else prior["sig_age_lo"]
        chi2 += ((age_gyr - prior["a0_gyr"]) / age_sigma) ** 2

    return float(chi2)


def summarize_chi_square(
    model_mags: Mapping[str, float],
    observed_abs_mags: Mapping[str, float],
    sigma_phot: float,
    mass: float,
    log10_age: float,
    feh: float,
    prior: Mapping[str, float],
) -> ChiSquareSummary:
    """Return split and total chi-square terms for a single model evaluation."""
    chi2_data = chi2_photometric(
        model_mags=model_mags,
        observed_abs_mags=observed_abs_mags,
        sigma_phot=sigma_phot,
    )
    chi2_reg = chi2_prior(mass=mass, log10_age=log10_age, feh=feh, prior=prior)
    return ChiSquareSummary(chi2_phot=chi2_data, chi2_prior=chi2_reg)


def reduced_chi2_from_csv(csv_path: str | Path, n_fit_params: int = 4) -> dict[str, float]:
    """Compute reduced chi-square values for each star in a fit-results CSV.

    The CSV is expected to include:
      - ``hostname``
      - either ``chi2`` or ``chi2_total``
      - one or more ``model_*`` columns for the fitted photometric bands

    The per-row degrees of freedom are computed as::

        dof = N_model_bands_with_finite_values - n_fit_params

    where ``n_fit_params`` defaults to 4 (mass, age, [Fe/H], Av).
    """
    path = Path(csv_path)
    reduced: dict[str, float] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return reduced

        fieldnames = set(reader.fieldnames)
        chi2_col = "chi2" if "chi2" in fieldnames else "chi2_total"
        if chi2_col not in fieldnames:
            raise ValueError("CSV must include either 'chi2' or 'chi2_total'.")

        model_cols = [name for name in reader.fieldnames if name.startswith("model_")]
        if not model_cols:
            raise ValueError("CSV must include at least one 'model_*' column.")

        for row in reader:
            hostname = row.get("hostname", "")
            if not hostname:
                continue

            chi2_val = float(row[chi2_col])
            n_bands = 0
            for col in model_cols:
                value = row.get(col, "")
                try:
                    if np.isfinite(float(value)):
                        n_bands += 1
                except (TypeError, ValueError):
                    continue

            dof = n_bands - n_fit_params
            reduced[hostname] = float(chi2_val / dof) if dof > 0 else np.nan

    return reduced
