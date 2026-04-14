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


def reduced_chi2(chi2_total: float, n_obs_bands: int, n_fit_params: int = 4) -> float:
    """Compute reduced chi-square for a single fit result.

    Uses ``dof = n_obs_bands - n_fit_params`` and returns ``np.nan`` when
    there are no positive degrees of freedom.
    """
    dof = int(n_obs_bands) - int(n_fit_params)
    return float(chi2_total / dof) if dof > 0 else float(np.nan)


def chi2_photometric(
    model_mags: Mapping[str, float],
    observed_abs_mags: Mapping[str, float],
    observed_abs_mag_errors: Mapping[str, float],
) -> float:
    """Compute photometric chi-square using all overlapping finite bands."""
    chi2 = 0.0
    n_used = 0

    for band, observed in observed_abs_mags.items():
        predicted = model_mags.get(band, np.nan)
        sigma_band = observed_abs_mag_errors.get(band, np.nan)
        if not np.isfinite(predicted) or not np.isfinite(sigma_band) or sigma_band <= 0:
            continue
        chi2 += ((observed - predicted) / sigma_band) ** 2
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
        chi2 += (mass - prior["m0"]) ** 2

    if np.isfinite(prior["feh0"]):
        chi2 += (feh - prior["feh0"]) ** 2

    if np.isfinite(prior["a0_gyr"]):
        age_gyr = (10.0 ** log10_age) / 1e9
        age_sigma = prior["sig_age_hi"] if age_gyr >= prior["a0_gyr"] else prior["sig_age_lo"]
        chi2 += ((age_gyr - prior["a0_gyr"]) / age_sigma) ** 2

    return float(chi2)


def summarize_chi_square(
    model_mags: Mapping[str, float],
    observed_abs_mags: Mapping[str, float],
    observed_abs_mag_errors: Mapping[str, float],
    mass: float,
    log10_age: float,
    feh: float,
    prior: Mapping[str, float],
) -> ChiSquareSummary:
    """Return split and total chi-square terms for a single model evaluation."""
    chi2_data = chi2_photometric(
        model_mags=model_mags,
        observed_abs_mags=observed_abs_mags,
        observed_abs_mag_errors=observed_abs_mag_errors,
    )
    chi2_reg = chi2_prior(mass=mass, log10_age=log10_age, feh=feh, prior=prior)
    return ChiSquareSummary(chi2_phot=chi2_data, chi2_prior=chi2_reg)

