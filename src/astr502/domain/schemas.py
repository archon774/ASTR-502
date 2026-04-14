from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class FitResultSchema:
    """Serializable schema for one star's best-fit interpolation result."""

    hostname: str
    mass: float
    age_yr: float
    feh: float
    av: float
    chi2_phot: float
    chi2_prior: float
    chi2_total: float
    chi2_reduced: float
    n_obs_bands: int
    distance_pc: float
    model_magnitudes: Mapping[str, float] = field(default_factory=dict)
    mcmc_summary: Mapping[str, Mapping[str, float]] | None = None

    def to_record(self) -> dict[str, float | str]:
        record: dict[str, float | str] = {
            "hostname": self.hostname,
            "mass": self.mass,
            "age_yr": self.age_yr,
            "feh": self.feh,
            "av": self.av,
            "chi2_phot": self.chi2_phot,
            "chi2_prior": self.chi2_prior,
            "chi2_total": self.chi2_total,
            "chi2_reduced": self.chi2_reduced,
            "n_obs_bands": self.n_obs_bands,
            "distance_pc": self.distance_pc,
        }
        for band, mag in self.model_magnitudes.items():
            record[f"model_{band}"] = mag

        if self.mcmc_summary is not None:
            for param_name, stats in self.mcmc_summary.items():
                for stat_name, value in stats.items():
                    record[f"mcmc_{param_name}_{stat_name}"] = float(value)
        return record
