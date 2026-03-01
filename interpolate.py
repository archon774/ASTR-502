from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from find_mag import PhotometryMerger
from read_spot_models import SPOT
from stats import LikelihoodSummary, dataframe_log_likelihood


@dataclass(frozen=True)
class IsochroneFitResult:
    """Best-fit statistics for one (age, metallicity) isochrone section."""

    age_log10_yr: float
    metallicity_dex: float
    log_likelihood: float
    n_used: int
    predicted_mass_mean: float
    predicted_mass_median: float


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in str(col).lower():
                return str(col)
    return None


def _as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _extract_metallicity_from_path(file_path: str | Path) -> float:
    """Infer metallicity from SPOT filename style like f000.isoc or fm05.isoc."""
    stem = Path(file_path).stem.lower()

    # f000 -> 0.00, fp05 -> +0.5, fm05 -> -0.5
    m = re.search(r"f([pm]?)(\d+)", stem)
    if not m:
        return float("nan")

    sign = m.group(1)
    digits = m.group(2)
    value = float(digits) / 100.0
    if sign == "m":
        value *= -1.0
    return value


def _build_color_mag_interpolator(
    isochrone_df: pd.DataFrame,
    color_col: str,
    mag_col: str,
):
    """Build scipy.interpolate interp1d(color -> magnitude) for one isochrone."""
    track = isochrone_df[[color_col, mag_col]].dropna().sort_values(color_col)
    if len(track) < 2:
        return None

    # Remove repeated x values to keep interp1d stable.
    track = track.loc[~track[color_col].duplicated(keep="first")]
    if len(track) < 2:
        return None

    return interp1d(
        track[color_col].to_numpy(),
        track[mag_col].to_numpy(),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )


def fit_isochrone_section_to_targets(
    isochrone_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    age_log10_yr: float,
    metallicity_dex: float,
    sigma_mag: float = 0.05,
) -> tuple[IsochroneFitResult, LikelihoodSummary, pd.DataFrame]:
    """Fit one isochrone section to the target CMD and score with log-likelihood."""
    iso = _as_numeric(isochrone_df)
    targets = _as_numeric(targets_df)

    # Target photometric columns created by PhotometryMerger.join_photometry_and_distances.
    target_color_col = "BP_RP_abs"
    target_mag_col = "G_abs"

    iso_bp_col = _find_col(iso, ["g_bp", "bp"])
    iso_rp_col = _find_col(iso, ["g_rp", "rp"])
    iso_g_col = _find_col(iso, ["g", "gmag", "gaia_g"])
    mass_col = _find_col(iso, ["m/m", "mass", "mini", "m_ini", "mact"])

    if not (iso_bp_col and iso_rp_col and iso_g_col):
        raise ValueError(
            "Could not identify BP/RP/G columns in SPOT isochrone section; "
            f"columns were: {list(iso.columns)}"
        )

    work = iso[[iso_bp_col, iso_rp_col, iso_g_col] + ([mass_col] if mass_col else [])].copy()
    work["iso_color"] = work[iso_bp_col] - work[iso_rp_col]
    work["iso_mag"] = work[iso_g_col]

    interpolator = _build_color_mag_interpolator(work, "iso_color", "iso_mag")
    if interpolator is None:
        result = IsochroneFitResult(
            age_log10_yr=float(age_log10_yr),
            metallicity_dex=float(metallicity_dex),
            log_likelihood=float("-inf"),
            n_used=0,
            predicted_mass_mean=float("nan"),
            predicted_mass_median=float("nan"),
        )
        empty = pd.DataFrame(columns=[target_color_col, target_mag_col, "iso_mag_pred", "mass_pred"])
        summary = LikelihoodSummary(0, float("-inf"), pd.Series(dtype=float))
        return result, summary, empty

    mask = targets[target_color_col].notna() & targets[target_mag_col].notna()
    eval_df = targets.loc[mask, [target_color_col, target_mag_col]].copy()
    eval_df["iso_mag_pred"] = interpolator(eval_df[target_color_col].to_numpy())

    ll_summary = dataframe_log_likelihood(
        observed=eval_df[target_mag_col],
        predicted=eval_df["iso_mag_pred"],
        sigma=sigma_mag,
    )

    # Estimate masses by nearest color location on the isochrone if mass exists.
    if mass_col:
        ref = work[["iso_color", mass_col]].dropna().sort_values("iso_color")
        if len(ref) >= 2 and ref["iso_color"].nunique() >= 2:
            mass_interp = interp1d(
                ref["iso_color"].to_numpy(),
                ref[mass_col].to_numpy(),
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            eval_df["mass_pred"] = mass_interp(eval_df[target_color_col].to_numpy())
        else:
            eval_df["mass_pred"] = np.nan
    else:
        eval_df["mass_pred"] = np.nan

    result = IsochroneFitResult(
        age_log10_yr=float(age_log10_yr),
        metallicity_dex=float(metallicity_dex),
        log_likelihood=ll_summary.log_likelihood,
        n_used=ll_summary.n_used,
        predicted_mass_mean=float(pd.to_numeric(eval_df["mass_pred"], errors="coerce").mean()),
        predicted_mass_median=float(pd.to_numeric(eval_df["mass_pred"], errors="coerce").median()),
    )

    return result, ll_summary, eval_df


def fit_spot_grid_to_targets(
    targets_df: pd.DataFrame,
    spot_iso_files: Iterable[str | Path],
    sigma_mag: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every SPOT age section from every metallicity file to target data."""
    results: list[IsochroneFitResult] = []
    best_eval: pd.DataFrame | None = None
    best_ll = float("-inf")

    for iso_file in spot_iso_files:
        metallicity = _extract_metallicity_from_path(iso_file)
        sections = SPOT(str(iso_file), verbose=False).read_iso_file()

        for age, section_df in sections.items():
            try:
                fit, _, eval_df = fit_isochrone_section_to_targets(
                    isochrone_df=section_df,
                    targets_df=targets_df,
                    age_log10_yr=float(age),
                    metallicity_dex=metallicity,
                    sigma_mag=sigma_mag,
                )
            except ValueError:
                continue

            results.append(fit)
            if fit.log_likelihood > best_ll:
                best_ll = fit.log_likelihood
                best_eval = eval_df

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        "log_likelihood", ascending=False
    )
    return results_df.reset_index(drop=True), (best_eval if best_eval is not None else pd.DataFrame())


def load_targets(phot_csv: str | Path, dist_csv: str | Path) -> pd.DataFrame:
    """Load merged target list from photometry + distance catalogues."""
    merger = PhotometryMerger()
    return merger.join_photometry_and_distances(phot_csv=phot_csv, dist_csv=dist_csv)
