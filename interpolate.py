import pandas as pd
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from find_mag import PhotometryMerger


def _available_bands(obs: Dict[str, float],
                     model_df: pd.DataFrame,
                     mag_cols: Optional[Iterable[str]] = None):
    obs_keys = set(obs.keys())
    if mag_cols is None:
        cand = set(mag_cols)
    else:
        cand = set(model_df.columns)
    bands = list(obs_keys.intersection(cand))
    return bands

def _row_log_likelihood(row: pd.Series,
                        obs: Dict[str, float],
                        errs: Dict[str, float],
                        bands: Iterable[str]) -> float:
    ll = 0.0
    for b in bands:
        m_obs = obs.get(b)
        s = errs.get(b)
        m_mod = row.get(b)
        if m_obs is None or s is None or m_mod is None:
            continue
        if pd.isna(m_obs) or pd.isna(s) or pd.isna(m_mod) or s <=0:
            continue
        resid = m_obs - m_mod
        ll += -0.5 * (resid * resid) / (s * s) + np.log(2.0 * np.pi * s * s)
    return ll

def brute_force(
        observed_mags: Dict[str, float],
        observed_errs: Dict[str, float],
        model_df: pd.DataFrame,
        mag_cols: Optional[Iterable[str]] = None,
        mass_col: str = "mass",
        age_col: str = "age",
        feh_col: str = "mh"
) -> Tuple[pd.Series, pd.DataFrame]:
    bands = _available_bands(observed_mags, model_df, mag_cols)
    if len(bands) == 0:
        raise ValueError("No bands available")

    loglikes = model_df.apply(lambda r: _row_log_likelihood(r, observed_mags, observed_errs,bands), axis=1).to_numpy()
    max_ll = np.nanmax(loglikes)

    if not np.isfinite(max_ll):
        raise RuntimeError("All likelihoods are non-finite")

    shift = loglikes - max_ll
    probs = np.exp(shift)

    probs[~np.isfinite(shift)] = 0.0
    s = probs.sum()
    if s <=0:
        probs[:] = 0.0
    else:
        probs /= s

    results = model_df.copy()
    results['loglike'] = loglikes
    results['prob'] = probs

    best_idx = int(np.nanargmax(loglikes))
    best_row = results.iloc[best_idx]
    return best_row, results

observed = {
    'G_abs':0,
    'BP_RP_abs':0,
}

errors = {
    'G_abs': 0,
    'BP_RP_abs': 0,
}
phot_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv'
dist_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv'
model_df = PhotometryMerger.join_photometry_and_distances(phot_csv, dist_csv)

