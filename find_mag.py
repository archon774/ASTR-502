from fetch_iso import IsochroneFetcher, IsochronePlotter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Iterable


GAIA_G_CANDS = ['sy_gmag', 'sy_g', 'G', 'g', 'gmag', 'phot_g_mean_mag', 'g_band']
GAIA_BP_CANDS = ['sy_bpmag', 'sy_bp', 'BP', 'bp', 'bpmag', 'phot_bp_mean_mag']
GAIA_RP_CANDS = ['sy_rpmag', 'sy_rp', 'RP', 'rp', 'rpmag', 'phot_rp_mean_mag']
PARALLAX_CANDS = ['parallax', 'plx', 'parallax_mas']
DIST_CANDS = ['distance_pc', 'distance', 'st_dist', 'dist', 'bj_dist_pc']


def _find_col(df: pd.DataFrame, candidates):
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def _join_key(df1: pd.DataFrame,
              df2: pd.DataFrame,
              extra_candidates=None) -> Optional[str]:
    common = set(df1.columns).intersection(df2.columns)
    if common:
        for pref in ('source_id', 'id', 'star_id', 'obj_id', 'object_id'):
            if pref in common:
                return pref
        return sorted(common)[0]
    candidates = list(extra_candidates or []) + ['source_id','id','star_id','name','objid','gaia_source_id','sourceid']
    for cand in candidates:
        if cand in df1.columns and cand in df2.columns:
            return cand
    return None


def _abs_mag_series(apparent: pd.Series, distance_pc: pd.Series) -> pd.Series:
    # apparent may be non-numeric -> coerce; distance must be positive
    a = pd.to_numeric(apparent, errors='coerce')
    d = pd.to_numeric(distance_pc, errors='coerce')
    valid = d > 0
    abs_mag = pd.Series(np.nan, index=a.index, dtype=float)
    abs_mag[valid] = a[valid] - 5.0 * np.log10(d[valid]) + 5.0
    return abs_mag


def join_photometry_and_distances(phot_csv: Path | str,
                                  dist_csv: Path | str,
                                  on: Optional[str] = None,
                                  how: str = 'inner') -> pd.DataFrame:
    phot_df = pd.read_csv(phot_csv)
    dist_df = pd.read_csv(dist_csv)

    if on is None:
        on = _join_key(phot_df, dist_df)
        if on is None:
            raise ValueError('Could not find a join key. Provide a func')

    joined = phot_df.merge(dist_df, on=on, how=how, suffixes=('_phot', '_dist'))
        # compute distance_pc (add/overwrite column named distance_pc)

        # locate Gaia columns in the merged frame (photometry file likely contains these)
    g_col = _find_col(joined, GAIA_G_CANDS)
    bp_col = _find_col(joined, GAIA_BP_CANDS)
    rp_col = _find_col(joined, GAIA_RP_CANDS)


        # compute absolute mags where possible
    if g_col is not None:
            joined['G_abs'] = _abs_mag_series(joined[g_col], joined['distance_pc'])
    else:
        joined['G_abs'] = np.nan

    if bp_col is not None:
        joined['BP_abs'] = _abs_mag_series(joined[bp_col], joined['distance_pc'])
    else:
        joined['BP_abs'] = np.nan

    if rp_col is not None:
        joined['RP_abs'] = _abs_mag_series(joined[rp_col], joined['distance_pc'])
    else:
        joined['RP_abs'] = np.nan

        # convenience color column
    if 'BP_abs' in joined.columns and 'RP_abs' in joined.columns:
        joined['BP_RP_abs'] = joined['BP_abs'] - joined['RP_abs']
    return joined

phot_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv'
dist_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv'
df = join_photometry_and_distances(phot_csv, dist_csv)

