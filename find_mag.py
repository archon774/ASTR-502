from fetch_iso import IsochroneFetcher, IsochronePlotter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def _get_distance_pc(star, distance_pc=None):
    if distance_pc is not None:
        return float(distance_pc)

    for k in ('distance_pc', 'st_dist'):
        if star.get(k) is not None:
            return float(star[k])

    #parallax in mas to distance in pc

    for k in ('parallax', 'st_parallax'):
        if star.get(k) is not None:
            plx = float(star[k])
            if plx > 0:
                return 1000.0 / plx

    raise ValueError('Could not find distance_pc')

def _abs_mag(apparent_mag, distance_pc):
    return float(apparent_mag) - 5.0 * np.log10(distance_pc) + 5.0

def _find_col(df: pd.DataFrame, candidates):
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def _find_best_model(iso_df: pd.DataFrame, star: dict):

    teff_cands = ['teff', 'teff_k', 'teff_eff']
    logg_cands = ['logg', 'log_g']
    mh_cands = ['mh', 'met', 'feh', '[m/h]', '[fe/h]']

    teff_col = _find_col(iso_df, teff_cands)
    logg_col = _find_col(iso_df, logg_cands)
    mh_col = _find_col(iso_df, mh_cands)

    obs = {}
    if star.get('st_teff') is not None and teff_col is not None:
        obs['teff'] = (float(star['st_teff']), teff_col)
    if star.get('st_logg') is not None and logg_col is not None:
        obs['logg'] = (float(star['st_logg']), logg_col)
    for k in ('st_met', 'st_mh', 'st_feh'):
        if star.get(k) is not None and mh_col is not None:
            obs['mh'] = (float(star[k]), mh_col)
            break

    if not obs:
        return None

    # normalization scales so differences are comparable
    scales = {'teff': 500.0, 'logg': 1.0, 'mh': 0.5}

    best_idx = None
    best_dist = np.inf
    n = len(iso_df)
    for i in range(n):
        ssd = 0.0
        valid = True
        for key, (obs_val, col) in obs.items():
            try:
                model_val = iso_df.iloc[i][col]
                if pd.isna(model_val):
                    valid = False
                    break
                diff = float(model_val) - float(obs_val)
                ssd += (diff / scales.get(key, 1.0)) ** 2
            except Exception:
                valid = False
                break
        if valid and ssd < best_dist:
            best_dist = ssd
            best_idx = i

    if best_idx is None:
        return None
    return iso_df.iloc[best_idx]

def plot_star(csv_path, fetcher: IsochroneFetcher, plotter: IsochronePlotter, age=None, MH=None, label='Star'):

    csv_path = Path(csv_path)
    star = parse_star_csv(csv_path)

    if age is None:
        if star.get('st_age') is not None:
            age = float(star['st_age'])
        else:
            raise ValueError('No age provided and star age not found in CSV')
        if MH is None and star.get('st_met') is not None:
            MH = float(star['st_met'])
        if MH is None:
            MH = 0.0
            print("No MH provided, defaulting to 0.0")

    iso_df = fetcher.fetch(age, MH)

    #plot the isochrone
    fig = plotter.plot(age, MH, labels='Isochrone')

    #try to find a model that matches the star
    model = _find_best_model(iso_df, star)
    if model is None:
        raise ValueError('No model provided for star comparison')



def parse_star_csv(csv_path: Path):
    #build this to extract magnitudes and stellar parameters from a CSV file (waiting for other team)
    return None