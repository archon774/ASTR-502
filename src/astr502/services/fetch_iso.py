from __future__ import annotations

from typing import Optional, Tuple

import ezpadova
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class IsochroneFetcher:
    def __init__(
        self,
        photsys: str = "gaiaEDR3",
        step_age: float = 0.01,
        step_mh: float = 0.01,
    ) -> None:
        self.photsys = photsys
        self.step_age = float(step_age)
        self.step_mh = float(step_mh)
        self._cache: dict[tuple, pd.DataFrame] = {}

        self.bp_candidates = ["bp", "g_bp", "gbp", "phot_bp_mean_mag"]
        self.rp_candidates = ["rp", "g_rp", "grp", "phot_rp_mean_mag"]
        self.g_candidates = ["g ", " g", "gmag", "phot_g_mean_mag", "g_band", "g"]

    def _ensure_df(self, obj: pd.DataFrame | list[pd.DataFrame]) -> pd.DataFrame:
        if isinstance(obj, list):
            return pd.concat(obj, ignore_index=True)
        return obj

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        for cand in candidates:
            for col in df.columns:
                if cand.lower() in col.lower():
                    return col
        return None

    def _norm_triplet(self, val: Optional[float], step: float) -> Optional[Tuple[float, float, float]]:
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return tuple(val)
        return float(val), float(val), float(step)

    def _cache_key(self, logage_triplet, mh_triplet) -> tuple:
        return (
            tuple(logage_triplet) if logage_triplet is not None else None,
            tuple(mh_triplet) if mh_triplet is not None else None,
            self.photsys,
        )

    def fetch(self, logage: float, mh: float) -> pd.DataFrame:
        logage_t = self._norm_triplet(logage, self.step_age)
        mh_t = self._norm_triplet(mh, self.step_mh)
        key = self._cache_key(logage_t, mh_t)
        if key in self._cache:
            return self._cache[key]

        raw = ezpadova.get_isochrones(logage=logage_t, MH=mh_t, photsys_file=self.photsys)
        df = self._ensure_df(raw)
        self._cache[key] = df
        return df

    def fetch_grid(self, logages: list[float], mhs: list[float]) -> list[pd.DataFrame]:
        return [self.fetch(logage, mh) for logage in logages for mh in mhs]

    def photometry(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, tuple[str, str, str]]:
        bp_col = self._find_col(df, self.bp_candidates)
        rp_col = self._find_col(df, self.rp_candidates)
        g_col = self._find_col(df, self.g_candidates)

        if not (bp_col and rp_col and g_col):
            raise RuntimeError("Could not locate Gaia BP/RP/G columns in isochrone data")

        color = df[bp_col] - df[rp_col]
        mag = df[g_col]
        return color, mag, (bp_col, rp_col, g_col)


class IsochronePlotter:
    def __init__(self, fetcher: IsochroneFetcher):
        self.fetcher = fetcher

    def plot(self, logage: float, mh: float, label: str = "A"):
        df = self.fetcher.fetch(logage, mh)
        color, mag, _ = self.fetcher.photometry(df)

        plt.plot(color, mag, ".", ms=4, color="C0", label=label)
        plt.gca().invert_yaxis()
        plt.xlabel("BP-RP")
        plt.ylabel("G (mag)")
        plt.grid()
        return plt.gcf()
