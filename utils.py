from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from find_mag import PhotometryMerger


logger = logging.getLogger(__name__)


class LoggingUtils:
    """Logging helpers for interpolation workflows."""

    @staticmethod
    def configure_debug_logging(log_dir: str | Path = "logs") -> Path:
        """Configure root logger and return the active log file path."""
        logs_path = Path(log_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = logs_path / f"interpolate_{run_stamp}.log"

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logger.debug("Debug logging initialized at %s", log_file)
        return log_file

    @staticmethod
    def extract_run_stamp_from_logging() -> str:
        """Reuse interpolate log timestamp when available; otherwise create a new one."""
        pattern = re.compile(r"interpolate_(\d{8}_\d{6})\.log$")
        for handler in logging.getLogger().handlers:
            filename = getattr(handler, "baseFilename", None)
            if not filename:
                continue
            match = pattern.search(str(filename))
            if match:
                return match.group(1)
        return time.strftime("%Y%m%d_%H%M%S")


class DataFrameUtils:
    """Common dataframe utilities for matching/interpolation routines."""

    @staticmethod
    def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
        for cand in candidates:
            for col in df.columns:
                if cand.lower() in str(col).lower():
                    return str(col)
        return None

    @staticmethod
    def as_numeric(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out


class IsochroneUtils:
    """Helper methods to normalize SPOT isochrones and create interpolators."""

    @staticmethod
    def extract_metallicity_from_path(file_path: str | Path) -> float:
        """Infer metallicity from SPOT filename style like f000.isoc or fm05.isoc."""
        stem = Path(file_path).stem.lower()
        match = re.search(r"f([pm]?)(\d+)", stem)
        if not match:
            return float("nan")

        sign = match.group(1)
        digits = match.group(2)
        value = float(digits) / 100.0
        if sign == "m":
            value *= -1.0
        return value

    @staticmethod
    def build_color_mag_interpolator(
        isochrone_df: pd.DataFrame,
        color_col: str,
        mag_col: str,
    ):
        """Build scipy.interpolate interp1d(color -> magnitude) for one isochrone."""
        track = isochrone_df[[color_col, mag_col]].dropna().sort_values(color_col)
        if len(track) < 2:
            return None

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

    @staticmethod
    def prepare_isochrone_track(isochrone_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        """Normalize one SPOT isochrone section into CMD-friendly columns."""
        iso = DataFrameUtils.as_numeric(isochrone_df)
        iso_bp_col = DataFrameUtils.find_col(iso, ["BP_mag"])
        iso_rp_col = DataFrameUtils.find_col(iso, ["RP_mag"])
        iso_g_col = DataFrameUtils.find_col(iso, ["G_mag"])
        mass_col = DataFrameUtils.find_col(iso, ["Mass"])

        if not (iso_bp_col and iso_rp_col and iso_g_col):
            raise ValueError(
                "Could not identify BP/RP/G columns in SPOT isochrone section; "
                f"columns were: {list(iso.columns)}"
            )

        work_cols = [iso_bp_col, iso_rp_col, iso_g_col] + ([mass_col] if mass_col else [])
        work = iso[work_cols].copy()
        work["iso_color"] = work[iso_bp_col] - work[iso_rp_col]
        work["iso_mag"] = work[iso_g_col]
        return work, mass_col


class TargetDataLoader:
    """Loader for merged photometry + distance target catalogs."""

    @staticmethod
    def load_targets(phot_csv: str | Path, dist_csv: str | Path) -> pd.DataFrame:
        logger.info("Loading targets from photometry=%s and distances=%s", phot_csv, dist_csv)
        merger = PhotometryMerger()
        merged = merger.join_photometry_and_distances(phot_csv=phot_csv, dist_csv=dist_csv)
        logger.info("Loaded merged target table with %d rows and %d columns", *merged.shape)
        required = {"BP_RP_abs", "G_abs"}
        missing = required.difference(merged.columns)
        if missing:
            logger.error("Merged target table is missing required columns: %s", sorted(missing))
        else:
            valid = merged[list(required)].dropna()
            logger.debug("Targets with finite BP_RP_abs and G_abs: %d / %d", len(valid), len(merged))
        return merged


class ResultsManager:
    """Utilities for run-stamped outputs generated by interpolation fitting."""

    @staticmethod
    def default_plot_save_path(output_dir: str | Path = "figs") -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = LoggingUtils.extract_run_stamp_from_logging()
        return out_dir / f"interpolate_{run_stamp}_candidate_fits.png"

    @staticmethod
    def save_best_fit_candidates(
        best_eval_df: pd.DataFrame,
        output_dir: str | Path = "results",
    ) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = LoggingUtils.extract_run_stamp_from_logging()
        out_path = out_dir / f"interpolate_{run_stamp}_candidate_fits.csv"
        if "hostname" not in best_eval_df.columns:
            logger.warning(
                "Saving best-fit candidates without a 'hostname' column. "
                "Check that Hostname is preserved in upstream merged photometry."
            )
        best_eval_df.to_csv(out_path, index=False)
        logger.info("Wrote best-fit candidate magnitudes to %s", out_path)
        return out_path
