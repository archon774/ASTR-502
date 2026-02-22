"""Fit Gaia absolute magnitudes to local MIST isochrones.

This module estimates stellar ages and metallicities from Gaia absolute
magnitudes using fully-downloaded MIST isochrones read via
`read_mist_models.py`.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from find_mag import PhotometryMerger

LOGGER = logging.getLogger(__name__)

REQUIRED_MAG_COLS = ("G_abs", "BP_abs", "RP_abs")
TARGET_ID_COL = "target_id"
DEFAULT_MIST_DIR = Path("isochrones/MIST_v1.2_vvcrit0.0_full_isos")


@dataclass(frozen=True)
class IsochroneGridSpec:
    """Configuration for the (logAge, [M/H]) fitting grid."""

    logage_min: float = 6.0
    logage_max: float = 10.2
    logage_step: float = 0.05
    mh_min: float = -2.0
    mh_max: float = 0.6
    mh_step: float = 0.05


def _validate_target_table(targets: pd.DataFrame) -> None:
    required = {TARGET_ID_COL, *REQUIRED_MAG_COLS}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"targets is missing required column(s): {', '.join(sorted(missing))}")


def _sample_list(values: Iterable[Any], n: int = 10) -> list[Any]:
    values_list = list(values)
    return values_list[:n]


def _missing_reason(row: pd.Series, required_cols: Iterable[str]) -> str:
    missing = [col for col in required_cols if pd.isna(row.get(col))]
    return f"missing: {','.join(missing)}" if missing else "missing: none"


def _load_read_mist_models_module(mist_dir: Path):
    candidates = [
        Path("read_mist_models.py"),
        Path("isochrones/read_mist_models.py"),
        mist_dir / "read_mist_models.py",
        mist_dir.parent / "read_mist_models.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("read_mist_models", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            LOGGER.info("Loaded read_mist_models module from %s", path)
            return module
    raise FileNotFoundError(
        "Could not find read_mist_models.py. Checked: " + ", ".join(str(p) for p in candidates)
    )


def _to_df(block: Any) -> pd.DataFrame:
    if isinstance(block, pd.DataFrame):
        return block.copy()
    if isinstance(block, dict):
        return pd.DataFrame(block)
    return pd.DataFrame(block)


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _parse_mh_from_filename(path: Path) -> Optional[float]:
    match = re.search(r"feh_([pm])(\d+\.\d+)", path.name)
    if not match:
        return None
    sign = -1.0 if match.group(1) == "m" else 1.0
    return sign * float(match.group(2))


def _extract_logage(df: pd.DataFrame, fallback_age: Optional[float]) -> Optional[float]:
    col = _find_column(
        df,
        [
            "log10_isochrone_age_yr",
            "log10_isochrone_age",
            "log_age",
            "logAge",
            "log_age_yr",
        ],
    )
    if col is not None:
        val = pd.to_numeric(df[col], errors="coerce").dropna()
        if not val.empty:
            return float(val.iloc[0])
    return float(fallback_age) if fallback_age is not None else None


def _extract_mh(df: pd.DataFrame, fallback_mh: Optional[float]) -> Optional[float]:
    col = _find_column(df, ["[Fe/H]_init", "[Fe/H]", "feh", "MH", "[M/H]"])
    if col is not None:
        val = pd.to_numeric(df[col], errors="coerce").dropna()
        if not val.empty:
            return float(val.iloc[0])
    return fallback_mh


def _normalize_mist_block(df: pd.DataFrame, logage: float, mh: float) -> Optional[pd.DataFrame]:
    g_col = _find_column(df, ["Gaia_G_EDR3", "Gaia_G_DR2Rev", "Gmag"])
    bp_col = _find_column(df, ["Gaia_BP_EDR3", "Gaia_BP_DR2Rev", "G_BPmag"])
    rp_col = _find_column(df, ["Gaia_RP_EDR3", "Gaia_RP_DR2Rev", "G_RPmag"])

    if not (g_col and bp_col and rp_col):
        return None

    out = pd.DataFrame(
        {
            "Gmag": pd.to_numeric(df[g_col], errors="coerce"),
            "G_BPmag": pd.to_numeric(df[bp_col], errors="coerce"),
            "G_RPmag": pd.to_numeric(df[rp_col], errors="coerce"),
            "logAge": float(logage),
            "MH": float(mh),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Gmag", "G_BPmag", "G_RPmag"])


def _iter_mist_blocks(model_obj: Any) -> Iterable[tuple[Optional[float], Any]]:
    if hasattr(model_obj, "isocmds"):
        ages = list(getattr(model_obj, "ages", []))
        for idx, block in enumerate(model_obj.isocmds):
            age = ages[idx] if idx < len(ages) else None
            yield age, block
        return
    if hasattr(model_obj, "isos"):
        ages = list(getattr(model_obj, "ages", []))
        for idx, block in enumerate(model_obj.isos):
            age = ages[idx] if idx < len(ages) else None
            yield age, block
        return
    raise ValueError("Unknown read_mist_models object: expected `.isocmds` or `.isos`")


class MISTInterpolator:
    def __init__(self, models: dict[tuple[float, float], pd.DataFrame]):
        self._models = models
        self.logages = np.array(sorted({k[0] for k in models.keys()}), dtype=float)
        self.mhs = np.array(sorted({k[1] for k in models.keys()}), dtype=float)

    def __call__(self, logage: float, mh: float, what: Iterable[str]) -> pd.DataFrame:
        age_key = float(self.logages[np.argmin(np.abs(self.logages - float(logage)))])
        mh_key = float(self.mhs[np.argmin(np.abs(self.mhs - float(mh)))])
        model = self._models[(age_key, mh_key)]
        cols = [c for c in what if c in model.columns]
        return model[cols].copy()


def _select_grid(available: np.ndarray, vmin: float, vmax: float, step: float) -> np.ndarray:
    filtered = np.array([v for v in available if vmin <= v <= vmax], dtype=float)
    if filtered.size == 0:
        return filtered
    if step <= 0:
        return filtered
    targets = np.arange(vmin, vmax + 0.5 * step, step)
    selected = sorted({float(filtered[np.argmin(np.abs(filtered - t))]) for t in targets})
    return np.array(selected, dtype=float)


def build_isochrone_interpolator(
    grid_spec: IsochroneGridSpec,
    *,
    extra_columns: Optional[Iterable[str]] = None,
    mist_dir: str | Path = DEFAULT_MIST_DIR,
) -> tuple[MISTInterpolator, np.ndarray, np.ndarray, list[str]]:
    """Load local MIST isochrones and build an interpolator-like interface."""

    mist_path = Path(mist_dir)
    if not mist_path.exists():
        raise FileNotFoundError(f"MIST directory does not exist: {mist_path}")

    read_mist_models = _load_read_mist_models_module(mist_path)

    iso_files = sorted(mist_path.glob("*.iso.cmd"))
    reader_name = "ISOCMD"
    if not iso_files:
        iso_files = sorted(mist_path.glob("*.iso"))
        reader_name = "ISO"
    if not iso_files:
        raise FileNotFoundError(f"No MIST .iso.cmd or .iso files found in {mist_path}")

    if not hasattr(read_mist_models, reader_name):
        raise AttributeError(f"read_mist_models.py does not expose {reader_name}")

    reader = getattr(read_mist_models, reader_name)

    model_map: dict[tuple[float, float], pd.DataFrame] = {}
    for iso_file in iso_files:
        obj = reader(str(iso_file))
        fallback_mh = _parse_mh_from_filename(iso_file)
        for age_hint, block in _iter_mist_blocks(obj):
            block_df = _to_df(block)
            logage = _extract_logage(block_df, fallback_age=age_hint)
            mh = _extract_mh(block_df, fallback_mh=fallback_mh)
            if logage is None or mh is None:
                continue
            norm = _normalize_mist_block(block_df, logage=float(logage), mh=float(mh))
            if norm is None or norm.empty:
                continue
            model_map[(float(logage), float(mh))] = norm

    if not model_map:
        raise RuntimeError("No usable MIST models were loaded (Gaia G/BP/RP columns not found).")

    interpolator = MISTInterpolator(model_map)
    logages = _select_grid(interpolator.logages, grid_spec.logage_min, grid_spec.logage_max, grid_spec.logage_step)
    mhs = _select_grid(interpolator.mhs, grid_spec.mh_min, grid_spec.mh_max, grid_spec.mh_step)

    if logages.size == 0 or mhs.size == 0:
        raise RuntimeError("Requested grid bounds do not overlap loaded MIST isochrone grid.")

    model_cols = ["Gmag", "G_BPmag", "G_RPmag", "evol"]
    if extra_columns:
        for col in extra_columns:
            if col not in model_cols:
                model_cols.append(col)

    LOGGER.info(
        "Loaded MIST isochrones: files=%d, unique_logAge=%d, unique_MH=%d",
        len(iso_files),
        len(interpolator.logages),
        len(interpolator.mhs),
    )
    return interpolator, logages, mhs, model_cols


def prepare_targets_from_csv(
    phot_csv: str,
    dist_csv: str,
    *,
    target_id_col: str = "target_id",
    join_on: Optional[str] = None,
    merge_how: str = "inner",
) -> pd.DataFrame:
    merger = PhotometryMerger()
    merged = merger.join_photometry_and_distances(phot_csv, dist_csv, on=join_on, how=merge_how)

    if target_id_col not in merged.columns:
        merged[target_id_col] = merged.index.astype(str)

    targets = merged.rename(columns={target_id_col: TARGET_ID_COL}).copy()
    return targets


def _coerce_interp_result(model: Any, expected_columns: list[str]) -> Optional[pd.DataFrame]:
    if isinstance(model, pd.DataFrame):
        model_df = model.copy()
    elif isinstance(model, dict):
        model_df = pd.DataFrame(model)
    else:
        model_df = pd.DataFrame(model)

    if model_df.empty:
        return None

    required = [c for c in expected_columns if c in ("Gmag", "G_BPmag", "G_RPmag")]
    for col in required:
        if col not in model_df.columns:
            return None

    return model_df


def _build_model_cache(
    interpolator: MISTInterpolator,
    logage_grid: np.ndarray,
    mh_grid: np.ndarray,
    model_cols: list[str],
) -> dict[tuple[float, float], Optional[dict[str, np.ndarray]]]:
    cache: dict[tuple[float, float], Optional[dict[str, np.ndarray]]] = {}

    for logage in logage_grid:
        for mh in mh_grid:
            key = (float(logage), float(mh))
            try:
                raw_model = interpolator(float(logage), float(mh), what=model_cols)
            except Exception:
                cache[key] = None
                continue

            model_df = _coerce_interp_result(raw_model, model_cols)
            if model_df is None:
                cache[key] = None
                continue

            finite_rows = np.isfinite(model_df[["Gmag", "G_BPmag", "G_RPmag"]]).all(axis=1)
            if finite_rows.sum() == 0:
                cache[key] = None
                continue

            model_df = model_df.loc[finite_rows].reset_index(drop=True)
            cache[key] = {
                "Gmag": model_df["Gmag"].to_numpy(dtype=float),
                "G_BPmag": model_df["G_BPmag"].to_numpy(dtype=float),
                "G_RPmag": model_df["G_RPmag"].to_numpy(dtype=float),
                "phase": np.arange(len(model_df), dtype=int),
            }

    return cache


def _fit_single_star(star_row, model_cache, logage_grid, mh_grid, *, delta_chi2_1sig_joint, store_surface):
    target_id = star_row[TARGET_ID_COL]
    obs_g, obs_bp, obs_rp = float(star_row["G_abs"]), float(star_row["BP_abs"]), float(star_row["RP_abs"])
    e_g, e_bp, e_rp = float(star_row["e_G_abs"]), float(star_row["e_BP_abs"]), float(star_row["e_RP_abs"])

    chi2_surface = np.full((len(logage_grid), len(mh_grid)), np.nan, dtype=float)
    phase_surface = np.full((len(logage_grid), len(mh_grid)), -1, dtype=int)

    best_chi2, best_phase, best_logage, best_mh = np.inf, np.nan, np.nan, np.nan
    n_grid_evaluated = 0

    for i, logage in enumerate(logage_grid):
        for j, mh in enumerate(mh_grid):
            model = model_cache.get((float(logage), float(mh)))
            if model is None:
                continue

            chi2_points = (
                ((obs_g - model["Gmag"]) / e_g) ** 2
                + ((obs_bp - model["G_BPmag"]) / e_bp) ** 2
                + ((obs_rp - model["G_RPmag"]) / e_rp) ** 2
            )

            if chi2_points.size == 0 or not np.isfinite(chi2_points).any():
                continue

            k = int(np.nanargmin(chi2_points))
            chi2_min = float(chi2_points[k])
            if not np.isfinite(chi2_min):
                continue

            chi2_surface[i, j] = chi2_min
            phase_surface[i, j] = int(model["phase"][k])
            n_grid_evaluated += 1

            if chi2_min < best_chi2:
                best_chi2, best_logage, best_mh, best_phase = chi2_min, float(logage), float(mh), int(model["phase"][k])

    if not np.isfinite(best_chi2):
        row = {
            "target_id": target_id,
            "best_logAge": np.nan,
            "best_age_yr": np.nan,
            "best_MH": np.nan,
            "best_chi2": np.nan,
            "best_phase_index": np.nan,
            "n_grid_evaluated": int(n_grid_evaluated),
            "logAge_lo_1sig": np.nan,
            "logAge_hi_1sig": np.nan,
            "MH_lo_1sig": np.nan,
            "MH_hi_1sig": np.nan,
            "fit_status": "no_valid_iso",
        }
        return row, (chi2_surface if store_surface else None)

    in_1sig = np.isfinite(chi2_surface) & (chi2_surface <= best_chi2 + float(delta_chi2_1sig_joint))
    if np.any(in_1sig):
        age_idx, mh_idx = np.where(in_1sig)
        logage_lo, logage_hi = float(logage_grid[np.min(age_idx)]), float(logage_grid[np.max(age_idx)])
        mh_lo, mh_hi = float(mh_grid[np.min(mh_idx)]), float(mh_grid[np.max(mh_idx)])
    else:
        logage_lo = logage_hi = mh_lo = mh_hi = np.nan

    row = {
        "target_id": target_id,
        "best_logAge": best_logage,
        "best_age_yr": float(10.0**best_logage),
        "best_MH": best_mh,
        "best_chi2": best_chi2,
        "best_phase_index": best_phase,
        "n_grid_evaluated": int(n_grid_evaluated),
        "logAge_lo_1sig": logage_lo,
        "logAge_hi_1sig": logage_hi,
        "MH_lo_1sig": mh_lo,
        "MH_hi_1sig": mh_hi,
        "fit_status": "ok",
    }
    return row, (chi2_surface if store_surface else None)


def fit_targets_to_isochrones(
    targets: pd.DataFrame,
    *,
    logage_min: float = 6.0,
    logage_max: float = 10.2,
    logage_step: float = 0.05,
    mh_min: float = -2.0,
    mh_max: float = 0.6,
    mh_step: float = 0.05,
    default_e_g: float = 0.05,
    default_e_bp: float = 0.08,
    default_e_rp: float = 0.08,
    delta_chi2_1sig_joint: float = 2.30,
    store_surfaces: bool = False,
    log_level: int = logging.INFO,
    mist_dir: str | Path = DEFAULT_MIST_DIR,
) -> tuple[pd.DataFrame, Optional[dict[str, Any]]]:
    _validate_target_table(targets)
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    t_start = time.perf_counter()
    working = targets.copy()

    for col in REQUIRED_MAG_COLS:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if "e_G_abs" not in working.columns:
        working["e_G_abs"] = float(default_e_g)
    if "e_BP_abs" not in working.columns:
        working["e_BP_abs"] = float(default_e_bp)
    if "e_RP_abs" not in working.columns:
        working["e_RP_abs"] = float(default_e_rp)

    for err_col, default_val in (("e_G_abs", default_e_g), ("e_BP_abs", default_e_bp), ("e_RP_abs", default_e_rp)):
        numeric = pd.to_numeric(working[err_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        working[err_col] = numeric.fillna(float(default_val))
        non_positive = working[err_col] <= 0
        if non_positive.any():
            working.loc[non_positive, err_col] = float(default_val)

    valid_mask = working[list(REQUIRED_MAG_COLS)].notna().all(axis=1)
    valid_targets = working.loc[valid_mask].copy()
    invalid_targets = working.loc[~valid_mask].copy()

    if LOGGER.isEnabledFor(logging.DEBUG) and not invalid_targets.empty:
        invalid_examples = [{"target_id": row[TARGET_ID_COL], "reason": _missing_reason(row, REQUIRED_MAG_COLS)} for _, row in invalid_targets.iterrows()]
        LOGGER.debug("Invalid target samples (up to 10): %s", _sample_list(invalid_examples, n=10))

    grid_spec = IsochroneGridSpec(
        logage_min=logage_min,
        logage_max=logage_max,
        logage_step=logage_step,
        mh_min=mh_min,
        mh_max=mh_max,
        mh_step=mh_step,
    )

    interpolator, logage_grid, mh_grid, model_cols = build_isochrone_interpolator(grid_spec, mist_dir=mist_dir)
    model_cache = _build_model_cache(interpolator, logage_grid, mh_grid, model_cols)

    results: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"logAge_grid": logage_grid.copy(), "MH_grid": mh_grid.copy(), "surfaces": {}}

    for _, row in invalid_targets.iterrows():
        results.append(
            {
                "target_id": row[TARGET_ID_COL],
                "best_logAge": np.nan,
                "best_age_yr": np.nan,
                "best_MH": np.nan,
                "best_chi2": np.nan,
                "best_phase_index": np.nan,
                "n_grid_evaluated": 0,
                "logAge_lo_1sig": np.nan,
                "logAge_hi_1sig": np.nan,
                "MH_lo_1sig": np.nan,
                "MH_hi_1sig": np.nan,
                "fit_status": "insufficient_data",
            }
        )

    for _, row in valid_targets.iterrows():
        fitted, surface = _fit_single_star(
            row,
            model_cache,
            logage_grid,
            mh_grid,
            delta_chi2_1sig_joint=delta_chi2_1sig_joint,
            store_surface=store_surfaces,
        )
        results.append(fitted)
        if store_surfaces and surface is not None:
            diagnostics["surfaces"][str(row[TARGET_ID_COL])] = surface

    result_df = pd.DataFrame(results)
    LOGGER.info("Finished fitting %d targets in %.3fs", len(result_df), time.perf_counter() - t_start)

    if not store_surfaces:
        diagnostics = None

    return result_df, diagnostics


__all__ = [
    "IsochroneGridSpec",
    "build_isochrone_interpolator",
    "prepare_targets_from_csv",
    "fit_targets_to_isochrones",
]
