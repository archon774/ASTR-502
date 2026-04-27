# ASTR-502

Repository for ASTR 502 stellar isochrone interpolation, single-target fitting, and batch catalog fitting workflows.

## Repository structure

```text
data/
  raw/
    catalogs/
      ASTR502_Mega_Target_List.csv
      ASTR502_Master_Photometry_List.csv
      k2_star_ages.csv
      kepler_star_ages.csv
    isochrones/
      MIST/
      SPOTS/

outputs/
  logs/               # timestamped interpolation logs
  results/            # fit result CSV outputs
  figs/               # generated diagnostic plots

src/astr502/
  domain/
    schemas.py        # fit-result schemas
    stats.py          # chi-square and residual terms
  data/
    paths.py          # canonical repo/data/output paths
    catalogs.py       # catalog loading + priors + observed magnitudes
    utils.py          # logging + SPOT helpers
    readers/
      read_spot_models.py
      read_mist_models.py
  modeling/
    extinction.py     # extinction by band
    interpolate.py    # interpolation grid + optimization (+ optional emcee)
  services/
    fit_runtime.py    # single-target + batch runtime wrappers
    plots.py          # observed-vs-table age scatter plotting
    gyro.py           # compare fitted ages against Kepler ages
    regress.py        # linear regression of interpolated vs Kepler ages

scripts/
  fit_single_star.py  # CLI wrapper for one hostname
  fit_target_list.py  # CLI wrapper for full/subset target-list runs
  fetch_iso.py        # ezpadova isochrone fetching/plot helper classes
  find_mag.py         # photometry + distance merge helper class
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## CLI scripts

### 1) Fit a single star

```bash
python scripts/fit_single_star.py [--mega-csv PATH] [--phot-csv PATH] [--output-csv PATH] [--av-min 0.0] [--av-max 3.0] [--quiet]
```

Notes:
- The script currently prompts interactively for hostname (`Hostname:`).
- Default catalog paths are `data/raw/catalogs/ASTR502_Mega_Target_List.csv` and `data/raw/catalogs/ASTR502_Master_Photometry_List.csv`.
- `--quiet` prints a compact CSV-like line for the fitted target.

### 2) Fit a target list (all or subset)

```bash
python scripts/fit_target_list.py \
  [--mega-csv PATH] [--phot-csv PATH] \
  [--hostnames HOST1 HOST2 ...] \
  [--output-csv PATH] [--stop-on-error] \
  [--workers N] [--parallel-backend threads|processes] \
  [--emcee] [--nwalkers N] [--nsteps N] [--burn-in N] \
  [--quiet]
```

Notes:
- Omitting `--hostnames` runs all hostnames from the mega catalog.
- `--workers > 1` enables concurrent fitting.
- `--parallel-backend` controls concurrency implementation (`processes` default in script).
- `--emcee` enables MCMC sampling in addition to the optimization best fit.

### 3) Isochrone fetch/plot utility classes

```bash
python scripts/fetch_iso.py
```

This file mainly provides reusable classes:
- `IsochroneFetcher`
- `IsochronePlotter`

for querying and visualizing ezpadova isochrones from Python if using Parsec models. Note that currently only SPOTs is fully supported.

### 4) Photometry + distance merge utility class

```bash
python scripts/find_mag.py
```

This file provides `PhotometryMerger`, which can:
- join photometry and distance catalogs,
- compute absolute magnitudes (`G_abs`, `BP_abs`, `RP_abs`),
- save merged output CSVs (default: `data/processed/joined_photometry_and_distances.csv`).

## Additional runnable service modules

These are under `src/astr502/services` and can be run directly:

```bash
python src/astr502/services/plots.py
python src/astr502/services/gyro.py
python src/astr502/services/regress.py
python src/astr502/services/regress.py --comparison-source k2
python src/astr502/services/regress.py --comparison-source external --comparison-csv path/to/comparison.csv
```

- `plots.py` generates observed-vs-table age scatter plots in `outputs/figs/`.
- `gyro.py` compares recent fit ages against Kepler & K2 gyrochronology ages.
- `regress.py` computes a best-fit line between latest interpolated ages and a selected comparison catalog (`kepler_star_ages.csv`, `k2_star_ages.csv`, or an external CSV with `tic_ids` and `st_age` columns).

## Maintainers

Repository owner: James Atkisson  
Email: atk@unc.edu

This was written for the ASTR 502 course at the University of North Carolina at Chapel Hill

Instructor: Dr. Andrew Mann  
