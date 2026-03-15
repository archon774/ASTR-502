# ASTR-502

Repository for ASTR 502 stellar isochrone interpolation and target fitting workflows.

## Current structure

```text
src/astr502/
  domain/
    schemas.py              # fit result dataclasses
    stats.py                # chi-square terms
  data/
    catalogs.py             # catalog loading + priors + observed magnitudes
    utils.py                # SPOT file discovery + helper utilities
    readers/
      read_spot_models.py
      read_mist_models.py
  modeling/
    extinction.py           # extinction per photometric band
    interpolate.py          # interpolation + optimizer core
  services/
    fit_runtime.py          # single-target and batch runtime wrappers
    fetch_iso.py            # ezpadova fetcher + plotter classes
    photometry_merge.py     # catalog merge + abs mag derivation

scripts/
  fit_single_star.py
  fit_target_list.py
  fetch_iso.py
  find_mag.py

data/raw/isochrones/
  MIST/
  SPOTS/
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Use module scripts with `PYTHONPATH=src` so imports resolve from `src/astr502`:

```bash
PYTHONPATH=src python scripts/fit_single_star.py <hostname> --mega-csv <mega.csv> --phot-csv <phot.csv>
PYTHONPATH=src python scripts/fit_target_list.py --mega-csv <mega.csv> --phot-csv <phot.csv>
PYTHONPATH=src python scripts/find_mag.py --phot-csv <phot.csv> --dist-csv <dist.csv>
```

> Note: default catalog paths can be configured via `ASTR502_MEGA_CSV` and `ASTR502_PHOT_CSV`.
