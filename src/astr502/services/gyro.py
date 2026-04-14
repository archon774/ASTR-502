from __future__ import annotations

from src.astr502.data.paths import KEPLER_AGES, K2_AGES, OUTPUT_RESULTS_DIR

LATEST_RESULTS = OUTPUT_RESULTS_DIR / "interpolate_20260409_125607_candidate_fits.csv"

def compare_gyro_ages(

) -> None:
    """Compare gyro ages from K2 and Kepler catalogs."""
    k2_df = K2_AGES.read_csv()
    kepler_df = KEPLER_AGES.read_csv()

    k2_ages = k2_df["age_posterior"].values
    kepler_ages = kepler_df["age_posterior"].values



    for k2_age, kepler_age in zip(k2_ages, kepler_ages):
        print(f"K2 age: {k2_age:.3f} Gyr, Kepler age: {kepler_age:.3f} Gyr")