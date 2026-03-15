#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from astr502.services.photometry_merge import PhotometryMerger


def main() -> None:
    parser = argparse.ArgumentParser(description="Join photometry and distance catalogs and compute absolute magnitudes.")
    parser.add_argument("--phot-csv", required=True)
    parser.add_argument("--dist-csv", required=True)
    parser.add_argument("--join-key", default=None)
    parser.add_argument("--how", default="inner")
    parser.add_argument("--output-csv", default="outputs/results/photometry_joined.csv")
    args = parser.parse_args()

    merger = PhotometryMerger()
    df = merger.join_photometry_and_distances(
        phot_csv=args.phot_csv,
        dist_csv=args.dist_csv,
        on=args.join_key,
        how=args.how,
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"rows={len(df)} output={out}")


if __name__ == "__main__":
    main()
