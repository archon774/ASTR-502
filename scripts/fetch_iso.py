#!/usr/bin/env python3
from __future__ import annotations

import argparse

from astr502.services.fetch_iso import IsochroneFetcher, IsochronePlotter


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and optionally plot one ezpadova isochrone.")
    parser.add_argument("--logage", type=float, required=True)
    parser.add_argument("--mh", type=float, required=True)
    parser.add_argument("--photsys", default="gaiaEDR3")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--label", default="Isochrone")
    args = parser.parse_args()

    fetcher = IsochroneFetcher(photsys=args.photsys)
    df = fetcher.fetch(logage=args.logage, mh=args.mh)
    print(f"rows={len(df)} cols={len(df.columns)}")

    if args.plot:
        from pathlib import Path

        out = Path("outputs/results/isochrone_plot.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter = IsochronePlotter(fetcher)
        fig = plotter.plot(logage=args.logage, mh=args.mh, label=args.label)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved={out}")


if __name__ == "__main__":
    main()
