#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.astr502.services.fit_runtime import fit_target_list_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit all or a subset of target stars.")
    parser.add_argument("--mega-csv", default=None, help="Path to Mega target list CSV")
    parser.add_argument("--phot-csv", default=None, help="Path to photometry CSV")
    parser.add_argument("--hostnames", nargs="*", default=None, help="Optional list of hostnames to fit")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path. Defaults to timestamped file in outputs/results.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first failed target")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Use >1 to process hostnames concurrently.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("threads", "processes"),
        default="threads",
        help="Parallel executor backend when --workers > 1.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Max queued futures when running in parallel. Lower values reduce memory pressure.",
    )
    parser.add_argument("--no-emcee", action="store_true", help="Disable emcee sampling to speed up batch fitting")
    parser.add_argument("--nwalkers", type=int, default=None, help="Override emcee nwalkers")
    parser.add_argument("--nsteps", type=int, default=None, help="Override emcee nsteps")
    parser.add_argument("--burn-in", type=int, default=None, help="Override emcee burn-in")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runtime_kwargs = {
        "hostnames": args.hostnames,
        "output_csv": args.output_csv,
        "continue_on_error": not args.stop_on_error,
        "verbose": not args.quiet,
        "workers": args.workers,
        "parallel_backend": args.parallel_backend,
        "max_in_flight": args.max_in_flight,
        "run_emcee": not args.no_emcee,
    }
    if args.nwalkers is not None:
        runtime_kwargs["nwalkers"] = args.nwalkers
    if args.nsteps is not None:
        runtime_kwargs["nsteps"] = args.nsteps
    if args.burn_in is not None:
        runtime_kwargs["burn_in"] = args.burn_in
    if args.mega_csv:
        runtime_kwargs["mega_csv_path"] = args.mega_csv
    if args.phot_csv:
        runtime_kwargs["phot_csv_path"] = args.phot_csv

    fits, failures = fit_target_list_runtime(**runtime_kwargs)
    output_label = args.output_csv if args.output_csv is not None else "auto-timestamped in outputs/results"
    print(f"success={len(fits)} failures={len(failures)} output={output_label}")


if __name__ == "__main__":
    main()
