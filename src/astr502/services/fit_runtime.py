from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable
from functools import partial
import logging
import time

from src.astr502.data.catalogs import DEFAULT_MEGA_CSV, DEFAULT_PHOT_CSV
from src.astr502.data.utils import LoggingUtils
from src.astr502.domain.schemas import FitResultSchema
from src.astr502.modeling.interpolate import fit_best_params, load_catalogs, save_fit_results_to_csv

logger = logging.getLogger(__name__)


def fit_single_star_runtime(
    hostname: str,
    *,
    mega_csv_path: str = DEFAULT_MEGA_CSV,
    phot_csv_path: str = DEFAULT_PHOT_CSV,
    output_csv: str | None = None,
    **fit_kwargs,
) -> FitResultSchema:
    """Runtime helper to fit one star from the catalog inputs."""
    run_stamp = LoggingUtils.run_timestamp()
    log_file = LoggingUtils.configure_debug_logging(run_stamp=run_stamp)
    start_time = time.perf_counter()
    logger.info("Starting single-star interpolation run at %s for hostname=%s", run_stamp, hostname)

    load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)
    fit, _ = fit_best_params(hostname=hostname, **fit_kwargs)
    saved_csv_path = save_fit_results_to_csv([fit], output_csv=output_csv, run_stamp=run_stamp)

    elapsed_seconds = time.perf_counter() - start_time
    logger.info(
        "Completed single-star interpolation run at %s in %.3f seconds. output_csv=%s log_file=%s",
        run_stamp,
        elapsed_seconds,
        saved_csv_path,
        log_file,
    )
    return fit


def fit_target_list_runtime(
    *,
    mega_csv_path: str = DEFAULT_MEGA_CSV,
    phot_csv_path: str = DEFAULT_PHOT_CSV,
    hostnames: list[str] | None = None,
    output_csv: str | None = None,
    continue_on_error: bool = True,
    verbose: bool = True,
    workers: int = 1,
    parallel_backend: str = "threads",
    **fit_kwargs,
) -> tuple[list[FitResultSchema], list[tuple[str, str]]]:
    """Runtime helper to fit a user-provided or catalog-derived host list."""
    run_stamp = LoggingUtils.run_timestamp()
    log_file = LoggingUtils.configure_debug_logging(run_stamp=run_stamp)
    start_time = time.perf_counter()
    logger.info("Starting target-list interpolation run at %s", run_stamp)

    load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)

    if hostnames is None:
        import pandas as pd

        mega_df = pd.read_csv(mega_csv_path)
        hostnames = [str(h) for h in mega_df["hostname"].dropna().unique().tolist()]

    fits: list[FitResultSchema] = []
    failures: list[tuple[str, str]] = []
    worker_count = max(1, int(workers))

    if worker_count == 1:
        for hostname in hostnames:
            try:
                fit, _ = fit_best_params(hostname=hostname, verbose=verbose, **fit_kwargs)
                fits.append(fit)
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append((hostname, str(exc)))
                logger.exception("[%s] fit failed", hostname)
                if verbose:
                    print(f"[{hostname}] fit failed: {exc}")
    else:
        completed, failed = _fit_hostnames_parallel(
            hostnames=hostnames,
            mega_csv_path=mega_csv_path,
            phot_csv_path=phot_csv_path,
            worker_count=worker_count,
            parallel_backend=parallel_backend,
            verbose=verbose,
            fit_kwargs=fit_kwargs,
            continue_on_error=continue_on_error,
        )
        fits.extend(completed)
        failures.extend(failed)

    saved_csv_path: str | None = None
    if fits:
        saved_csv_path = save_fit_results_to_csv(fits, output_csv=output_csv, run_stamp=run_stamp)

    elapsed_seconds = time.perf_counter() - start_time
    logger.info(
        (
            "Completed target-list interpolation run at %s in %.3f seconds "
            "with %d successes and %d failures. output_csv=%s log_file=%s"
        ),
        run_stamp,
        elapsed_seconds,
        len(fits),
        len(failures),
        saved_csv_path,
        log_file,
    )

    if verbose:
        logger.info("Completed fits: %d success, %d failed", len(fits), len(failures))
        print(f"Completed fits: {len(fits)} success, {len(failures)} failed")
        if saved_csv_path is not None:
            logger.info("Saved successful fits to: %s", saved_csv_path)
            print(f"Saved successful fits to: {saved_csv_path}")

    return fits, failures


def _fit_hostname_worker(
    hostname: str,
    *,
    verbose: bool,
    fit_kwargs: dict,
) -> FitResultSchema:
    fit, _ = fit_best_params(hostname=hostname, verbose=verbose, **fit_kwargs)
    return fit


def _process_pool_initializer(mega_csv_path: str, phot_csv_path: str) -> None:
    load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)


def _fit_hostnames_parallel(
    *,
    hostnames: Iterable[str],
    mega_csv_path: str,
    phot_csv_path: str,
    worker_count: int,
    parallel_backend: str,
    verbose: bool,
    fit_kwargs: dict,
    continue_on_error: bool,
) -> tuple[list[FitResultSchema], list[tuple[str, str]]]:
    host_list = list(hostnames)
    if parallel_backend not in {"threads", "processes"}:
        raise ValueError("parallel_backend must be one of: threads, processes")

    executor_type: type[concurrent.futures.Executor]
    if parallel_backend == "processes":
        executor_type = concurrent.futures.ProcessPoolExecutor
    else:
        executor_type = concurrent.futures.ThreadPoolExecutor

    fits_by_hostname: dict[str, FitResultSchema] = {}
    failures: list[tuple[str, str]] = []

    worker = partial(_fit_hostname_worker, verbose=verbose, fit_kwargs=fit_kwargs)

    executor_kwargs = {"max_workers": worker_count}
    if parallel_backend == "processes":
        executor_kwargs.update(
            {
                "initializer": _process_pool_initializer,
                "initargs": (mega_csv_path, phot_csv_path),
            }
        )

    with executor_type(**executor_kwargs) as executor:
        futures = {
            executor.submit(worker, hostname): hostname
            for hostname in host_list
        }

        for future in concurrent.futures.as_completed(futures):
            hostname = futures[future]
            try:
                fits_by_hostname[hostname] = future.result()
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append((hostname, str(exc)))
                logger.exception("[%s] fit failed", hostname)
                if verbose:
                    print(f"[{hostname}] fit failed: {exc}")

    ordered_fits = [fits_by_hostname[h] for h in host_list if h in fits_by_hostname]
    return ordered_fits, failures
