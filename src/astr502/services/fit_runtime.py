from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable
from typing import Callable

from src.astr502.data.catalogs import DEFAULT_MEGA_CSV, DEFAULT_PHOT_CSV
from src.astr502.domain.schemas import FitResultSchema
from src.astr502.modeling.interpolate import fit_best_params, load_catalogs, save_fit_results_to_csv


def fit_single_star_runtime(
    hostname: str,
    *,
    mega_csv_path: str = DEFAULT_MEGA_CSV,
    phot_csv_path: str = DEFAULT_PHOT_CSV,
    output_csv: str | None = None,
    **fit_kwargs,
) -> FitResultSchema:
    """Runtime helper to fit one star from the catalog inputs."""
    load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)
    fit, _ = fit_best_params(hostname=hostname, **fit_kwargs)
    if output_csv is not None:
        save_fit_results_to_csv([fit], output_csv=output_csv)
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
    max_in_flight: int | None = None,
    **fit_kwargs,
) -> tuple[list[FitResultSchema], list[tuple[str, str]]]:
    """Runtime helper to fit a user-provided or catalog-derived host list."""
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
                if verbose:
                    print(f"[{hostname}] fit failed: {exc}")
    else:
        completed, failed = _fit_hostnames_parallel(
            hostnames=hostnames,
            mega_csv_path=mega_csv_path,
            phot_csv_path=phot_csv_path,
            worker_count=worker_count,
            parallel_backend=parallel_backend,
            max_in_flight=max_in_flight,
            verbose=verbose,
            fit_kwargs=fit_kwargs,
            continue_on_error=continue_on_error,
        )
        fits.extend(completed)
        failures.extend(failed)

    saved_csv_path: str | None = None
    if fits:
        saved_csv_path = save_fit_results_to_csv(fits, output_csv=output_csv)

    if verbose:
        print(f"Completed fits: {len(fits)} success, {len(failures)} failed")
        if saved_csv_path is not None:
            print(f"Saved successful fits to: {saved_csv_path}")

    return fits, failures


def _fit_hostname_worker_loaded(
    hostname: str,
    *,
    verbose: bool,
    fit_kwargs: dict,
) -> FitResultSchema:
    # Catalogs are preloaded either in the parent (threads) or in each process via initializer.
    fit, _ = fit_best_params(hostname=hostname, verbose=verbose, **fit_kwargs)
    return fit


def _fit_hostnames_parallel(
    *,
    hostnames: Iterable[str],
    mega_csv_path: str,
    phot_csv_path: str,
    worker_count: int,
    parallel_backend: str,
    max_in_flight: int | None,
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

    if max_in_flight is None:
        max_in_flight = max(1, worker_count * 4)
    max_in_flight = max(1, int(max_in_flight))

    if parallel_backend == "processes":
        with executor_type(
            max_workers=worker_count,
            initializer=load_catalogs,
            initargs=(mega_csv_path, phot_csv_path),
        ) as executor:
            _collect_parallel_results(
                host_list=host_list,
                max_in_flight=max_in_flight,
                worker_submit=lambda hostname: executor.submit(
                    _fit_hostname_worker_loaded,
                    hostname,
                    verbose=verbose,
                    fit_kwargs=fit_kwargs,
                ),
                fits_by_hostname=fits_by_hostname,
                failures=failures,
                continue_on_error=continue_on_error,
                verbose=verbose,
            )
    else:
        # Thread workers can share loaded catalogs and isochrone caches.
        load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)
        with executor_type(max_workers=worker_count) as executor:
            _collect_parallel_results(
                host_list=host_list,
                max_in_flight=max_in_flight,
                worker_submit=lambda hostname: executor.submit(
                    _fit_hostname_worker_loaded,
                    hostname,
                    verbose=verbose,
                    fit_kwargs=fit_kwargs,
                ),
                fits_by_hostname=fits_by_hostname,
                failures=failures,
                continue_on_error=continue_on_error,
                verbose=verbose,
            )

    ordered_fits = [fits_by_hostname[h] for h in host_list if h in fits_by_hostname]
    return ordered_fits, failures


def _collect_parallel_results(
    *,
    host_list: list[str],
    max_in_flight: int,
    worker_submit: Callable[[str], concurrent.futures.Future[FitResultSchema]],
    fits_by_hostname: dict[str, FitResultSchema],
    failures: list[tuple[str, str]],
    continue_on_error: bool,
    verbose: bool,
) -> None:
    in_flight: dict[concurrent.futures.Future[FitResultSchema], str] = {}
    host_iter = iter(host_list)

    def _submit_until_full() -> None:
        while len(in_flight) < max_in_flight:
            try:
                hostname = next(host_iter)
            except StopIteration:
                break
            in_flight[worker_submit(hostname)] = hostname

    _submit_until_full()

    while in_flight:
        done, _ = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            hostname = in_flight.pop(future)
            try:
                fits_by_hostname[hostname] = future.result()
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append((hostname, str(exc)))
                if verbose:
                    print(f"[{hostname}] fit failed: {exc}")
        _submit_until_full()
