import time

import pytest

from labs.lab4_concurrency import (
    N_INCREMENTS,
    PARSE_SIZE,
    SHARDS,
    SafeCounter,
    chunked_map,
    cpu_task,
    demo_race,
    etl,
    io_task,
    run_async,
    run_processes,
    run_serial,
    run_threads,
    tiny_task,
)

ARGS = [10_000, 20_000, 5_000]


@pytest.mark.parametrize("runner", [run_serial, run_threads, run_processes])
def test_runner_returns_results_in_input_order(runner):
    assert runner(cpu_task, ARGS) == [cpu_task(n) for n in ARGS]


def test_run_threads_overlaps_waiting():
    start = time.perf_counter()
    run_threads(io_task, [0.2] * 4, workers=4)
    assert time.perf_counter() - start < 0.6, "the four sleeps did not overlap"


def test_safe_counter_loses_nothing():
    assert demo_race(SafeCounter) == 4 * N_INCREMENTS


def test_chunked_map():
    assert chunked_map(tiny_task, range(10_000), workers=2, chunksize=1_000) == [x * x for x in range(10_000)]


def test_run_async_returns_results_in_input_order():
    assert run_async([0.05, 0.02, 0.03]) == [0.05, 0.02, 0.03]


def test_run_async_overlaps_waiting():
    start = time.perf_counter()
    run_async([0.2, 0.1, 0.15])
    assert time.perf_counter() - start < 0.35, "the three sleeps did not overlap"


def test_etl():
    start = time.perf_counter()
    assert etl() == [cpu_task(PARSE_SIZE)] * SHARDS
    assert time.perf_counter() - start < 1.5, "fetching alone takes 1.8 s when serial"
