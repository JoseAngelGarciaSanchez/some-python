"""Lab 4 — Threads, processes and the GIL.

Fluent Python, 2nd ed., chapters 19, 20 and 21. About 22 minutes.

Implement the functions until their tests pass, then run the experiments::

    uv run pytest tests/test_lab4_concurrency.py
    uv run python -m labs.lab4_concurrency

Run this from a terminal, never from a notebook. Process pools re-import the
main module in every worker, and a notebook has no importable main module, so
``ProcessPoolExecutor`` hangs or fails there. For the same reason the
``if __name__ == "__main__":`` guard at the bottom is mandatory. Since 3.14 no
platform starts workers with ``fork`` by default (Linux uses ``forkserver``,
macOS and Windows use ``spawn``), so the guard matters everywhere.

Tasks 1, 3 and 4 are the core of the lab, Task 5 is a bonus and Task 6 is the
capstone. Task 2 is written: fill in its table even if you run out of time
for the code. Lab 5 continues with free-threading.

Task 2 — Measure, then explain
------------------------------
Run the experiments and fill in this table with what YOUR machine reports::

                  serial    threads    processes
    CPU-bound     ......    .......    .........
    I/O-bound     ......    .......

a) Why do threads not speed up ``cpu_task``? Be precise about what the GIL
   does and does not lock.
b) Why are threads sometimes *slower* than serial for ``cpu_task``?
c) Why do threads work perfectly well for ``io_task``?
d) Processes beat threads on ``cpu_task``, but the speedup stays below the
   number of cores. Name two places the missing time goes.
e) You replace ``cpu_task`` with a NumPy matrix product of the same duration.
   Predict the thread result and justify it.
"""

import multiprocessing
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

from labs.common import timed

# The CPUs this process may actually run on. On a laptop it equals
# os.cpu_count(); on a SLURM node or a container pinned to a CPU set it can be
# far lower. Sizing a pool with os.cpu_count() on a shared node gets you throttled.
CPUS = os.process_cpu_count() or 1


def describe_runtime() -> str:
    """Describe the interpreter and machine. Print it before any benchmark."""
    gil = "on" if sys._is_gil_enabled() else "OFF (free-threaded)"
    return (
        f"Python {sys.version.split()[0]} | GIL {gil} | {CPUS} usable CPUs"
        f" | start method: {multiprocessing.get_start_method()}"
    )


# Both workloads live at module level, so they can be pickled to worker processes.


def cpu_task(n: int) -> int:
    """Count the primes below ``n`` in pure Python. Holds the GIL throughout."""
    count = 0
    for x in range(2, n):
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                break
        else:
            count += 1
    return count


def io_task(delay: float) -> float:
    """Simulate a network call. ``time.sleep`` releases the GIL while it waits."""
    time.sleep(delay)
    return delay


def run_serial[A, R](fn: Callable[[A], R], args: Iterable[A]) -> list[R]:
    """Call ``fn`` on each argument, one after the other.

    Task 1 — Three ways to run the same work. All three runners return the
    results in input order. Use ``concurrent.futures`` for the two pooled
    runners, never raw ``threading.Thread`` objects.
    """
    raise NotImplementedError


def run_threads[A, R](fn: Callable[[A], R], args: Iterable[A], workers: int = 4) -> list[R]:
    """Like ``run_serial``, on a pool of ``workers`` threads."""
    raise NotImplementedError


def run_processes[A, R](fn: Callable[[A], R], args: Iterable[A], workers: int = 4) -> list[R]:
    """Like ``run_serial``, on a pool of ``workers`` processes."""
    raise NotImplementedError


N_INCREMENTS = 200_000
N_WIDE = 20_000


class RacyCounter:
    """The realistic bug: ``+=`` is a load, an add and a store."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1


class WideRacyCounter:
    """The same bug, with the window forced open so that it fires every time."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        current = self.value
        time.sleep(0)  # an explicit yield point: hand the GIL to another thread
        self.value = current + 1


class SafeCounter:
    """A counter that never loses an increment.

    Task 3 — A race condition.

    a) Run the experiments: does ``RacyCounter`` lose increments on your
       machine? On many machines it loses none, even over millions of
       increments. The bug is real and the test passes anyway.
       ``WideRacyCounter`` has the same bug with a wider window and loses
       most of its increments, every run.
    b) Implement this class, using a ``threading.Lock`` as a context manager.
    c) In a comment, describe a third approach that needs no lock at all.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def increment(self) -> None:
        raise NotImplementedError


def demo_race(counter_cls: type, n_threads: int = 4, times: int = N_INCREMENTS) -> int:
    """Increment one shared counter ``times`` times from each of ``n_threads`` threads."""
    counter = counter_cls()

    def hammer() -> None:
        for _ in range(times):
            counter.increment()

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counter.value


def tiny_task(x: int) -> int:
    return x * x


def chunked_map[A, R](fn: Callable[[A], R], args: Iterable[A], workers: int = 4, chunksize: int = 500) -> list[R]:
    """Like ``run_processes``, but ship the arguments to workers in chunks.

    Task 4 — What breaks in a process pool.

    a) Uncomment the lambda line at the bottom of this file and run it. What
       exactly cannot cross the process boundary, and why? Name three other
       things that fail the same way.
    b) Uncomment ``send_big_payload()`` and run it. What does the wall time
       tell you about shipping large data to a process pool? What are the two
       standard fixes?
    c) Implement this function, so that 10,000 tiny tasks are not sent to the
       workers one at a time.
    d) Since 3.14 no platform defaults to ``fork``. What does that mean for a
       worker that reads a module-level global assigned inside
       ``if __name__ == "__main__":``?
    """
    raise NotImplementedError


def send_big_payload(n_rows: int = 200_000, workers: int = 2) -> None:
    """Sum a large list in worker processes four times, and time it."""
    payload = list(range(n_rows))
    with timed(f"4 x sum of a {n_rows:,}-item list"), ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(sum, [payload] * 4))


async def async_io_task(delay: float) -> float:
    """The coroutine version of ``io_task``.

    Task 5 (bonus) — asyncio. Use ``asyncio.sleep``, and say in a comment why
    ``time.sleep`` would be a disaster here.
    """
    raise NotImplementedError


def run_async(delays: Iterable[float]) -> list[float]:
    """Run ``async_io_task`` for every delay concurrently, in a single thread.

    Return the results in input order. Use ``asyncio.gather`` inside
    ``asyncio.run``.
    """
    raise NotImplementedError


SHARDS = 12
FETCH_DELAY = 0.15
PARSE_SIZE = 20_000


def etl() -> list[int]:
    """Fetch ``SHARDS`` shards, then parse each one. Return the parse results.

    Task 6 (capstone) — The right tool for each stage.

    * Fetching a shard is ``io_task(FETCH_DELAY)``: I/O-bound.
    * Parsing a shard is ``cpu_task(PARSE_SIZE)``: CPU-bound.

    Give each stage the executor that suits it. Then time it against a fully
    serial version, and against one executor shared by both stages.
    """
    raise NotImplementedError


def benchmark() -> None:
    cpu_args = [300_000] * 4
    io_args = [0.25] * 8

    print("\nCPU-bound: 4 x count the primes below 300,000")
    with timed("serial"):
        run_serial(cpu_task, cpu_args)
    with timed("threads (4)"):
        run_threads(cpu_task, cpu_args, 4)
    with timed("processes (4)"):
        run_processes(cpu_task, cpu_args, 4)

    print("\nI/O-bound: 8 x sleep 0.25 s")
    with timed("serial"):
        run_serial(io_task, io_args)
    with timed("threads (8)"):
        run_threads(io_task, io_args, 8)


if __name__ == "__main__":
    print(describe_runtime())

    print(f"\nRacyCounter      {demo_race(RacyCounter):>9,} / {4 * N_INCREMENTS:,}")
    print(f"WideRacyCounter  {demo_race(WideRacyCounter, times=N_WIDE):>9,} / {4 * N_WIDE:,}")

    try:
        benchmark()
    except NotImplementedError:
        print("\nBenchmark skipped: implement Task 1 first.")

    # run_processes(lambda x: x * x, range(4))  # Task 4a
    # send_big_payload()  # Task 4b
