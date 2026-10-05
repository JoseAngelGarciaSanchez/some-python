"""
LAB 4 — Threads, processes, the GIL                (~22 min)
Python 3.14. Fluent Python, 2nd ed.: chapters 19, 20 and 21.

!! RUN THIS AS A SCRIPT FROM A TERMINAL, NOT IN A NOTEBOOK !!

    python3.14 lab4_concurrency.py

Process pools re-import the __main__ module in each worker. A notebook has no
importable __main__, so ProcessPoolExecutor either hangs or raises there.
The `if __name__ == "__main__":` guard at the bottom is mandatory for the
same reason — without it, each worker would re-run the whole script and
recursively spawn more workers. Since 3.14 the default start method on Linux
is "forkserver" rather than "fork", which makes that guard load-bearing on
every platform, not just Windows and macOS.

Tasks 1-4 are core, 5 is bonus, 6 is the capstone.
Lab 5 continues into free-threading.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from common import run_checks, timed

# On a laptop these are the same. On a SLURM node, in Docker, or in a k8s pod
# they are NOT: cpu_count() reports the machine, sched_getaffinity() reports
# what you are actually allowed to use. Sizing a pool with cpu_count() on a
# shared cluster is how you get throttled.
N_CPU = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def describe_runtime() -> str:
    """Always print this before a benchmark. Numbers without it are useless."""
    import multiprocessing as mp
    gil = "on" if sys._is_gil_enabled() else "OFF (free-threaded)"
    return (f"{sys.implementation.name} {sys.version.split()[0]} | GIL {gil} | "
            f"{N_CPU} usable CPU(s) | start method: {mp.get_start_method()}")


# ---------------------------------------------------------------------------
# The two workloads. Both must be module-level functions so they can be
# pickled and sent to worker processes.
# ---------------------------------------------------------------------------

def cpu_task(n: int) -> int:
    """Pure-Python CPU work: count primes below n. Holds the GIL throughout."""
    count = 0
    for x in range(2, n):
        limit = int(x ** 0.5)
        for d in range(2, limit + 1):
            if x % d == 0:
                break
        else:
            count += 1
    return count


def io_task(delay: float) -> float:
    """Simulated network call. time.sleep RELEASES the GIL."""
    time.sleep(delay)
    return delay


# ---------------------------------------------------------------------------
# TASK 1 — Three ways to run the same work
#
# Implement the three runners. All three take a function and a list of
# arguments and return the list of results IN INPUT ORDER.
#
# Use concurrent.futures for 1b and 1c — do not hand-roll threading.Thread.
# ---------------------------------------------------------------------------

def run_serial(fn, args: list) -> list:
    raise NotImplementedError


def run_threads(fn, args: list, workers: int = 4) -> list:
    raise NotImplementedError


def run_processes(fn, args: list, workers: int = 4) -> list:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 2 — Measure, then explain
#
# Run benchmark() (called automatically at the bottom) and fill in this table
# from what you actually observe on YOUR machine:
#
#                     serial     threads    processes
#   CPU-bound         ......     ......     ......
#   I/O-bound         ......     ......     ......
#
# Answer in writing:
#   a) Why do threads not speed up cpu_task? Be precise about what the GIL
#      does and does not lock.
#   b) Why are threads sometimes *slower* than serial for cpu_task?
#   c) Why do threads work perfectly well for io_task?
#   d) Processes beat threads on cpu_task but the speedup is below N_CPU.
#      Name two sources of the missing time.
#   e) You replace cpu_task with a numpy matrix multiplication of the same
#      duration. Predict the thread result and justify it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TASK 3 — A race condition
#
# `RacyCounter.increment` is not thread-safe: `+= 1` is load, add, store, and
# the interpreter can switch threads between those steps.
#
# 3a. Run demo_race(RacyCounter) and see whether you can make it fail on your
#     machine. On many machines it will NOT lose a single increment, even
#     with millions of iterations. That is the point: the bug is real and the
#     test passes anyway. WideRacyCounter has the same bug with a wider
#     window and loses ~75% of its increments every time.
# 3b. Implement SafeCounter using threading.Lock as a context manager.
# 3c. In a comment: describe a third approach that needs no lock at all.
# ---------------------------------------------------------------------------

N_INCREMENTS = 200_000
N_WIDE = 20_000


class RacyCounter:
    """The realistic bug: += is load, add, store — three separate bytecodes."""

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1


class WideRacyCounter:
    """The same bug, with the window forced open so it always fires."""

    def __init__(self):
        self.value = 0

    def increment(self):
        tmp = self.value
        time.sleep(0)            # an explicit yield point: hand off the GIL
        self.value = tmp + 1


class SafeCounter:
    def __init__(self):
        raise NotImplementedError

    def increment(self):
        raise NotImplementedError


def hammer(counter, times: int = N_INCREMENTS):
    for _ in range(times):
        counter.increment()


def demo_race(counter_cls, n_threads: int = 4, times: int = N_INCREMENTS) -> int:
    import threading
    counter = counter_cls()
    threads = [threading.Thread(target=hammer, args=(counter, times))
               for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter.value


# ---------------------------------------------------------------------------
# TASK 4 — What breaks in a process pool
#
# 4a. Uncomment the lambda line in _t4 and observe the error. What exactly
#     cannot cross the process boundary, and why? Name three other things
#     that fail the same way.
# 4b. `send_big_payload` measures the cost of shipping data to workers.
#     Run it. What does the wall time tell you about ProcessPoolExecutor and
#     large arrays? What are the two standard fixes?
# 4c. Implement chunked_map: same as executor.map, but pass a chunksize so
#     that 10_000 tiny tasks are not sent one at a time.
# ---------------------------------------------------------------------------

def tiny_task(x: int) -> int:
    return x * x


def chunked_map(fn, args: list, workers: int = 4, chunksize: int = 500) -> list:
    """Like run_processes, but batching arguments into chunks per worker."""
    raise NotImplementedError


# 4d. Print multiprocessing.get_start_method(). On Python 3.14 under Linux you
#     should see "forkserver", not "fork" — the default changed. Explain what
#     that means for a worker function that reads a module-level global set
#     inside `if __name__ == "__main__":`. (Hint: forkserver re-imports the
#     module as __mp_main__, so the guard body never runs in the worker.)


def send_big_payload(n_rows: int = 200_000, workers: int = 2) -> None:
    """Each call ships a large list to a worker. Watch the clock."""
    payload = list(range(n_rows))
    with timed(f"pickling {n_rows}-element list x4"):
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(sum, [payload] * 4))


# ---------------------------------------------------------------------------
# TASK 5 (BONUS) — asyncio
#
# Reimplement the I/O benchmark with asyncio: run all the sleeps concurrently
# in ONE thread and return the results in order.
# Use asyncio.sleep (never time.sleep — say why in a comment) and
# asyncio.gather.
# ---------------------------------------------------------------------------

async def async_io_task(delay: float) -> float:
    raise NotImplementedError


def run_async(delays: list[float]) -> list[float]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 6 (CAPSTONE) — Pick the right tool per stage
#
# A mini ETL with two stages:
#   stage 1: fetch 12 shards   -> I/O-bound  (io_task, 0.15s each)
#   stage 2: parse each shard  -> CPU-bound  (cpu_task, 20_000 each)
#
# Implement etl() so that EACH stage uses the executor that suits it.
# Return the list of stage-2 results. Then compare your total wall time
# against the fully serial version and against using one executor for both.
# ---------------------------------------------------------------------------

SHARDS = 12
FETCH_DELAY = 0.15
PARSE_SIZE = 20_000


def etl() -> list[int]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
def benchmark():
    print(f"\n--- benchmark: {describe_runtime()} ---")
    cpu_args = [300_000] * 4
    io_args = [0.25] * 8

    print(" CPU-bound (4 x count primes below 300k):")
    with timed("   serial"):
        run_serial(cpu_task, cpu_args)
    with timed("   threads(4)"):
        run_threads(cpu_task, cpu_args, 4)
    with timed("   processes(4)"):
        run_processes(cpu_task, cpu_args, 4)

    print(" I/O-bound (8 x sleep 0.25s):")
    with timed("   serial"):
        run_serial(io_task, io_args)
    with timed("   threads(8)"):
        run_threads(io_task, io_args, 8)


# ---------------------------------------------------------------------------
def _t1():
    args = [10_000, 20_000, 5_000]
    expected = [cpu_task(a) for a in args]
    assert run_serial(cpu_task, args) == expected
    assert run_threads(cpu_task, args, 3) == expected, "order not preserved"
    assert run_processes(cpu_task, args, 2) == expected, "order not preserved"


def _t3():
    assert demo_race(SafeCounter, 4) == 4 * N_INCREMENTS


def _t4():
    # 4a: uncomment to see the failure, then re-comment.
    # with ProcessPoolExecutor(max_workers=2) as ex:
    #     list(ex.map(lambda x: x * x, range(4)))
    got = chunked_map(tiny_task, list(range(10_000)), workers=2, chunksize=1000)
    assert got == [x * x for x in range(10_000)]


def _t5():
    out = run_async([0.05, 0.02, 0.03])
    assert out == [0.05, 0.02, 0.03], out


def _t6():
    out = etl()
    assert len(out) == SHARDS
    assert all(v == cpu_task(PARSE_SIZE) for v in out)


if __name__ == "__main__":
    run_checks("Lab 4 — concurrency", [
        ("1  run_serial/threads/processes", _t1),
        ("3  SafeCounter", _t3),
        ("4  chunked_map", _t4),
        ("5  run_async (bonus)", _t5),
        ("6  etl (capstone)", _t6),
    ])
    print(f"\n  RacyCounter     -> {demo_race(RacyCounter, 4):>9,}"
          f"   (expected {4 * N_INCREMENTS:,})")
    print(f"  WideRacyCounter -> "
          f"{demo_race(WideRacyCounter, 4, N_WIDE):>9,}"
          f"   (expected {4 * N_WIDE:,})")
    try:
        benchmark()
    except NotImplementedError:
        print("\n  (benchmark skipped — implement Task 1 first)")
    # send_big_payload()      # uncomment for task 4b
