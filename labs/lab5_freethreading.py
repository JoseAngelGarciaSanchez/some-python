"""
LAB 5 — Free-threaded Python (PEP 703 / PEP 779)          (~10 min)
Python 3.14. This lab has no chapter in Fluent Python: the 2nd edition was
published in 2022, three years before the free-threaded build became
officially supported. This is the part of the book that is now out of date.

THE POINT OF THIS LAB IS TO RUN IT TWICE, WITH THE SAME BINARY:

    python3.14t -X gil=1  lab5_freethreading.py     # GIL forced ON
    python3.14t           lab5_freethreading.py     # GIL OFF

Same machine, same interpreter, same code. The only variable is the GIL.
Keep both outputs side by side — the comparison IS the exercise.

If you only have the ordinary build, `python3.14 lab5_freethreading.py`
still runs; you will just get the GIL-on column twice.

Getting the free-threaded build:
    uv python install 3.14t          # then use `uv run --python 3.14t ...`
    # or download the "freethreaded" installer from python.org

NOTE ON HARDWARE: none of this shows a speedup on a single-core machine or a
1-CPU container. Check the core count printed by Task 1 before drawing any
conclusion from Task 2.
"""

import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from common import run_checks, timed

N_CPU = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


# ---------------------------------------------------------------------------
# TASK 1 — Know what you are running on
#
# Return a dict with these four keys:
#   'version'       -> e.g. '3.14.4'
#   'freethreaded'  -> True if this BINARY is a free-threaded build
#   'gil_enabled'   -> True if the GIL is active RIGHT NOW
#   'cpus'          -> usable cores
#
# The middle two are NOT the same question, and that is the whole task.
# A free-threaded binary launched with `-X gil=1` is a free-threaded build
# with the GIL switched back on.
#
# Hints: sys._is_gil_enabled()  and  sysconfig.get_config_var('Py_GIL_DISABLED')
# ---------------------------------------------------------------------------

def runtime_info() -> dict:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The two workloads.
# ---------------------------------------------------------------------------

def pure_python_task(n: int = 300_000) -> int:
    """Pure-Python CPU work. Holds the GIL for its whole duration."""
    count = 0
    for x in range(2, n):
        limit = int(x ** 0.5)
        for d in range(2, limit + 1):
            if x % d == 0:
                break
        else:
            count += 1
    return count


def c_extension_task(_ignored=None) -> bytes:
    """CPU work inside a C extension that RELEASES the GIL while it runs.
    Stands in for numpy / pandas / sklearn, which do the same thing."""
    return hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 1_000_000)


# ---------------------------------------------------------------------------
# TASK 2 — Measure the speedup
#
# Run `fn` `n_tasks` times serially, then again in a ThreadPoolExecutor with
# `n_tasks` workers. Return (serial_seconds, threaded_seconds, speedup).
#
# Then run this file under BOTH interpreters and fill in the table:
#
#                          GIL on          GIL off
#   pure_python_task       speedup ....    speedup ....
#   c_extension_task       speedup ....    speedup ....
#
# Answer in writing:
#   a) Which cell changes the most between the two columns, and why?
#   b) Why does c_extension_task barely change? What was already true of it
#      before free-threading existed?
#   c) Free-threaded Python carries roughly 5-10% single-thread overhead.
#      Given your numbers, when is that a good trade and when is it not?
#   d) Your team's pipeline is 90% pandas and 10% pure-Python glue. Estimate
#      what free-threading buys you. (Amdahl's law.)
# ---------------------------------------------------------------------------

def measure(fn, n_tasks: int = 4) -> tuple[float, float, float]:
    """Return (serial_s, threaded_s, speedup)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 3 — Does removing the GIL create new bugs?
#
# 3a. Run race_test() under both interpreters. Compare the number of lost
#     increments.
# 3b. `shared_list.append` is called from 4 threads at once in list_test().
#     Predict whether the list ends up with the right length WITHOUT the GIL,
#     then check. Was your prediction right?
# 3c. Write down the rule that reconciles 3a and 3b: which operations does
#     CPython guarantee are atomic, and which does it not?
#
# You do not need to write code for this task — read it and run it — but you
# DO need to write down the answer to 3c.
# ---------------------------------------------------------------------------

def race_test(n_threads: int = 4, per_thread: int = 200_000) -> tuple[int, int]:
    """Returns (observed, expected) for an unsynchronised counter."""
    class Counter:
        def __init__(self):
            self.value = 0

        def bump(self):
            self.value += 1        # load, add, store — three bytecodes

    c = Counter()

    def hammer():
        for _ in range(per_thread):
            c.bump()

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return c.value, n_threads * per_thread


def list_test(n_threads: int = 4, per_thread: int = 50_000) -> tuple[int, int]:
    """Returns (observed, expected) for concurrent list.append."""
    shared: list[int] = []

    def hammer():
        for i in range(per_thread):
            shared.append(i)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(shared), n_threads * per_thread


# ---------------------------------------------------------------------------
# TASK 4 (BONUS) — The third model: subinterpreters (PEP 734, new in 3.14)
#
# Python 3.14 also added `concurrent.interpreters` and an
# InterpreterPoolExecutor. Each worker is a separate interpreter inside the
# SAME process, with its own GIL — so it parallelises like processes, but
# without a full process per worker.
#
# Implement run_in_interpreters using InterpreterPoolExecutor, the same way
# you used ThreadPoolExecutor. Then answer: what can you NOT pass to a
# subinterpreter worker, and how does that compare to a process pool?
#
# from concurrent.futures import InterpreterPoolExecutor
# ---------------------------------------------------------------------------

def run_in_interpreters(fn, args: list, workers: int = 4) -> list:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 5 — The written question that matters
#
# Your lab publishes a Python library used by other researchers.
# Should you switch it to the free-threaded build for the next release?
# Write five lines. Address: is it the default yet, what does the ecosystem
# support look like, what is the single-thread cost, and what would you have
# to audit in your own code before trusting it under real parallelism?
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def _t1():
    info = runtime_info()
    assert set(info) >= {"version", "freethreaded", "gil_enabled", "cpus"}
    assert isinstance(info["freethreaded"], bool)
    assert isinstance(info["gil_enabled"], bool)
    assert info["cpus"] >= 1
    assert info["version"].startswith("3.")
    # A non-free-threaded build can never have the GIL off.
    if not info["freethreaded"]:
        assert info["gil_enabled"] is True


def _t2():
    serial, threaded, speedup = measure(pure_python_task, 2)
    assert serial > 0 and threaded > 0
    assert abs(speedup - serial / threaded) < 1e-6, "speedup must be serial/threaded"


def _t4():
    out = run_in_interpreters(pure_python_task, [50_000, 60_000], workers=2)
    assert out == [pure_python_task(50_000), pure_python_task(60_000)]


def report():
    info = runtime_info()
    print(f"\n  Python {info['version']} | free-threaded build: {info['freethreaded']}"
          f" | GIL enabled now: {info['gil_enabled']} | {info['cpus']} CPU(s)")
    if info["cpus"] < 2:
        print("  !! only one usable core — no speedup is possible here,"
              " the numbers below are expected to be flat")

    print("\n  pure-Python CPU work (4 tasks):")
    s, t, sp = measure(pure_python_task, 4)
    print(f"    serial {s:6.2f}s   threads {t:6.2f}s   speedup {sp:5.2f}x")

    print("  C-extension CPU work, releases the GIL (4 tasks):")
    s, t, sp = measure(c_extension_task, 4)
    print(f"    serial {s:6.2f}s   threads {t:6.2f}s   speedup {sp:5.2f}x")

    got, exp = race_test()
    print(f"\n  unsynchronised counter: {got:,} / {exp:,}  (lost {exp - got:,})")
    got, exp = list_test()
    print(f"  concurrent list.append: {got:,} / {exp:,}  (lost {exp - got:,})")


if __name__ == "__main__":
    run_checks("Lab 5 — free-threading", [
        ("1  runtime_info", _t1),
        ("2  measure", _t2),
        ("4  run_in_interpreters (bonus)", _t4),
    ])
    try:
        report()
    except NotImplementedError:
        print("  (report skipped — implement Tasks 1 and 2 first)")
