"""Lab 5 — Free-threaded Python (PEP 703, PEP 779).

Python 3.14. About 10 minutes. There is no Fluent Python chapter for this lab:
the 2nd edition (2022) predates the officially supported free-threaded build.
This is the part of the book that is now out of date.

Implement Tasks 1 and 2 until their tests pass::

    uv run pytest tests/test_lab5_freethreading.py

Then run the experiments TWICE with the same free-threaded binary, changing
only the GIL, from the repository root. Keep both outputs side by side: the
comparison IS the exercise::

    uv python install 3.14t
    uv run --no-project --python 3.14t python -X gil=1 -m labs.lab5_freethreading
    uv run --no-project --python 3.14t python -m labs.lab5_freethreading

The first run forces the GIL on, the second leaves it off. With only the
standard build, ``uv run python -m labs.lab5_freethreading`` still works: you
just get the GIL-on column twice. Check the CPU count it prints first: no
speedup is possible on a single-core machine or a one-CPU container.

Tasks 1 and 2 are the core of the lab and Task 4 is a bonus. Tasks 3 and 5
are written.

Task 2 — The speedup table
--------------------------
Fill it in from your two runs::

                          GIL on           GIL off
    pure-Python task      speedup ....     speedup ....
    C-extension task      speedup ....     speedup ....

a) Which cell changes the most between the two columns, and why?
b) Why does the C-extension task barely change? What was already true of it
   before free-threading existed?
c) The free-threaded build costs roughly 5-10% single-threaded. Given your
   numbers, when is that a good trade, and when is it not?
d) Your team's pipeline is 90% pandas and 10% pure-Python glue. Estimate what
   free-threading buys you (Amdahl's law).

Task 3 — Does removing the GIL create new bugs?
-----------------------------------------------
a) Compare the unsynchronised counter between your two runs: how many
   increments are lost each time?
b) Before looking, predict whether concurrent ``list.append`` keeps the right
   length WITHOUT the GIL. Were you right?
c) Write down the rule that reconciles a) and b): which operations does
   CPython guarantee are atomic, and which does it not?

Task 5 — The question that matters
----------------------------------
Your lab publishes a Python library used by other researchers. Should the next
release switch to the free-threaded build? Answer in five lines. Cover whether
it is the default yet, what ecosystem support looks like, the single-thread
cost, and what you would audit in your own code before trusting it under real
parallelism.
"""

import hashlib
import threading
from collections.abc import Callable, Iterable
from typing import NamedTuple


class RuntimeInfo(NamedTuple):
    version: str  # for example "3.14.5"
    freethreaded: bool  # this BINARY was built without a GIL
    gil_enabled: bool  # the GIL is running RIGHT NOW
    cpus: int  # CPUs this process may use


def runtime_info() -> RuntimeInfo:
    """Describe the interpreter this code is running on.

    Task 1 — Know what you are running on. ``freethreaded`` and
    ``gil_enabled`` are different questions, and that is the whole task:
    ``python3.14t -X gil=1`` is a free-threaded build with the GIL switched
    back on. Hints: ``sysconfig.get_config_var("Py_GIL_DISABLED")``,
    ``sys._is_gil_enabled()`` and ``os.process_cpu_count()``.
    """
    raise NotImplementedError


def pure_python_task(n: int = 300_000) -> int:
    """Count the primes below ``n`` in pure Python. Holds the GIL throughout."""
    count = 0
    for x in range(2, n):
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                break
        else:
            count += 1
    return count


def c_extension_task() -> bytes:
    """CPU work inside C code that releases the GIL while it runs.

    It stands in for NumPy, pandas or scikit-learn, which do the same.
    """
    return hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 1_000_000)


class Timing(NamedTuple):
    serial_s: float
    threaded_s: float

    @property
    def speedup(self) -> float:
        return self.serial_s / self.threaded_s


def measure(task: Callable[[], object], n_tasks: int = 4) -> Timing:
    """Time ``n_tasks`` calls of ``task`` serially, then on ``n_tasks`` threads.

    Task 2 — Measure the speedup. Make one untimed warm-up call first: never
    time a cold call. Use ``time.perf_counter`` and a ``ThreadPoolExecutor``.
    """
    raise NotImplementedError


def race_test(n_threads: int = 4, per_thread: int = 200_000) -> tuple[int, int]:
    """Return ``(observed, expected)`` for an unsynchronised shared counter."""

    class Counter:
        value = 0

        def bump(self) -> None:
            self.value += 1  # a load, an add and a store

    counter = Counter()

    def hammer() -> None:
        for _ in range(per_thread):
            counter.bump()

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counter.value, n_threads * per_thread


def list_test(n_threads: int = 4, per_thread: int = 50_000) -> tuple[int, int]:
    """Return ``(observed, expected)`` for concurrent ``list.append`` calls."""
    shared: list[int] = []

    def hammer() -> None:
        for i in range(per_thread):
            shared.append(i)  # noqa: PERF402 - the appends are the experiment

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return len(shared), n_threads * per_thread


def run_in_interpreters[A, R](fn: Callable[[A], R], args: Iterable[A], workers: int = 4) -> list[R]:
    """Call ``fn`` on each argument on a pool of subinterpreters, results in input order.

    Task 4 (bonus) — The third model (PEP 734, new in 3.14). Each worker of a
    ``concurrent.futures.InterpreterPoolExecutor`` is a separate interpreter
    in the SAME process, with its own GIL: it runs pure Python in parallel
    like a process pool, without a process per worker.

    Then answer: what can you NOT pass to a subinterpreter worker, and how
    does that compare with a process pool?
    """
    raise NotImplementedError


def report() -> None:
    info = runtime_info()
    print(
        f"Python {info.version} | free-threaded build: {info.freethreaded}"
        f" | GIL enabled now: {info.gil_enabled} | {info.cpus} usable CPUs"
    )
    if info.cpus < 2:
        print("Only one usable CPU: no speedup is possible, expect flat numbers.")

    print()
    for label, task in [("pure-Python task", pure_python_task), ("C-extension task", c_extension_task)]:
        timing = measure(task)
        print(
            f"{label:<18} serial {timing.serial_s:5.2f} s   4 threads {timing.threaded_s:5.2f} s"
            f"   speedup {timing.speedup:4.2f}x"
        )


if __name__ == "__main__":
    try:
        report()
    except NotImplementedError:
        print("Timings skipped: implement Tasks 1 and 2 first.")

    observed, expected = race_test()
    print(f"\nunsynchronised counter  {observed:>9,} / {expected:,}")
    observed, expected = list_test()
    print(f"concurrent list.append  {observed:>9,} / {expected:,}")
