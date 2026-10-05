"""
LAB 3 — Functions, decorators, iterators, generators      (~18 min)
Fluent Python, 2nd ed.: chapters 7, 9 and 17.

    python lab3_generators.py

Tasks 1-4 are core, 5-6 are bonus.
"""

from common import approx, iter_log_lines, run_checks

# ---------------------------------------------------------------------------
# TASK 1 — A decorator
#
# Write @timed: it prints "<name> took X.XXXs" and returns the wrapped
# function's result unchanged.
#
# Requirements:
#   * works on functions taking any *args / **kwargs,
#   * preserves __name__ and __doc__ (use functools.wraps),
#   * exposes the last measured duration as `fn.last_duration`,
#   * uses time.perf_counter, not time.time.
# ---------------------------------------------------------------------------

def timed(func):
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 2 — Batching
#
# Yield successive lists of at most `size` items from any iterable.
# The last batch may be short. Must work on an infinite iterator, so do NOT
# call list() on the input.
#
#   chunked(range(7), 3) -> [0,1,2], [3,4,5], [6]
#
# Hint: itertools.islice + iter() + a while loop.
#
# Python 3.14 ships itertools.batched, which does exactly this in C. Write it
# by hand first (the point is the iterator protocol), then compare your
# version against itertools.batched at the end. Note one difference: batched
# yields TUPLES, not lists, and rejects n < 1.
# ---------------------------------------------------------------------------

def chunked(iterable, size: int):
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 3 — A streaming pipeline
#
# The log stream is too big to fit in memory. Build a lazy pipeline of
# generators — NO list is allowed to hold more than a handful of items.
#
# 3a. parse(lines): yield (timestamp:int, level:str, service:str, latency:float)
#     Silently skip comment lines (starting with '#') and malformed lines
#     (wrong field count, or a field that fails to convert).
#
# 3b. only(records, level): yield records whose level matches.
#
# 3c. latency_stats(records): consume the stream and return
#     {service: (count, mean_latency)} with the mean rounded to 2 decimals.
#     Must be O(number of services) in memory, not O(number of records).
#
# Then wire them together in pipeline_stats().
# ---------------------------------------------------------------------------

def parse(lines):
    raise NotImplementedError


def only(records, level: str):
    raise NotImplementedError


def latency_stats(records) -> dict[str, tuple[int, float]]:
    raise NotImplementedError


def pipeline_stats(n_lines: int = 50_000) -> dict[str, tuple[int, float]]:
    """WARN-level latency per service, computed in one lazy pass."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 4 — The exhausted-iterator bug
#
# The function below is wrong. It returns 0 for the mean, every time.
# Explain why in one sentence, then fix it WITHOUT materialising the stream
# into a list (that would defeat the point).
#
# Hint: how many times can you iterate a generator?
# ---------------------------------------------------------------------------

def broken_summary(records):
    n = sum(1 for _ in records)
    total = sum(r[3] for r in records)          # <- the bug
    return n, (total / n if n else 0.0)


def fixed_summary(records) -> tuple[int, float]:
    """Return (count, mean latency) in a single pass."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 5 (BONUS) — itertools.groupby
#
# Return {service: count} using itertools.groupby.
# groupby only groups CONSECUTIVE equal keys, so something has to happen to
# the input first. Do that, then explain in a comment why this approach is
# strictly worse than a defaultdict counter for this particular job.
# ---------------------------------------------------------------------------

def count_by_service_groupby(records) -> dict[str, int]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 6 (BONUS) — Memoisation
#
# `slow_feature` is an expensive, pure function. Make repeated calls with the
# same argument free, using functools.cache (or lru_cache). Then answer:
# what happens if the argument is a list instead of an int, and why?
# ---------------------------------------------------------------------------

_CALLS = {"n": 0}


def slow_feature(x: int) -> int:
    _CALLS["n"] += 1
    return sum(i * i for i in range(x))


cached_feature = None    # replace with the memoised version of slow_feature


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _t1():
    @timed
    def work(a, b=2):
        """docstring"""
        return a * b

    assert work(3, b=4) == 12
    assert work.__name__ == "work" and work.__doc__ == "docstring"
    assert isinstance(work.last_duration, float) and work.last_duration >= 0


def _t2():
    from itertools import count
    assert [list(c) for c in chunked(range(7), 3)] == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunked([], 3)) == []
    inf = chunked(count(), 4)          # must not hang
    assert list(next(inf)) == [0, 1, 2, 3]


def _t3():
    recs = list(parse(["# junk",
                       "1700000000,INFO,serve,12.5",
                       "1700000001,WARN,train,",
                       "bad line",
                       "1700000002,ERROR,ingest,30.0"]))
    assert recs == [(1700000000, "INFO", "serve", 12.5),
                    (1700000002, "ERROR", "ingest", 30.0)], recs
    assert [r[1] for r in only(iter(recs), "ERROR")] == ["ERROR"]

    stats = pipeline_stats(20_000)
    assert set(stats) <= {"ingest", "train", "serve"}
    assert all(c > 0 and lat > 0 for c, lat in stats.values())
    # 'train' is the slow service by construction
    assert stats["train"][1] > stats["serve"][1]

    # Laziness check: parse() on an infinite stream must return instantly.
    import itertools
    head = itertools.islice(parse(iter_log_lines(10**9)), 3)
    assert len(list(head)) == 3


def _t4():
    data = [(1, "INFO", "serve", 10.0), (2, "INFO", "serve", 20.0)]
    assert broken_summary(iter(data)) == (2, 0.0)      # confirms the bug
    assert fixed_summary(iter(data)) == (2, 15.0)


def _t5():
    data = [(1, "INFO", "serve", 1.0), (2, "INFO", "train", 1.0),
            (3, "INFO", "serve", 1.0)]
    assert count_by_service_groupby(iter(data)) == {"serve": 2, "train": 1}


def _t6():
    assert cached_feature is not None, "define cached_feature"
    before = _CALLS["n"]
    a = cached_feature(2000)
    b = cached_feature(2000)
    assert a == b == sum(i * i for i in range(2000))
    assert _CALLS["n"] == before + 1, "second call was not served from cache"


if __name__ == "__main__":
    run_checks("Lab 3 — generators", [
        ("1  timed", _t1),
        ("2  chunked", _t2),
        ("3  streaming pipeline", _t3),
        ("4  fixed_summary", _t4),
        ("5  groupby (bonus)", _t5),
        ("6  cached_feature (bonus)", _t6),
    ])
