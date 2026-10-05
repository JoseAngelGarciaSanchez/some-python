"""Lab 3 — Functions, decorators, iterators and generators.

Fluent Python, 2nd ed., chapters 7, 9 and 17. About 18 minutes.

Implement each function until its tests pass::

    uv run pytest tests/test_lab3_generators.py

Tasks 1-4 are the core of the lab, Tasks 5 and 6 are bonuses.
"""

from collections.abc import Callable, Iterable, Iterator

from labs.common import LogRecord


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorate ``func`` so that each call prints ``"<name> took 0.0123s"``.

    Task 1 — A decorator. The wrapper must:

    * accept any positional and keyword arguments, and return ``func``'s result;
    * keep ``func``'s ``__name__`` and ``__doc__`` (see ``functools.wraps``);
    * store the duration of the last call, in seconds, as ``last_duration`` on
      the wrapper, even when ``func`` raises;
    * measure with ``time.perf_counter``, never ``time.time``.
    """
    raise NotImplementedError


def chunked[T](iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items from ``iterable``.

    Task 2 — The iterator protocol. The last chunk may be short. ``iterable``
    may be infinite, so never call ``list()`` on it. Hint: ``iter()``,
    ``itertools.islice`` and a ``while`` loop.

    Once it passes, compare it with ``itertools.batched`` (3.12+), which does
    the same thing in C. What does ``batched`` yield instead of lists?

    >>> list(chunked(range(7), 3))
    [[0, 1, 2], [3, 4, 5], [6]]
    """
    raise NotImplementedError


# Task 3 — A streaming pipeline. The log is too big for memory, so chain lazy
# generators: no list may ever hold more than a handful of records.


def parse(lines: Iterable[str]) -> Iterator[LogRecord]:
    """Lazily turn raw log lines into ``LogRecord`` objects.

    Task 3a. Silently skip comment lines (starting with ``#``) and malformed
    lines: the wrong number of fields, or a field that fails to convert.
    """
    raise NotImplementedError


def only(records: Iterable[LogRecord], level: str) -> Iterator[LogRecord]:
    """Lazily keep the records at ``level``.

    Task 3b.
    """
    raise NotImplementedError


def latency_stats(records: Iterable[LogRecord]) -> dict[str, tuple[int, float]]:
    """Consume ``records`` and return ``{service: (count, mean_latency_ms)}``.

    Task 3c. Round the mean to 2 decimals. Memory must grow with the number
    of services, not with the number of records.
    """
    raise NotImplementedError


def pipeline_stats(n_lines: int = 50_000) -> dict[str, tuple[int, float]]:
    """Return the WARN-level latency stats of ``common.iter_log_lines(n_lines)``.

    Task 3d. Wire the three stages together in one lazy pass.
    """
    raise NotImplementedError


def broken_summary(records: Iterable[LogRecord]) -> tuple[int, float]:
    """Return ``(count, mean_latency_ms)``. Buggy: the mean is always 0."""
    count = sum(1 for _ in records)
    total = sum(record.latency_ms for record in records)
    return count, (total / count if count else 0.0)


def fixed_summary(records: Iterable[LogRecord]) -> tuple[int, float]:
    """Return ``(count, mean_latency_ms)``, or ``(0, 0.0)`` for no records.

    Task 4 — The exhausted iterator. Explain in one sentence why
    ``broken_summary`` always reports a mean of 0. Then fix it without
    materialising ``records`` into a list: that would defeat streaming.
    """
    raise NotImplementedError


def count_by_service_groupby(records: Iterable[LogRecord]) -> dict[str, int]:
    """Return ``{service: number_of_records}`` using ``itertools.groupby``.

    Task 5 (bonus). ``groupby`` only groups *consecutive* equal keys, so the
    input needs preparing first. Then explain in a comment why this is
    strictly worse than a ``collections.Counter`` for this particular job,
    and when ``groupby`` is the right tool.
    """
    raise NotImplementedError


def slow_feature(x: int) -> int:
    """An expensive, pure feature computation."""
    return sum(i * i for i in range(x))


def cached_feature(x: int) -> int:
    """Return ``slow_feature(x)``, computing it at most once per ``x``.

    Task 6 (bonus) — Memoisation. Use ``functools.cache`` (or ``lru_cache``)
    rather than a hand-made dict. Then answer: what happens when the argument
    is a list instead of an int, and why?
    """
    raise NotImplementedError
