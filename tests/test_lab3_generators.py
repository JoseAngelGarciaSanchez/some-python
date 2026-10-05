import itertools
import time

import pytest

from labs.common import LogRecord, iter_log_lines
from labs.lab3_generators import (
    broken_summary,
    cached_feature,
    chunked,
    count_by_service_groupby,
    fixed_summary,
    latency_stats,
    only,
    parse,
    pipeline_stats,
    slow_feature,
    timed,
)


def test_timed_returns_result_and_keeps_metadata(capsys):
    @timed
    def work(a, b=2):
        """Multiply."""
        return a * b

    assert work(3, b=4) == 12
    assert (work.__name__, work.__doc__) == ("work", "Multiply.")
    assert work.last_duration >= 0
    assert "work took" in capsys.readouterr().out


def test_timed_measures_calls_that_raise():
    @timed
    def fail():
        time.sleep(0.01)
        raise ValueError

    with pytest.raises(ValueError):
        fail()
    assert fail.last_duration >= 0.01


def test_chunked_yields_lists_with_a_short_tail():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunked_of_nothing_is_empty():
    assert list(chunked([], 3)) == []


def test_chunked_is_lazy():
    assert next(chunked(itertools.count(), 4)) == [0, 1, 2, 3]


def test_parse_skips_comments_and_malformed_lines():
    lines = [
        "# rotated",
        "1700000000,INFO,serve,12.5",
        "1700000001,WARN,train,",
        "bad line",
        "soon,INFO,serve,1.0",
        "1700000002,ERROR,ingest,30.0",
    ]
    assert list(parse(lines)) == [
        LogRecord(1700000000, "INFO", "serve", 12.5),
        LogRecord(1700000002, "ERROR", "ingest", 30.0),
    ]


def test_parse_is_lazy():
    assert len(list(itertools.islice(parse(iter_log_lines(10**9)), 3))) == 3


def test_only_filters_by_level():
    records = [LogRecord(1, "INFO", "serve", 1.0), LogRecord(2, "ERROR", "serve", 2.0)]
    assert list(only(iter(records), "ERROR")) == records[1:]


def test_latency_stats():
    records = [
        LogRecord(1, "WARN", "serve", 10.0),
        LogRecord(2, "WARN", "serve", 20.0),
        LogRecord(3, "WARN", "train", 5.0),
    ]
    assert latency_stats(iter(records)) == {"serve": (2, 15.0), "train": (1, 5.0)}


def test_pipeline_stats():
    stats = pipeline_stats(20_000)
    warn_lines = sum(",WARN," in line for line in iter_log_lines(20_000))
    assert set(stats) == {"ingest", "train", "serve"}
    assert sum(count for count, _ in stats.values()) == warn_lines
    assert stats["train"][1] > stats["serve"][1], "train is the slow service"


RECORDS = [LogRecord(1, "INFO", "serve", 10.0), LogRecord(2, "INFO", "serve", 20.0)]


def test_broken_summary_is_broken():
    assert broken_summary(iter(RECORDS)) == (2, 0.0)


def test_fixed_summary():
    assert fixed_summary(iter(RECORDS)) == (2, 15.0)


def test_fixed_summary_of_nothing():
    assert fixed_summary(iter([])) == (0, 0.0)


def test_count_by_service_groupby():
    records = [
        LogRecord(1, "INFO", "serve", 1.0),
        LogRecord(2, "INFO", "train", 1.0),
        LogRecord(3, "INFO", "serve", 1.0),
    ]
    assert count_by_service_groupby(iter(records)) == {"serve": 2, "train": 1}


def test_cached_feature_computes_each_value_once():
    expected = slow_feature(2_000)
    assert cached_feature(2_000) == expected
    before = cached_feature.cache_info()
    assert cached_feature(2_000) == expected
    after = cached_feature.cache_info()
    assert (after.hits, after.misses) == (before.hits + 1, before.misses)
