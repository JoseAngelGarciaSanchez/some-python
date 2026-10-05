from dataclasses import FrozenInstanceError

import pytest

from labs.common import DOCS, make_runs
from labs.lab2_dicts_sets import (
    NormalizedDict,
    RunKey,
    accuracy_stats,
    build_index,
    group_by_model,
    schema_diff,
    search,
)

RUNS = make_runs(400, seed=11)


def test_group_by_model_covers_every_run():
    groups = group_by_model(RUNS)
    assert type(groups) is dict, "return a plain dict, not a defaultdict"
    assert set(groups) == {run.model for run in RUNS}
    assert sum(map(len, groups.values())) == len(RUNS)


def test_group_by_model_keeps_input_order():
    assert group_by_model(RUNS)["xgboost"] == [run for run in RUNS if run.model == "xgboost"]


def test_accuracy_stats():
    stats = accuracy_stats(RUNS)
    xgboost = [run.accuracy for run in RUNS if run.model == "xgboost"]
    assert set(stats) == {run.model for run in RUNS}
    assert stats["xgboost"] == (len(xgboost), pytest.approx(sum(xgboost) / len(xgboost), abs=1e-4), max(xgboost))


def test_accuracy_stats_makes_a_single_pass():
    assert accuracy_stats(iter(RUNS)) == accuracy_stats(RUNS)


@pytest.fixture(scope="module")
def index():
    return build_index(DOCS)


def test_build_index(index):
    assert index["tabular"] == {"d1", "d3"}
    assert all(type(ids) is frozenset for ids in index.values())


@pytest.mark.parametrize(
    ("must", "must_not", "expected"),
    [
        (["deep", "learning"], [], {"d2", "d5"}),
        (["very"], ["images"], {"d1"}),
        (["unicorn"], [], set()),
        ([], ["deep"], {"d1", "d3", "d4"}),
        ([], [], set(DOCS)),
    ],
)
def test_search(index, must, must_not, expected):
    assert search(index, set(DOCS), must, must_not) == expected


def test_search_leaves_all_ids_untouched(index):
    all_ids = set(DOCS)
    search(index, all_ids, ["tabular"], ["boosting"])
    assert all_ids == set(DOCS), "search() mutated the caller's set"


def test_schema_diff():
    old = {"lr": 0.001, "batch": 64, "dropout": 0.1}
    new = {"lr": 0.001, "batch": 128, "weight_decay": 0.0}
    assert schema_diff(old, new) == (["weight_decay"], ["dropout"], ["batch"])


def test_schema_diff_handles_unhashable_values():
    assert schema_diff({"tags": ["a"]}, {"tags": ["a", "b"]}) == ([], [], ["tags"])


@pytest.fixture
def row():
    return NormalizedDict({"batch_size": 64, "lr": 0.01})


@pytest.mark.parametrize("key", ["batch_size", "Batch Size", "BATCH-SIZE"])
def test_normalized_dict_finds_variants(row, key):
    assert row[key] == 64
    assert row.get(key) == 64
    assert key in row


def test_normalized_dict_missing_key(row):
    with pytest.raises(KeyError):
        row["dropout"]
    assert row.get("dropout", "default") == "default"
    assert "dropout" not in row


def make_key() -> RunKey:
    return RunKey(model="xgboost", dataset="titanic", lr=0.001, batch_size=64)


def test_run_key_is_hashable_by_value():
    assert len({make_key(), make_key()}) == 1


def test_run_key_is_immutable():
    with pytest.raises(FrozenInstanceError):
        make_key().lr = 0.5


def test_run_key_has_no_instance_dict():
    assert not hasattr(make_key(), "__dict__"), "use slots"
