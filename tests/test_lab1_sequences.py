import pytest

from labs.common import Run, make_raw_lines, make_runs
from labs.lab1_sequences import bucketize, build_grid, moving_average, to_record, top_k


def test_build_grid_shape():
    grid = build_grid(3, 4)
    assert len(grid) == 3
    assert all(row == [0, 0, 0, 0] for row in grid)


def test_build_grid_rows_are_independent():
    grid = build_grid(3, 4)
    grid[0][0] = 1
    assert [row[0] for row in grid] == [1, 0, 0], "rows are aliases of one list"


def test_to_record_parses_and_converts():
    record = to_record("run-00003,xgboost,titanic,0.001,64,0.8312,131.4")
    assert record == Run("run-00003", "xgboost", "titanic", 0.001, 64, 0.8312, 131.4)
    assert isinstance(record, Run)
    assert type(record.batch_size) is int


def test_to_record_round_trips_raw_lines():
    assert [to_record(line) for line in make_raw_lines(20)] == make_runs(20, seed=7)


def test_top_k_returns_the_best_runs_in_order():
    runs = make_runs(500, seed=3)
    best = top_k(runs, 5)
    accuracies = [run.accuracy for run in best]
    assert len(best) == 5
    assert accuracies == sorted(accuracies, reverse=True)
    assert accuracies[0] == max(run.accuracy for run in runs)


def test_top_k_breaks_ties_on_shorter_duration():
    runs = [
        Run("a", "m", "d", 0.1, 8, 0.90, 300.0),
        Run("b", "m", "d", 0.1, 8, 0.90, 100.0),
        Run("c", "m", "d", 0.1, 8, 0.95, 200.0),
    ]
    assert [run.run_id for run in top_k(runs, 3)] == ["c", "b", "a"]


def test_moving_average_warms_up_then_slides():
    assert moving_average([1, 2, 3, 4, 5], 3) == pytest.approx([1.0, 1.5, 2.0, 3.0, 4.0])


def test_moving_average_of_a_constant_is_constant():
    assert moving_average([7.0] * 10, 4) == pytest.approx([7.0] * 10)


def test_moving_average_of_nothing_is_empty():
    assert moving_average([], 3) == []


@pytest.mark.parametrize(
    ("score", "label"),
    [(0.65, "poor"), (0.70, "fair"), (0.799, "fair"), (0.80, "good"), (0.90, "excellent"), (0.99, "excellent")],
)
def test_bucketize(score, label):
    edges = [0.70, 0.80, 0.90]
    labels = ["poor", "fair", "good", "excellent"]
    assert bucketize([score], edges, labels) == [label]
