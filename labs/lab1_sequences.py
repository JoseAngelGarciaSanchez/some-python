"""
LAB 1 — Sequences, references, unpacking          (~20 min)
Fluent Python, 2nd ed.: chapters 2 and 6.

Replace every `raise NotImplementedError` with your implementation, then run:

    python lab1_sequences.py

Tasks 1-4 are the core. Task 5 is a bonus. Task 6 is pen-and-paper.
"""

from common import (
    ACCURACY,
    DURATION,
    MODEL,
    RUN_ID,
    approx,
    make_raw_lines,
    make_runs,
    run_checks,
)

# ---------------------------------------------------------------------------
# TASK 1 — Aliasing
#
# A colleague wrote:
#
#     grid = [[0] * cols] * rows
#
# and their confusion matrix updates all rows at once. Write a version that
# does not have that bug. One line is enough.
# ---------------------------------------------------------------------------


def build_grid(rows: int, cols: int) -> list[list[int]]:
    """Return a rows x cols grid of zeros whose rows are independent objects."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 2 — Unpacking with a star target
#
# Raw line:  'run-00003,xgboost,titanic,0.001,64,0.8312,131.4'
#
# Parse it into a tuple with the right types:
#   (str, str, str, float, int, float, float)
#
# Constraint: use *starred unpacking* to split the line into
# "identity fields" and "numeric fields" instead of indexing seven times.
# ---------------------------------------------------------------------------


def to_record(line: str) -> tuple:
    """Parse one raw CSV line into a properly typed tuple."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 3 — Sorting with a key
#
# Return the k runs with the highest accuracy, best first.
# Ties are broken by the *shorter* duration.
#
# Constraint: no manual loops over a sorted copy, and do not sort twice.
# Hint: operator.itemgetter, and think about what a tuple key does.
# ---------------------------------------------------------------------------


def top_k(runs: list[tuple], k: int) -> list[tuple]:
    """Return the k best runs: highest accuracy first, shortest duration wins ties."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 4 — A bounded buffer
#
# Compute the trailing moving average of `values` over `window` points.
# Output has the same length as the input; for the first window-1 positions,
# average over whatever is available so far.
#
# Constraint: use collections.deque with maxlen. Do NOT slice the list
# (values[i-window:i]) — we want O(1) work per point, not O(window).
# ---------------------------------------------------------------------------


def moving_average(values: list[float], window: int) -> list[float]:
    """Trailing moving average, one output value per input value."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 5 (BONUS) — Discretisation with bisect
#
# Turn continuous accuracies into ordinal bins, the way you would before
# a chi-square test or a stratified split.
#
#   edges  = [0.70, 0.80, 0.90]
#   labels = ['poor', 'fair', 'good', 'excellent']
#
#   0.65 -> 'poor'      (below the first edge)
#   0.70 -> 'fair'      (an exact edge goes to the UPPER bin)
#   0.85 -> 'good'
#   0.99 -> 'excellent'
#
# Constraint: use the bisect module. len(labels) == len(edges) + 1.
# ---------------------------------------------------------------------------


def bucketize(scores: list[float], edges: list[float], labels: list[str]) -> list[str]:
    """Map each score to its bin label."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 6 — Predict the output (no code to write; write your answer down)
#
#   a) t = (1, 2, [30, 40])
#      t[2] += [50, 60]
#      What happens? Is it an error? What does t look like afterwards?
#      Both answers are surprising. Explain using bytecode reasoning.
#      (Try: import dis; dis.dis('t[2] += [50, 60]'))
#
#   b) def add_run(run, log=[]):
#          log.append(run)
#          return log
#      Call add_run('a') then add_run('b'). What is returned the second time?
#      Why? What is the fix?
#
#   c) import copy
#      a = [[1, 2], [3, 4]]
#      b = list(a); c = copy.deepcopy(a)
#      a[0].append(99)
#      What are b and c now?
#
#   d) l = [1, 2, 3, 4, 5]
#      l[1:3] = [9]
#      print(l)   # ?
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _t1():
    g = build_grid(3, 4)
    assert len(g) == 3 and all(len(row) == 4 for row in g)
    g[0][0] = 1
    assert [row[0] for row in g] == [1, 0, 0], f"rows are aliased: {g}"


def _t2():
    line = "run-00003,xgboost,titanic,0.001,64,0.8312,131.4"
    rec = to_record(line)
    assert rec == ("run-00003", "xgboost", "titanic", 0.001, 64, 0.8312, 131.4), rec
    assert isinstance(rec[4], int), "batch_size must be an int"
    assert isinstance(rec[ACCURACY], float)
    # It must work on the whole batch too.
    assert all(len(to_record(l)) == 7 for l in make_raw_lines(20))


def _t3():
    runs = make_runs(500, seed=3)
    best = top_k(runs, 5)
    assert len(best) == 5
    accs = [r[ACCURACY] for r in best]
    assert accs == sorted(accs, reverse=True), "not sorted by accuracy"
    assert accs[0] == max(r[ACCURACY] for r in runs)
    tie = [
        ("a", "m", "d", 0.1, 8, 0.90, 300.0),
        ("b", "m", "d", 0.1, 8, 0.90, 100.0),
        ("c", "m", "d", 0.1, 8, 0.95, 200.0),
    ]
    assert [r[RUN_ID] for r in top_k(tie, 3)] == ["c", "b", "a"], "tie-break wrong"


def _t4():
    out = moving_average([1, 2, 3, 4, 5], 3)
    assert len(out) == 5
    assert approx(out[0], 1.0) and approx(out[1], 1.5) and approx(out[2], 2.0)
    assert approx(out[4], 4.0), out
    flat = moving_average([7.0] * 10, 4)
    assert all(approx(v, 7.0) for v in flat)


def _t5():
    edges = [0.70, 0.80, 0.90]
    labels = ["poor", "fair", "good", "excellent"]
    got = bucketize([0.65, 0.70, 0.799, 0.80, 0.85, 0.90, 0.99], edges, labels)
    assert got == ["poor", "fair", "fair", "good", "good", "excellent", "excellent"], (
        got
    )


if __name__ == "__main__":
    run_checks(
        "Lab 1 — sequences",
        [
            ("1  build_grid", _t1),
            ("2  to_record", _t2),
            ("3  top_k", _t3),
            ("4  moving_average", _t4),
            ("5  bucketize (bonus)", _t5),
        ],
    )
