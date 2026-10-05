"""Lab 1 — Sequences, references and unpacking.

Fluent Python, 2nd ed., chapters 2 and 6. About 20 minutes.

Implement each function until its tests pass::

    uv run pytest tests/test_lab1_sequences.py

Tasks 1-4 are the core of the lab and Task 5 is a bonus. Task 6 is pen and
paper: predict each answer first, then check it in a REPL.

Task 6 — Predict the output
---------------------------
a) ``t = (1, 2, [30, 40])`` then ``t[2] += [50, 60]``.
   Is it an error? What does ``t`` hold afterwards? Both answers are
   surprising. Explain them with ``import dis; dis.dis("t[2] += [50, 60]")``.

b) ``def add_run(run, log=[]): log.append(run); return log``.
   Call ``add_run("a")`` then ``add_run("b")``. What does the second call
   return, why, and what is the fix?

c) ``a = [[1, 2], [3, 4]]; b = list(a); c = copy.deepcopy(a)``
   then ``a[0].append(99)``. What are ``b`` and ``c`` now?

d) ``l = [1, 2, 3, 4, 5]; l[1:3] = [9]``. What is ``l``?
"""

from labs.common import Run


def build_grid(rows: int, cols: int) -> list[list[int]]:
    """Return a ``rows`` x ``cols`` grid of zeros whose rows are independent.

    Task 1 — Aliasing. A colleague wrote ``[[0] * cols] * rows`` and now every
    row of their confusion matrix updates at once. Fix it in one line.

    >>> grid = build_grid(2, 3)
    >>> grid[0][0] = 1
    >>> grid
    [[1, 0, 0], [0, 0, 0]]
    """
    raise NotImplementedError


def to_record(line: str) -> Run:
    """Parse one raw CSV line into a typed ``Run``.

    Task 2 — Unpacking. Split the line into its three identity fields and its
    four numeric fields with a single *starred* assignment, instead of
    indexing seven times. Convert each numeric field to its proper type.

    >>> to_record("run-00003,xgboost,titanic,0.001,64,0.8312,131.4")
    Run(run_id='run-00003', model='xgboost', dataset='titanic', lr=0.001, batch_size=64, accuracy=0.8312, duration_s=131.4)
    """
    raise NotImplementedError


def top_k(runs: list[Run], k: int) -> list[Run]:
    """Return the ``k`` best runs, highest accuracy first.

    Task 3 — Sorting with a key. Ties on accuracy go to the *shorter*
    duration. Sort exactly once, with a single key function.
    """
    raise NotImplementedError


def moving_average(values: list[float], window: int) -> list[float]:
    """Return the trailing moving average of ``values`` over ``window`` points.

    Task 4 — A bounded buffer. The output has one value per input value. The
    first ``window - 1`` positions average over whatever is available so far.

    Keep the window in a ``collections.deque(maxlen=window)``: never slice
    ``values``. Stretch goal: do constant work per point, whatever ``window``.

    >>> moving_average([1, 2, 3, 4, 5], window=3)
    [1.0, 1.5, 2.0, 3.0, 4.0]
    """
    raise NotImplementedError


def bucketize(scores: list[float], edges: list[float], labels: list[str]) -> list[str]:
    """Map each score to the label of the bin it falls in.

    Task 5 (bonus) — Binary search. ``labels`` has one more entry than
    ``edges``. A score equal to an edge goes to the *upper* bin. Use the
    ``bisect`` module.

    >>> bucketize([0.65, 0.70, 0.85, 0.99], [0.70, 0.80, 0.90], ["poor", "fair", "good", "excellent"])
    ['poor', 'fair', 'good', 'excellent']
    """
    raise NotImplementedError
