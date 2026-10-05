"""Lab 2 — Dicts, sets, hashing and record types.

Fluent Python, 2nd ed., chapters 3 and 5. About 20 minutes.

Implement each function until its tests pass::

    uv run pytest tests/test_lab2_dicts_sets.py

Tasks 1-4 are the core of the lab, Tasks 5 and 6 are bonuses.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from labs.common import Run


def group_by_model(runs: Iterable[Run]) -> dict[str, list[Run]]:
    """Group runs by model name, keeping input order within each group.

    Task 1 — Grouping. Build the groups with ``collections.defaultdict``, but
    return a plain ``dict``. Then answer in a comment: why is
    ``d.setdefault(k, []).append(v)`` a reasonable alternative, and how does
    reading ``d[missing_key]`` differ between the two?
    """
    raise NotImplementedError


def accuracy_stats(runs: Iterable[Run]) -> dict[str, tuple[int, float, float]]:
    """Return ``{model: (n_runs, mean_accuracy, best_accuracy)}``.

    Task 2 — Aggregation. Make a single pass over ``runs``: it may be an
    iterator. Round the mean to 4 decimals.
    """
    raise NotImplementedError


def build_index(docs: dict[str, str]) -> dict[str, frozenset[str]]:
    """Map each word to the ids of the documents that contain it.

    Task 3a — Inverted index. Lowercase the text and split it on whitespace.

    >>> build_index({"a": "Deep learning", "b": "deep sea"})["deep"] == {"a", "b"}
    True
    """
    raise NotImplementedError


def search(
    index: dict[str, frozenset[str]],
    all_ids: set[str],
    must: Iterable[str],
    must_not: Iterable[str] = (),
) -> set[str]:
    """Return the documents that contain every ``must`` word and no ``must_not`` word.

    Task 3b — Boolean retrieval. An unknown word matches nothing. With no
    ``must`` words, start from ``all_ids``. Use set algebra (``&``, ``|``,
    ``-``), never ``word in text``.
    """
    raise NotImplementedError


def schema_diff(old: dict[str, object], new: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    """Return the sorted ``(added, removed, changed)`` keys between two rows.

    Task 4 — Schema drift. A key has *changed* when it is in both rows with
    different values. Dict views are set-like: compute ``added`` and
    ``removed`` without writing a loop.

    >>> schema_diff({"lr": 0.1, "batch": 64, "dropout": 0.1}, {"lr": 0.1, "batch": 128, "decay": 0.0})
    (['decay'], ['dropout'], ['batch'])
    """
    raise NotImplementedError


class NormalizedDict(dict):
    """A dict that forgives inconsistent column names.

    Task 5 (bonus) — ``__missing__``. Upstream sends ``"Batch Size"``,
    ``"batch_size"`` and ``"BATCH-SIZE"`` for the same column. When a key is
    missing, retry with its normalised form: lowercase, with ``-`` and spaces
    turned into ``_``. A key that is still missing raises ``KeyError``.

    Trap: ``dict.get`` and ``in`` never call ``__missing__``. Override them too.

    >>> row = NormalizedDict({"batch_size": 64})
    >>> row["Batch Size"], row.get("BATCH-SIZE"), "batch size" in row
    (64, 64, True)
    """

    @staticmethod
    def normalize(key: str) -> str:
        raise NotImplementedError

    def __missing__(self, key: str) -> object:
        raise NotImplementedError

    def get(self, key: str, default: object = None) -> object:
        raise NotImplementedError

    def __contains__(self, key: object) -> bool:
        raise NotImplementedError


@dataclass
class RunKey:
    """The identity of a run, used as a dict key or set member.

    Task 6 (bonus) — Hashability. Millions of these will exist, so the type
    must be hashable, immutable and memory-efficient. Change only the
    arguments of the ``@dataclass`` decorator.

    Then explain in a comment why this class is a bug factory::

        class Key:
            def __init__(self, tags):
                self.tags = tags  # a list

            def __hash__(self):
                return hash(tuple(self.tags))

            def __eq__(self, other):
                return self.tags == other.tags
    """

    model: str
    dataset: str
    lr: float
    batch_size: int
