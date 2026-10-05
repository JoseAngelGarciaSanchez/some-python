"""
LAB 2 — Dicts, sets, hashing, record types        (~20 min)
Fluent Python, 2nd ed.: chapters 3 and 5.

    python lab2_dicts_sets.py

Tasks 1-4 are core, 5-6 are bonus.
"""

from common import (ACCURACY, DATASET, DOCS, MODEL, RUN_ID, approx, make_runs,
                    run_checks)

# ---------------------------------------------------------------------------
# TASK 1 — Grouping
#
# Group runs by model name. Return a plain dict mapping model -> list of runs,
# preserving the order in which runs appear in the input.
#
# Do it with collections.defaultdict. Then, in a comment, say why
# `d.setdefault(k, []).append(v)` is a reasonable alternative and what the
# one behavioural difference is (hint: what does d[missing_key] do afterwards?).
# ---------------------------------------------------------------------------

def group_by_model(runs: list[tuple]) -> dict[str, list[tuple]]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 2 — Aggregation
#
# For each model return (n_runs, mean_accuracy, best_accuracy).
# Round the mean to 4 decimals. One pass over the data is enough.
# ---------------------------------------------------------------------------

def accuracy_stats(runs: list[tuple]) -> dict[str, tuple[int, float, float]]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 3 — Inverted index + boolean retrieval
#
# 3a. Build word -> frozenset of document ids from `docs` (id -> text).
#     Lowercase, split on whitespace.
#
# 3b. Answer a query: return the set of doc ids that contain EVERY word in
#     `must` and NONE of the words in `must_not`. An unknown word matches
#     nothing. `must` may be empty, in which case start from all documents.
#
#     Constraint: use set algebra (&, |, -), not loops with `if word in text`.
# ---------------------------------------------------------------------------

def build_index(docs: dict[str, str]) -> dict[str, frozenset[str]]:
    raise NotImplementedError


def search(index: dict[str, frozenset[str]], all_ids: set[str],
           must: list[str], must_not: list[str] = ()) -> set[str]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 4 — Schema drift with dict views
#
# Two versions of a feature row arrive from an upstream pipeline. Report:
#   added   : keys only in `new`
#   removed : keys only in `old`
#   changed : keys in both whose value differs
#
# Constraint: `.keys()` and `.items()` are set-like. Use that. No explicit
# loops for `added` and `removed`. Return three sorted lists.
# ---------------------------------------------------------------------------

def schema_diff(old: dict, new: dict) -> tuple[list, list, list]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 5 (BONUS) — __missing__
#
# Upstream sends column names inconsistently: 'Batch Size', 'batch_size',
# 'BATCH-SIZE'. Write a dict subclass whose lookups normalise the key
# (lowercase, and '-'/' ' -> '_') before giving up.
#
#   d = NormalizedDict({'batch_size': 64})
#   d['Batch Size']   -> 64
#   d['BATCH-SIZE']   -> 64
#   d['nope']         -> KeyError
#   d.get('Batch Size') -> 64        <- this one is a trap, see the note
#
# Note: dict.get and dict.__contains__ do NOT go through __missing__.
# You must override them too.
# ---------------------------------------------------------------------------

class NormalizedDict(dict):
    @staticmethod
    def _norm(key):
        raise NotImplementedError

    def __missing__(self, key):
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 6 (BONUS) — Hashability
#
# Define a record type for a run that is:
#   * usable as a dict key / set member,
#   * immutable,
#   * memory-efficient (millions of these will exist).
#
# Use a frozen dataclass with slots=True. Then explain in a comment why the
# following class is a *bug factory*:
#
#     class Key:
#         def __init__(self, tags): self.tags = tags        # a list
#         def __hash__(self): return hash(tuple(self.tags))
#         def __eq__(self, o): return self.tags == o.tags
# ---------------------------------------------------------------------------

# from dataclasses import dataclass
# @dataclass(frozen=True, slots=True)
# class RunKey:
#     ...

RunKey = None  # replace me


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
RUNS = make_runs(400, seed=11)


def _t1():
    g = group_by_model(RUNS)
    assert isinstance(g, dict)
    assert sum(len(v) for v in g.values()) == len(RUNS)
    assert set(g) == {r[MODEL] for r in RUNS}
    first = next(r for r in RUNS if r[MODEL] == "xgboost")
    assert g["xgboost"][0] == first, "input order not preserved"


def _t2():
    stats = accuracy_stats(RUNS)
    assert set(stats) == {r[MODEL] for r in RUNS}
    n, mean, best = stats["xgboost"]
    xs = [r[ACCURACY] for r in RUNS if r[MODEL] == "xgboost"]
    assert n == len(xs)
    assert approx(mean, round(sum(xs) / len(xs), 4), 1e-4)
    assert approx(best, max(xs))


def _t3():
    idx = build_index(DOCS)
    assert idx["tabular"] == frozenset({"d1", "d3"}), idx.get("tabular")
    assert isinstance(idx["deep"], frozenset)
    ids = set(DOCS)
    assert search(idx, ids, ["deep", "learning"]) == {"d2", "d5"}
    assert search(idx, ids, ["very"], ["images"]) == {"d1"}
    assert search(idx, ids, ["unicorn"]) == set()
    assert search(idx, ids, [], ["deep"]) == {"d1", "d3", "d4"}


def _t4():
    old = {"lr": 0.001, "batch": 64, "dropout": 0.1}
    new = {"lr": 0.001, "batch": 128, "weight_decay": 0.0}
    added, removed, changed = schema_diff(old, new)
    assert added == ["weight_decay"], added
    assert removed == ["dropout"], removed
    assert changed == ["batch"], changed


def _t5():
    d = NormalizedDict({"batch_size": 64, "lr": 0.01})
    assert d["Batch Size"] == 64
    assert d["BATCH-SIZE"] == 64
    assert d["lr"] == 0.01
    assert d.get("Batch Size") == 64
    assert d.get("nope", "dflt") == "dflt"
    assert "Batch Size" in d and "nope" not in d
    try:
        d["nope"]
    except KeyError:
        pass
    else:
        raise AssertionError("missing key should raise KeyError")


def _t6():
    assert RunKey is not None, "define RunKey"
    a = RunKey(model="xgboost", dataset="titanic", lr=0.001, batch_size=64)
    b = RunKey(model="xgboost", dataset="titanic", lr=0.001, batch_size=64)
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1
    try:
        a.lr = 0.5
    except Exception:
        pass
    else:
        raise AssertionError("RunKey must be immutable")
    assert not hasattr(a, "__dict__"), "use slots=True"


if __name__ == "__main__":
    run_checks("Lab 2 — dicts & sets", [
        ("1  group_by_model", _t1),
        ("2  accuracy_stats", _t2),
        ("3  build_index / search", _t3),
        ("4  schema_diff", _t4),
        ("5  NormalizedDict (bonus)", _t5),
        ("6  RunKey (bonus)", _t6),
    ])
