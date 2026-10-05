"""
common.py — shared helpers and synthetic data for the four labs.

No third-party dependencies. Standard library only, Python 3.14.

Put this file next to the lab files and just `from common import ...`.
"""

import random
import time
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Synthetic dataset: results of a hyper-parameter sweep.
# Deliberately stored as plain tuples so that Lab 1 has something to unpack
# and Lab 2 has something to turn into a proper record type.
# ---------------------------------------------------------------------------

MODELS = ("logreg", "random_forest", "xgboost", "mlp", "svm")
DATASETS = ("titanic", "mnist", "adult", "cifar10")
SERVICES = ("ingest", "train", "serve")

# Field order of a run tuple. Use these constants instead of magic numbers.
RUN_ID, MODEL, DATASET, LR, BATCH, ACCURACY, DURATION = range(7)

RUN_FIELDS = ("run_id", "model", "dataset", "lr", "batch_size", "accuracy", "duration_s")


def make_runs(n: int = 2_000, seed: int = 0) -> list[tuple]:
    """Return `n` experiment runs as tuples:

        (run_id, model, dataset, lr, batch_size, accuracy, duration_s)
    """
    rng = random.Random(seed)
    runs = []
    for i in range(n):
        model = rng.choice(MODELS)
        dataset = rng.choice(DATASETS)
        lr = rng.choice((1e-4, 3e-4, 1e-3, 3e-3, 1e-2))
        batch = rng.choice((16, 32, 64, 128))
        # Give each model a slightly different quality so grouping is meaningful.
        bias = {"logreg": -0.06, "random_forest": 0.0, "xgboost": 0.04,
                "mlp": 0.01, "svm": -0.02}[model]
        acc = min(0.999, max(0.05, rng.gauss(0.80 + bias, 0.06)))
        duration = round(abs(rng.gauss(120, 60)) + 5.0, 1)
        runs.append((f"run-{i:05d}", model, dataset, lr, batch,
                     round(acc, 4), duration))
    return runs


def make_raw_lines(n: int = 20) -> list[str]:
    """A few raw CSV-ish lines, as they would come out of a log file."""
    return [",".join(str(f) for f in run) for run in make_runs(n, seed=7)]


# ---------------------------------------------------------------------------
# A "big" line stream for Lab 3. Never materialised as a list.
# ~3% of the lines are junk, so parsers must be defensive.
# ---------------------------------------------------------------------------

def iter_log_lines(n: int = 200_000, seed: int = 1):
    """Yield `n` log lines of the form 'timestamp,LEVEL,service,latency_ms'.

    About 3% of the lines are comments or malformed and must be skipped.
    """
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        if r < 0.02:
            yield f"# rotated at {1_700_000_000 + i}"
        elif r < 0.03:
            yield f"{1_700_000_000 + i},INFO,serve,"       # missing value
        else:
            level = rng.choices(("INFO", "WARN", "ERROR"), (0.80, 0.15, 0.05))[0]
            service = rng.choice(SERVICES)
            base = {"ingest": 20.0, "train": 180.0, "serve": 8.0}[service]
            latency = round(abs(rng.gauss(base, base / 3)), 2)
            yield f"{1_700_000_000 + i},{level},{service},{latency}"


# Small document collection for the inverted-index exercise (Lab 2).
DOCS: dict[str, str] = {
    "d1": "gradient boosting handles tabular data very well",
    "d2": "deep learning handles images and text very well",
    "d3": "logistic regression is a strong tabular baseline",
    "d4": "boosting and bagging are both ensemble methods",
    "d5": "transformers are deep learning models for text",
}


# ---------------------------------------------------------------------------
# Tiny test harness. Nothing clever — it just tells you what still fails.
# ---------------------------------------------------------------------------

def check(name: str, fn) -> bool:
    """Run `fn` and report. Returns True if it passed."""
    try:
        fn()
    except NotImplementedError:
        print(f"  [ ] {name:<38} todo")
        return False
    except AssertionError as exc:
        print(f"  [x] {name:<38} FAILED: {exc}")
        return False
    except Exception as exc:                      # noqa: BLE001 - teaching tool
        print(f"  [x] {name:<38} {type(exc).__name__}: {exc}")
        return False
    print(f"  [v] {name:<38} ok")
    return True


def run_checks(title: str, checks: list[tuple[str, object]]) -> None:
    print(f"\n=== {title} ===")
    passed = sum(check(name, fn) for name, fn in checks)
    print(f"  --- {passed}/{len(checks)} passing ---\n")


@contextmanager
def timed(label: str):
    """Wall-clock timer. Use perf_counter, never time.time(), for durations."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"  {label:<34} {dt:7.3f} s")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol
