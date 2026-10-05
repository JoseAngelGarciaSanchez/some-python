"""Synthetic datasets and small helpers shared by the labs.

Standard library only. Every generator is seeded, so results are reproducible.
"""

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

MODELS = ("logreg", "random_forest", "xgboost", "mlp", "svm")
DATASETS = ("titanic", "mnist", "adult", "cifar10")
SERVICES = ("ingest", "train", "serve")

# Each model gets a slightly different mean accuracy, so grouping is meaningful.
_MODEL_BIAS = {"logreg": -0.06, "random_forest": 0.0, "xgboost": 0.04, "mlp": 0.01, "svm": -0.02}
_SERVICE_LATENCY_MS = {"ingest": 20.0, "train": 180.0, "serve": 8.0}


class Run(NamedTuple):
    """One run of a hyper-parameter sweep."""

    run_id: str
    model: str
    dataset: str
    lr: float
    batch_size: int
    accuracy: float
    duration_s: float


class LogRecord(NamedTuple):
    """One parsed line of a service log."""

    timestamp: int
    level: str
    service: str
    latency_ms: float


def make_runs(n: int = 2_000, seed: int = 0) -> list[Run]:
    """Return ``n`` reproducible sweep results."""
    rng = random.Random(seed)
    runs = []
    for i in range(n):
        model = rng.choice(MODELS)
        dataset = rng.choice(DATASETS)
        lr = rng.choice((1e-4, 3e-4, 1e-3, 3e-3, 1e-2))
        batch_size = rng.choice((16, 32, 64, 128))
        accuracy = min(0.999, max(0.05, rng.gauss(0.80 + _MODEL_BIAS[model], 0.06)))
        duration_s = abs(rng.gauss(120, 60)) + 5.0
        runs.append(Run(f"run-{i:05d}", model, dataset, lr, batch_size, round(accuracy, 4), round(duration_s, 1)))
    return runs


def make_raw_lines(n: int = 20) -> list[str]:
    """Runs serialised as raw CSV lines, the way they come out of a log file."""
    return [",".join(map(str, run)) for run in make_runs(n, seed=7)]


def iter_log_lines(n: int = 200_000, seed: int = 1) -> Iterator[str]:
    """Lazily yield ``n`` lines of the form ``timestamp,LEVEL,service,latency_ms``.

    About 3% of the lines are comments or malformed, so parsers must be defensive.
    """
    rng = random.Random(seed)
    for i in range(n):
        timestamp = 1_700_000_000 + i
        r = rng.random()
        if r < 0.02:
            yield f"# rotated at {timestamp}"
        elif r < 0.03:
            yield f"{timestamp},INFO,serve,"  # missing latency
        else:
            level = rng.choices(("INFO", "WARN", "ERROR"), (0.80, 0.15, 0.05))[0]
            service = rng.choice(SERVICES)
            base = _SERVICE_LATENCY_MS[service]
            yield f"{timestamp},{level},{service},{abs(rng.gauss(base, base / 3)):.2f}"


# A tiny corpus for the inverted-index task in Lab 2.
DOCS = {
    "d1": "gradient boosting handles tabular data very well",
    "d2": "deep learning handles images and text very well",
    "d3": "logistic regression is a strong tabular baseline",
    "d4": "boosting and bagging are both ensemble methods",
    "d5": "transformers are deep learning models for text",
}


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Print the wall-clock time of the ``with`` block, if it completes.

    Uses ``time.perf_counter``: ``time.time`` is the wrong clock for durations.
    """
    start = time.perf_counter()
    yield
    print(f"  {label:<34} {time.perf_counter() - start:7.3f} s")
