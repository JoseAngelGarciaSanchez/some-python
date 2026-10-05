# Python for Data Science — Labs

Five hands-on labs on the parts of Python that decide whether data code is
correct and fast: references and sequences, dicts and sets, generators,
concurrency, and the free-threaded interpreter that ships with Python 3.14.
The companion book is Luciano Ramalho's *Fluent Python*, 2nd edition, and
[`course/student_handout.md`](course/student_handout.md) is the reference
sheet to keep open while you work.

| Lab | Topic | Time | *Fluent Python* |
|---|---|---|---|
| [1](labs/lab1_sequences.py) | Sequences, references, unpacking | 20 min | ch. 2, 6 |
| [2](labs/lab2_dicts_sets.py) | Dicts, sets, hashing, record types | 20 min | ch. 3, 5 |
| [3](labs/lab3_generators.py) | Decorators, iterators, generators | 18 min | ch. 7, 9, 17 |
| [4](labs/lab4_concurrency.py) | Threads, processes, the GIL | 22 min | ch. 19–21 |
| [5](labs/lab5_freethreading.py) | Free-threaded Python 3.14 | 10 min | — |

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.14 and pytest, and
installs the `labs` package in editable mode, so `from labs.common import Run`
works from anywhere:

```bash
uv sync
```

## Working on a lab

Each lab is a module of functions to implement. Every function's docstring is
its specification: what to build, the constraints, and an example. Replace
each `raise NotImplementedError` with your implementation, then run that lab's
tests until they pass:

```bash
uv run pytest tests/test_lab1_sequences.py -v
```

Written questions sit in the lab's module docstring or in the docstring of the
function they concern. Answer them in a comment or on paper: they are what the
debrief is built on.

Labs 4 and 5 also have experiments to run and interpret. Run them from a
terminal, never from a notebook, because process pools cannot start workers
from one:

```bash
uv run python -m labs.lab4_concurrency
```

Lab 5 compares the same free-threaded interpreter with the GIL forced on and
with it off. Run it from the repository root:

```bash
uv python install 3.14t
uv run --no-project --python 3.14t python -X gil=1 -m labs.lab5_freethreading
uv run --no-project --python 3.14t python -m labs.lab5_freethreading
```

## Layout

```
course/student_handout.md   reference sheet for the whole course
labs/                       the `labs` package
├── common.py               seeded synthetic datasets shared by the labs
└── labN_*.py               the exercises, one module per lab
tests/test_labN_*.py        their tests
```
