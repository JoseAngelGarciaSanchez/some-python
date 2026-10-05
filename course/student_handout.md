# Python for Data Science — Handout

**Python 3.14.** Keep this open during the labs. Companion book: Luciano
Ramalho, *Fluent Python*, 2nd ed. — chapter pointers at the end of each
section. Section 5 has no chapter pointer: it covers changes that postdate
the book.

---

## 0. How Python runs your code

```
your .py ──compile──▶ bytecode ──▶ the eval loop (C) ──▶ C code doing the work
```

No machine-code compilation. A C loop executes bytecode instructions one at
a time. `import dis; dis.dis("total = total + 1")` shows them.

A Python-level loop iteration costs tens of nanoseconds; the same work in C
costs under one. Hence:

> **Make the interpreter do fewer, bigger things.** One vectorised call is
> one trip into C. A hand-written loop is a million trips through the eval
> loop.

Everything is a heap object with a refcount: `sys.getsizeof(1)` is ~28 bytes.
Assignment copies a *reference*, never the object. Memory is managed by
reference counting plus a cycle collector — and making those refcounts safe
across threads without one global lock is exactly what took a decade to
solve.

**Where the time goes**

| Program is... | Bottleneck | Fix |
|---|---|---|
| waiting on network / disk / DB | I/O | overlap it: threads, asyncio |
| running Python bytecode | interpreter | better algorithm, better container, vectorise |
| running C (NumPy, pandas) | CPU / memory | more cores, better memory layout |
| swapping / `MemoryError` | memory | stream it: generators |

**Attack in this order.** Algorithm and data structure → let C do the work →
*then* parallelise. A dict instead of a linear scan wins ~1000×. Four cores
wins at most 4×, and never that in practice.

**Measure, don't guess**

```python
time.perf_counter()                         # never time.time() for durations
```
```bash
python -m cProfile -s cumtime script.py
python -m timeit -s "setup" "statement"
```
Warm up first. Report interpreter version, core count and GIL status with any
timing — a number without them is not a result.

---

## 1. References, sequences

**A name is a label on an object, never a box.** Assignment binds; it never
copies.

```python
a = [1, 2]; b = a; b.append(3)      # a is now [1, 2, 3]
x == y      # same value  (__eq__)
x is y      # same object (identity) — only use for None / sentinels
```

**Traps**

```python
grid = [[0] * 3] * 3        # all three rows are THE SAME list
def f(x, log=[]):           # default evaluated once, at def time
list(xs), xs[:], xs.copy()  # SHALLOW — inner objects still shared
t = (1, 2, [3]); t[2] += [4]   # raises TypeError AND mutates the list
```

**Sequence types**

| | Mutable | Immutable |
|---|---|---|
| Container (holds references) | `list`, `deque` | `tuple` |
| Flat (holds packed bytes) | `bytearray`, `array.array` | `str`, `bytes` |

**Unpacking**

```python
head, *rest = seq
first, *middle, last = seq
for name, (acc, dur) in items: ...
a, b = b, a
f(*args, **kwargs)
```

**Slicing** — `s[start:stop:step]`, `stop` exclusive. Slice assignment
splices and can change length: `l[1:3] = [9]`. `del l[1:3]` works too.

**Sorting** — `list.sort()` in place, returns `None`; `sorted()` returns a
new list. Both stable, both take `key=`.

```python
sorted(runs, key=lambda r: (r.acc, -r.dur), reverse=True)   # mixed direction
```

**When a list is the wrong answer**

| Need | Use |
|---|---|
| append/pop at both ends, sliding window | `collections.deque(maxlen=n)` |
| binary search / binning a sorted list | `bisect.bisect_right` |
| top-k without a full sort | `heapq.nlargest(k, it, key=...)` |
| a million homogeneous numbers | `array.array`, NumPy |
| membership tests in a loop | `set` |

> *Fluent Python* ch. 2, ch. 6

---

## 2. Dicts, sets, records

**Hashability contract.** If `a == b` then `hash(a) == hash(b)`, and the
hash must never change. Therefore anything hashable must be immutable in
every field used by `__eq__`. Mutating an object after putting it in a set
makes it unreachable *while still being in the set*.

**Dicts** — average O(1) lookup; keys must be hashable; **insertion order is
guaranteed** since 3.7; never add/remove keys while iterating.

**Missing keys — four tools**

```python
d.get(k, default)                 # read, never inserts
d.setdefault(k, []).append(v)     # builds the default on EVERY call
defaultdict(list)                 # factory only on a miss; INSERTS on read
class D(dict):
    def __missing__(self, k): ... # only called by d[k], NOT by .get / in
```

**Variants** — `Counter` (`.most_common`, supports `+ - & |`), `ChainMap`
(layered config), `MappingProxyType` (read-only view).

**Views are sets**

```python
new.keys() - old.keys()     # added
old.keys() - new.keys()     # removed
old.keys() & new.keys()     # common
```

**Sets** — `&` `|` `-` `^`. `{}` is an empty **dict**; use `set()`.
`frozenset` is hashable.

**Record types**

| | Mutable | Hashable | Memory |
|---|---|---|---|
| `tuple` | no | yes | lowest |
| `NamedTuple` | no | yes | low |
| `@dataclass` | yes | no | high |
| `@dataclass(frozen=True, slots=True)` | no | yes | low |

Default choice: `frozen=True, slots=True`. Use `NamedTuple` when the code
already indexes or unpacks the record. Type hints are **not enforced at
runtime**.

> *Fluent Python* ch. 3, ch. 5

---

## 3. Functions, iterators, generators

**Decorator skeleton**

```python
import functools, time

def timed(func):
    @functools.wraps(func)              # keeps __name__ / __doc__
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()        # never time.time()
        try:
            return func(*args, **kwargs)
        finally:                        # measure failures too
            print(f"{func.__name__}: {time.perf_counter() - t0:.4f}s")
    return wrapper
```

`@functools.cache` / `@lru_cache(maxsize=N)` — arguments must be
**hashable**; `cache` never evicts; never memoise an impure function.

**Closure trap**

```python
fns = [lambda: i for i in range(3)]     # all return 2
fns = [lambda i=i: i for i in range(3)] # fixed
```

**Iterable vs iterator**

- iterable → produces a fresh iterator each time (`list`, `dict`, `str`)
- iterator → single-use; `iter(x) is x` is `True`

```python
n     = sum(1 for _ in records)
total = sum(r.latency for r in records)   # 0! stream already exhausted
```
No exception, just a wrong number. Restructure into one pass.

**Generators and pipelines**

```python
sum(x * x for x in range(10**7))     # constant memory
```

```python
lines   = read_lines(path)                          # lazy
records = parse(lines)                              # lazy
warns   = (r for r in records if r.level == "WARN") # lazy
stats   = aggregate(warns)                          # only eager step
```
Peak memory = one record, whatever the file size.

**itertools**

`islice` · `chain` · `batched` (3.12+) · `tee` (buffers!) · `accumulate` ·
`product` · `combinations` ·
`groupby` — groups only **adjacent** equal keys, so sort first or use a
`defaultdict` counter instead.

> *Fluent Python* ch. 7, ch. 9, ch. 17

---

## 4. Concurrency

**Classify the workload first**

- **I/O-bound** — waiting on network/disk/DB. CPU idle.
- **CPU-bound** — computing. CPU saturated.

**The GIL** — one mutex; only one thread runs Python **bytecode** at a time.

It does **not**: make your data thread-safe · stay held during blocking I/O
· stay held inside C extensions that release it (NumPy, pandas, sklearn,
`zlib`, `hashlib`).

> Threads are useless for **pure-Python CPU** work, and excellent for
> everything else.

Check at runtime, never assume: `sys._is_gil_enabled()`. See section 5.

**One API for both**

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(fetch, urls))       # INPUT order

futures = {ex.submit(parse, s): s for s in shards}
for fut in as_completed(futures):
    fut.result()        # re-raises worker exceptions — futures you never
                        # inspect swallow every error
```

Pool size: CPU-bound → `len(os.sched_getaffinity(0))` (**not**
`os.cpu_count()`, which lies inside Docker/SLURM). I/O-bound → many more
than cores.

**Process pitfalls**

- Not picklable: lambdas, closures, nested functions, local classes, file
  handles, sockets, locks. Use module-level functions or
  `functools.partial`.
- Shipping big arrays can cost more than the compute. Use
  `multiprocessing.shared_memory`, `np.memmap`, or a worker `initializer`.
- Many tiny tasks → pass `chunksize=`.
- Start methods: `fork` / `spawn` / `forkserver`. Since 3.14 `fork` is no
  longer the default anywhere; Linux defaults to `forkserver`.
- **`if __name__ == "__main__":` is mandatory.** Process pools do not work
  in notebooks — run scripts from a terminal.

**Shared state**

```python
self.value += 1        # NOT atomic: load, add, store
with self._lock:       # Lock as a context manager
    self.value += 1
```
Best fix: don't share. Per-worker accumulators, combined at the end.
A race that passes your tests is still a race.

**asyncio** — one thread, one loop, cooperative. Best for *thousands* of
concurrent I/O operations. One blocking call (`time.sleep`, `requests.get`)
freezes the whole loop; use async-native clients or `asyncio.to_thread`.

**Data science notes** — NumPy's BLAS already multithreads; nesting your own
pool oversubscribes the CPU (set `OMP_NUM_THREADS=1` in workers). Prefer
`joblib` for array work. Beyond one machine: Dask, Ray, Spark.

**Decision tree**

```
0. Profiled? Algorithm and container right?   <- do this FIRST
1. Slow because it is WAITING?
   ├── yes ──> dozens of ops    -> ThreadPoolExecutor
   │           thousands of ops -> asyncio
   └── no, it is COMPUTING
       ├── hot loop inside NumPy / pandas / sklearn / C?
       │   ├── yes -> threads already work (GIL released); check the
       │   │          library isn't parallelising it already
       │   └── no, pure Python:
       │       ├── standard build -> ProcessPoolExecutor
       │       │                     or InterpreterPoolExecutor (3.14)
       │       └── 3.14t          -> threads finally work
       │                             audit shared state first
       └── data too big for one machine -> Dask / Ray / Spark
```

> *Fluent Python* ch. 19, ch. 20, ch. 21

---

## 5. Free-threading (3.14) — beyond the book

*Fluent Python* 2nd ed. is from 2022 and its concurrency chapters assume the
GIL is unconditional. That assumption changed with the version you are
running.

| | |
|---|---|
| PEP 703 (2023) | plan accepted to make the GIL optional, in three phases |
| Python 3.13 | phase I — experimental build, ~40% single-thread overhead |
| **Python 3.14** | phase II (PEP 779) — **officially supported**, still optional, binary `3.14t`, ~5–10% overhead |
| phase III | free-threaded becomes the default. Years away. |

**Detect it — two different questions**

```python
sysconfig.get_config_var("Py_GIL_DISABLED")   # was this BINARY built without a GIL?
sys._is_gil_enabled()                          # is the GIL running RIGHT NOW?
```
They differ: `python3.14t -X gil=1` is a free-threaded build with the GIL
switched back on. Importing a C extension that hasn't declared support also
turns it back on silently.

**What it does and does not change**

- ✅ Pure-Python CPU work in threads now scales with cores. The thirty-year
  rule has an exception.
- ❌ Does **not** speed up NumPy/pandas — they already released the GIL.
- ⚠️ Costs 5–10% single-threaded. It is a trade.
- ⚠️ Opt-in, not the default. Your users have the standard build.
- ⚠️ **Widens existing races.** It creates no new bug class; it makes latent
  ones fire. Audit shared mutable state before trusting it.

**What is guaranteed atomic**

Built-in container operations (`list.append`, `dict[k] = v`, `set.add`) stay
internally consistent with or without the GIL. **Compound operations you
write are not**: `self.value += 1`, `if k not in d: d[k] = v`. Those are your
races, on every build.

**Third option, new in 3.14** — subinterpreters (PEP 734):
`concurrent.interpreters`, `InterpreterPoolExecutor`. One interpreter per
worker, each with its own GIL, inside one process. Parallel pure Python on
the standard build, cheaper than processes.

```
threads      shared memory, no isolation, GIL-bound (until 3.14t)
subinterps   same process, isolated, own GIL each, medium cost
processes    full isolation, highest start-up and transfer cost
```

> No chapter — see PEP 703, PEP 779, and <https://py-free-threading.github.io/>

---

## Running the labs

```bash
cd labs
python3.14 lab1_sequences.py     # each run prints pass/fail per task
python3.14 lab4_concurrency.py   # MUST be a terminal, not a notebook

# Lab 5 is meant to be run TWICE — same binary, only the GIL differs
python3.14t -X gil=1 lab5_freethreading.py
python3.14t          lab5_freethreading.py
```

Get the interpreters with `uv python install 3.14 3.14t`.
