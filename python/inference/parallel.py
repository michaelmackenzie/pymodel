"""Reproducible random streams and parallel execution of toy work.

Seeds
-----
Every toy has its own random stream, derived from the run seed with numpy's
``SeedSequence(seed, spawn_key=key)``.  The key names the toy completely:

    (STREAM_<kind>, <point key>, toy index, <purpose>)

where the point key is the IEEE-754 bit pattern of the tested r (``float_key``) and the
purpose separates toy generation (0), the free fit (1) and the fit at each tested r
(2, float_key(r)).  The fitter's random generator (used only for perturbed restarts of failed
fits) is reseeded from the same keys before every fit.  A toy's content and all fits made on
it therefore depend only on (seed, key), never on which process ran it, in which order, or how
the toys were split into chunks: results are bit-identical for any ``--jobs`` and for any
split into jobs that are merged afterwards.

Execution
---------
``Executor.map(fn, tasks)`` runs ``fn(context, task)`` for every task and returns the results
in task order.  With ``jobs == 1`` it runs in-process on the caller's context.  With
``jobs > 1`` it uses a ``multiprocessing`` pool with the "spawn" start method: every worker
rebuilds its own likelihood from a picklable factory (backends hold ROOT / TensorFlow objects
that must not be forked or pickled).  The fitter state (bounds, frozen parameters) of the
caller is sent with every call, so workers always fit with the caller's settings.  Tasks and
results must be picklable: send ``portable(dataset)`` copies, never datasets that carry
backend-native caches.
"""

import contextlib
import io
import math
import multiprocessing
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from inference.model import Dataset

# stream identifiers (first element of every spawn key)
STREAM_GEN = 1        # fit / generate -t toys
STREAM_SB = 2         # CLs s+b toys at r
STREAM_B = 3          # CLs background-only toys (shared by every r)
STREAM_FC = 4         # Feldman-Cousins toys at r
STREAM_Q0 = 5         # significance toys
STREAM_OBS = 6        # fits of the observed data (fitter restarts only)
STREAM_IMPACT = 7     # impact fits

PURPOSE_GENERATE, PURPOSE_FREE_FIT, PURPOSE_FIT_AT = 0, 1, 2


def float_key(x: float) -> int:
    """Non-negative integer with the bit pattern of the double ``x`` (a spawn-key element)."""
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


class ToySeeds:
    """Deterministic per-key random generators derived from one run seed."""

    def __init__(self, seed: int):
        if seed < 0:
            raise ValueError(f"seeds must be non-negative, got {seed}")
        self.seed = int(seed)

    def rng(self, *key) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=tuple(int(k) for k in key)))

    def __repr__(self):
        return f"ToySeeds({self.seed})"


def as_seeds(rng) -> ToySeeds:
    """Accept a ToySeeds, an int seed, or a numpy Generator (one integer is drawn from it)."""
    if isinstance(rng, ToySeeds):
        return rng
    if isinstance(rng, (int, np.integer)):
        return ToySeeds(int(rng))
    if isinstance(rng, np.random.Generator):
        return ToySeeds(int(rng.integers(0, 2 ** 63 - 1)))
    raise TypeError(f"expected ToySeeds, int or numpy Generator, got {type(rng).__name__}")


@contextlib.contextmanager
def seeded_fitter(fitter, seeds: ToySeeds, *key):
    """Reseed the fitter's restart generator from ``key`` for the duration of a fit."""
    old = fitter.rng
    fitter.rng = seeds.rng(*key)
    try:
        yield fitter
    finally:
        fitter.rng = old


def portable(d: Dataset) -> Dataset:
    """A copy of a dataset without backend-native caches (safe to pickle)."""
    return Dataset(main=d.main, global_obs=dict(d.global_obs), label=d.label, truth=d.truth)


@dataclass
class WorkContext:
    """What a task function gets: the likelihood, a fitter and a per-process cache."""

    lik: object
    fitter: object
    cache: dict = field(default_factory=dict)


def fitter_state(fitter) -> dict:
    return {"bounds": dict(fitter.bounds), "frozen": sorted(fitter.frozen)}


def apply_fitter_state(fitter, state: dict):
    fitter.bounds = dict(state["bounds"])
    fitter.frozen = set(state["frozen"])


# ----- worker side ----------------------------------------------------------------------

_WORKER = {"ctx": None, "error": None, "state": None}


def _init_worker(factory):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _WORKER["ctx"] = factory()
    except BaseException as exc:  # reported by the first task instead of a hanging pool
        _WORKER["error"] = f"worker initialisation failed: {type(exc).__name__}: {exc}"


def _run_task(item):
    fn, state, task = item
    if _WORKER["error"]:
        raise RuntimeError(_WORKER["error"])
    ctx = _WORKER["ctx"]
    if state != _WORKER["state"]:
        apply_fitter_state(ctx.fitter, state)
        _WORKER["state"] = state
    return fn(ctx, task)


# ----- caller side ----------------------------------------------------------------------

class Executor:
    """Runs task functions serially (jobs == 1) or on a spawn-started process pool."""

    def __init__(self, lik, fitter, jobs: int = 1, factory: Optional[Callable[[], WorkContext]] = None):
        if jobs < 1:
            raise ValueError(f"--jobs must be >= 1, got {jobs}")
        if jobs > 1 and factory is None:
            raise ValueError("parallel execution needs a factory that rebuilds the likelihood in each worker")
        self.jobs = jobs
        self.ctx = WorkContext(lik=lik, fitter=fitter)
        self.factory = factory
        self._pool = None

    def chunk_size(self, n: int) -> int:
        """Toys per task: one task per unit of work serially, ~4 tasks per worker in parallel."""
        if self.jobs == 1:
            return max(1, n)
        return max(1, math.ceil(n / (4 * self.jobs)))

    def map(self, fn, tasks):
        tasks = list(tasks)
        if self.jobs == 1 or not tasks:
            return [fn(self.ctx, t) for t in tasks]
        if self._pool is None:
            mp = multiprocessing.get_context("spawn")
            self._pool = mp.Pool(self.jobs, initializer=_init_worker, initargs=(self.factory,))
        state = fitter_state(self.ctx.fitter)
        return self._pool.map(_run_task, [(fn, state, t) for t in tasks], chunksize=1)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._pool is not None and exc[0] is not None:
            self._pool.terminate()
            self._pool = None
        self.close()


def serial(lik, fitter) -> Executor:
    return Executor(lik, fitter, jobs=1)
