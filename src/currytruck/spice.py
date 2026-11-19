"""spice — parameter optimization by grid search and random search.

Season to taste: try combinations of parameters and keep the one that scores
best. The objective is called with each parameter set as keyword arguments.
"""

from __future__ import annotations

import functools
import itertools
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from currytruck.exceptions import EmptySpaceError
from currytruck.result import Result

Callback = Callable[[int, Any, float], Optional[bool]]


def _better(a: float, b: float, maximize: bool) -> bool:
    return a > b if maximize else a < b


def _call_with_kwargs(objective: Callable[..., float], params: dict[str, Any]) -> float:
    return float(objective(**params))


def _chunks(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    it = iter(items)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def _run_trials(
    objective: Callable[..., float],
    candidates: Iterable[dict[str, Any]],
    maximize: bool,
    method: str,
    keep_trials: bool,
    workers: Optional[int],
    backend: str,
    callback: Optional[Callback],
) -> Result:
    if backend not in ("thread", "process"):
        raise ValueError(f"backend must be 'thread' or 'process', got {backend!r}")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    best_params: Optional[dict[str, Any]] = None
    best_score = 0.0
    history: list[float] = []
    trials: list[tuple[dict[str, Any], float]] = []
    stopped = False
    evaluate = functools.partial(_call_with_kwargs, objective)

    pool = None
    if workers is not None and workers > 1:
        pool = (ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor)(max_workers=workers)
    try:
        # Evaluate in chunks so a callback can stop a parallel run early
        chunk_size = workers * 4 if pool is not None else 1
        for chunk in _chunks(candidates, chunk_size):
            scores = list(pool.map(evaluate, chunk)) if pool is not None else [evaluate(chunk[0])]
            for params, score in zip(chunk, scores):
                if keep_trials:
                    trials.append((params, score))
                if best_params is None or _better(score, best_score, maximize):
                    best_params, best_score = params, score
                history.append(best_score)
                if callback is not None and callback(len(history), best_params, best_score):
                    stopped = True
                    break
            if stopped:
                break
    finally:
        if pool is not None:
            pool.shutdown()

    if best_params is None:
        raise EmptySpaceError("Search space produced no candidates")

    return Result(
        x=best_params,
        score=best_score,
        maximize=maximize,
        history=history,
        n_evals=len(history),
        n_iter=len(history),
        converged=not stopped,
        message="Stopped by callback" if stopped else f"Evaluated {len(history)} candidates",
        method=method,
        extra={"trials": trials} if keep_trials else {},
    )


def grid_search(
    objective: Callable[..., float],
    space: Mapping[str, Any],
    *,
    maximize: bool = False,
    keep_trials: bool = True,
    workers: Optional[int] = None,
    backend: str = "thread",
    callback: Optional[Callback] = None,
) -> Result:
    """Exhaustively evaluate every combination of parameter values.

    Parameters
    ----------
    objective:
        Called as ``objective(**params)``; must return a number.
    space:
        ``{name: iterable_of_values}``. The Cartesian product is evaluated.
    maximize:
        If ``True``, higher scores are better. Default minimizes.
    keep_trials:
        Store every ``(params, score)`` pair in ``result.extra["trials"]``.
    workers:
        Evaluate candidates in parallel with this many workers. Results are
        identical to a serial run.
    backend:
        ``"thread"`` (default — best when the objective releases the GIL, e.g.
        numpy / ML training / I/O) or ``"process"`` (pure Python CPU-bound
        objectives; the objective must be picklable, i.e. a module-level
        function, not a lambda).
    callback:
        ``callback(iteration, best_params, best_score)`` after each
        evaluation; return ``True`` to stop early.

    Examples
    --------
    >>> r = grid_search(lambda x, y: (x - 2) ** 2 + y, {"x": [1, 2, 3], "y": [0, 1]})
    >>> r.x, r.score
    ({'x': 2, 'y': 0}, 0.0)
    """
    if not space:
        raise EmptySpaceError("Search space is empty")
    names = list(space)
    value_lists = [list(space[n]) for n in names]
    for name, values in zip(names, value_lists):
        if not values:
            raise EmptySpaceError(f"Parameter {name!r} has no values")

    candidates = (dict(zip(names, combo)) for combo in itertools.product(*value_lists))
    return _run_trials(
        objective, candidates, maximize, "grid_search", keep_trials, workers, backend, callback
    )


def _sampler(name: str, spec: Any) -> Callable[[random.Random], Any]:
    # (lo, hi) tuple of numbers -> uniform; ints on both ends -> randint
    if isinstance(spec, tuple) and len(spec) == 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec
    ):
        lo, hi = spec
        if lo > hi:
            raise ValueError(f"Parameter {name!r}: lower bound {lo} exceeds upper bound {hi}")
        if isinstance(lo, int) and isinstance(hi, int):
            return lambda rng: rng.randint(lo, hi)
        return lambda rng: rng.uniform(lo, hi)
    if callable(spec):
        return spec
    values = list(spec)
    if not values:
        raise EmptySpaceError(f"Parameter {name!r} has no values")
    return lambda rng: rng.choice(values)


def random_search(
    objective: Callable[..., float],
    space: Mapping[str, Any],
    *,
    n_iter: int = 100,
    maximize: bool = False,
    seed: Optional[int] = None,
    keep_trials: bool = True,
    workers: Optional[int] = None,
    backend: str = "thread",
    callback: Optional[Callback] = None,
) -> Result:
    """Evaluate ``n_iter`` randomly sampled parameter sets.

    Parameters
    ----------
    objective:
        Called as ``objective(**params)``; must return a number.
    space:
        ``{name: spec}`` where each spec is one of:

        - ``(lo, hi)`` tuple of ints — uniform integer in ``[lo, hi]``
        - ``(lo, hi)`` tuple with a float — uniform float in ``[lo, hi]``
        - a list / other iterable — uniform choice among its values
        - a callable ``f(rng) -> value`` — custom sampler given a
          ``random.Random``
    n_iter:
        Number of samples to evaluate.
    maximize:
        If ``True``, higher scores are better. Default minimizes.
    seed:
        Seed for reproducible sampling. Sampling happens in the calling
        thread, so a seeded run gives the same result with any ``workers``.
    keep_trials:
        Store every ``(params, score)`` pair in ``result.extra["trials"]``.
    workers, backend, callback:
        As for :func:`grid_search`.
    """
    if not space:
        raise EmptySpaceError("Search space is empty")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    rng = random.Random(seed)
    samplers = {name: _sampler(name, spec) for name, spec in space.items()}
    candidates = (
        {name: sample(rng) for name, sample in samplers.items()} for _ in range(n_iter)
    )
    return _run_trials(
        objective, candidates, maximize, "random_search", keep_trials, workers, backend, callback
    )
