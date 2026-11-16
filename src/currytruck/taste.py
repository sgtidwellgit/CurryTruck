"""taste — objective evaluation: fitness functions, scoring, and penalties.

Building blocks for writing objective functions, plus the classic benchmark
functions for testing optimizers.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Mapping, Optional, Sequence


def weighted_sum(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Combine named scores into one number: ``sum(weights[k] * scores[k])``.

    Keys in ``weights`` missing from ``scores`` raise ``KeyError``; extra keys
    in ``scores`` are ignored.

    >>> weighted_sum({"cost": 10, "time": 2}, {"cost": 1.0, "time": 5.0})
    20.0
    """
    return float(sum(w * scores[k] for k, w in weights.items()))


def minmax_normalize(values: Sequence[float], *, invert: bool = False) -> list[float]:
    """Rescale values to ``[0, 1]``. Constant input maps to all ``0.0``.

    ``invert=True`` maps the smallest value to 1 (useful for costs).

    >>> minmax_normalize([10, 20, 30])
    [0.0, 0.5, 1.0]
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [0.0] * len(values)
    out = [(v - lo) / span for v in values]
    return [1.0 - v for v in out] if invert else out


def penalized(
    objective: Callable[..., float],
    constraints: Sequence[Callable[..., float]],
    *,
    weight: float = 1e6,
    maximize: bool = False,
    squared: bool = True,
) -> Callable[..., float]:
    """Wrap ``objective`` with a penalty for violated constraints.

    Each constraint is a function of the same arguments returning a
    *violation amount*: ``<= 0`` means satisfied, ``> 0`` is how far it is
    broken. Turns a constrained problem into an unconstrained one for
    ``simmer`` / ``heat`` / ``spice``.

    Parameters
    ----------
    objective:
        The original objective.
    constraints:
        Violation functions (``g(x) <= 0`` form).
    weight:
        Penalty multiplier.
    maximize:
        Subtract the penalty instead of adding it.
    squared:
        Penalize the squared violation (smooth) rather than the absolute one.

    Examples
    --------
    >>> f = penalized(lambda x: x[0], [lambda x: 1 - x[0]], weight=10)   # x >= 1
    >>> f([2.0]), f([0.0])
    (2.0, 10.0)
    """

    @functools.wraps(objective)
    def wrapped(*args: Any, **kwargs: Any) -> float:
        value = float(objective(*args, **kwargs))
        total = 0.0
        for g in constraints:
            v = max(0.0, float(g(*args, **kwargs)))
            total += v * v if squared else v
        return value - weight * total if maximize else value + weight * total

    return wrapped


def counted(f: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``f`` so it counts its calls in ``.calls``; ``.reset()`` zeroes it.

    >>> g = counted(lambda x: x * 2)
    >>> g(1), g(2), g.calls
    (2, 4, 2)
    """

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.calls += 1  # type: ignore[attr-defined]
        return f(*args, **kwargs)

    def reset() -> None:
        wrapper.calls = 0  # type: ignore[attr-defined]

    wrapper.calls = 0  # type: ignore[attr-defined]
    wrapper.reset = reset  # type: ignore[attr-defined]
    return wrapper


def score_all(
    objective: Callable[[Any], float],
    candidates: Sequence[Any],
    *,
    maximize: bool = False,
    top: Optional[int] = None,
) -> list[tuple[Any, float]]:
    """Evaluate every candidate and return ``(candidate, score)`` pairs, best first.

    >>> score_all(lambda x: abs(x - 3), [1, 3, 5, 4], top=2)
    [(3, 0.0), (4, 1.0)]
    """
    scored = [(c, float(objective(c))) for c in candidates]
    scored.sort(key=lambda p: p[1], reverse=maximize)
    return scored[:top] if top is not None else scored


# --- Benchmark functions (all have a global minimum of 0) --------------------


def sphere(x: Sequence[float]) -> float:
    """Sum of squares. Convex, minimum 0 at the origin."""
    return float(sum(v * v for v in x))


def rosenbrock(x: Sequence[float]) -> float:
    """Rosenbrock's banana valley. Minimum 0 at ``(1, 1, ..., 1)``."""
    return float(
        sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))
    )


def rastrigin(x: Sequence[float]) -> float:
    """Highly multimodal. Minimum 0 at the origin; search domain ``[-5.12, 5.12]``."""
    return float(10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x))


def ackley(x: Sequence[float]) -> float:
    """Nearly flat outer region with a deep central hole. Minimum 0 at the origin."""
    n = len(x)
    s1 = sum(v * v for v in x) / n
    s2 = sum(math.cos(2 * math.pi * v) for v in x) / n
    return float(-20 * math.exp(-0.2 * math.sqrt(s1)) - math.exp(s2) + 20 + math.e)
