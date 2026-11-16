"""balance — multi-objective optimization with Pareto dominance.

When cheap, fast, and tasty pull in different directions, there is no single
best dish — only the ones nothing else beats on every count.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

from currytruck.exceptions import EmptySpaceError

Senses = Union[bool, Sequence[bool]]


def _senses(maximize: Senses, k: int) -> list[bool]:
    if isinstance(maximize, bool):
        return [maximize] * k
    senses = list(maximize)
    if len(senses) != k:
        raise ValueError(f"maximize has {len(senses)} entries for {k} objectives")
    return senses


def dominates(a: Sequence[float], b: Sequence[float], maximize: Senses = False) -> bool:
    """``True`` if ``a`` is at least as good as ``b`` everywhere and better somewhere.

    ``maximize`` is one bool for all objectives or one per objective.

    >>> dominates([1, 2], [2, 2])
    True
    >>> dominates([1, 3], [2, 2])
    False
    """
    if len(a) != len(b):
        raise ValueError("Points have different numbers of objectives")
    senses = _senses(maximize, len(a))
    strictly = False
    for x, y, mx in zip(a, b, senses):
        if mx:
            x, y = -x, -y
        if x > y:
            return False
        if x < y:
            strictly = True
    return strictly


def pareto_front(points: Sequence[Sequence[float]], maximize: Senses = False) -> list[int]:
    """Indices of the non-dominated points, in input order.

    >>> pareto_front([(1, 5), (2, 2), (3, 3), (5, 1)])
    [0, 1, 3]
    """
    return [
        i
        for i, p in enumerate(points)
        if not any(dominates(q, p, maximize) for j, q in enumerate(points) if j != i)
    ]


def pareto_rank(points: Sequence[Sequence[float]], maximize: Senses = False) -> list[int]:
    """Non-dominated sorting: rank 0 is the Pareto front, rank 1 the front once
    rank 0 is removed, and so on. Returns one rank per point.

    >>> pareto_rank([(1, 5), (2, 2), (3, 3), (5, 1)])
    [0, 0, 1, 0]
    """
    n = len(points)
    ranks = [-1] * n
    remaining = set(range(n))
    rank = 0
    while remaining:
        front = [
            i
            for i in remaining
            if not any(dominates(points[j], points[i], maximize) for j in remaining if j != i)
        ]
        for i in front:
            ranks[i] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def crowding_distance(points: Sequence[Sequence[float]]) -> list[float]:
    """Crowding distance of each point within a front (NSGA-II).

    Boundary points get ``inf``. Larger means more isolated — useful for
    picking a well-spread subset of a front.
    """
    n = len(points)
    if n == 0:
        return []
    k = len(points[0])
    dist = [0.0] * n
    for m in range(k):
        order = sorted(range(n), key=lambda i: points[i][m])
        lo, hi = points[order[0]][m], points[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = float("inf")
        if hi == lo:
            continue
        for pos in range(1, n - 1):
            i = order[pos]
            dist[i] += (points[order[pos + 1]][m] - points[order[pos - 1]][m]) / (hi - lo)
    return dist


def pareto_search(
    objectives: Sequence[Callable[[Any], float]],
    candidates: Sequence[Any],
    *,
    maximize: Senses = False,
    top: Optional[int] = None,
) -> list[tuple[Any, tuple[float, ...]]]:
    """Evaluate every candidate on every objective and return the Pareto front.

    Parameters
    ----------
    objectives:
        Functions scoring a candidate.
    candidates:
        Solutions to compare.
    maximize:
        One bool for all objectives or one per objective.
    top:
        If set, keep at most this many front members, preferring the most
        spread out (largest crowding distance).

    Returns
    -------
    list of ``(candidate, scores)`` tuples on the front, in input order.

    Examples
    --------
    >>> menu = [{"price": 8, "wait": 10}, {"price": 12, "wait": 3}, {"price": 13, "wait": 9}]
    >>> front = pareto_search([lambda d: d["price"], lambda d: d["wait"]], menu)
    >>> [c["price"] for c, _ in front]
    [8, 12]
    """
    if not candidates:
        raise EmptySpaceError("No candidates to evaluate")
    scores = [tuple(float(f(c)) for f in objectives) for c in candidates]
    front = pareto_front(scores, maximize)
    if top is not None and len(front) > top:
        dist = crowding_distance([scores[i] for i in front])
        keep = sorted(range(len(front)), key=lambda p: dist[p], reverse=True)[:top]
        front = sorted(front[p] for p in keep)
    return [(candidates[i], scores[i]) for i in front]
