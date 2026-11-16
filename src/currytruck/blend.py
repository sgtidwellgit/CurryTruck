"""blend — combinatorial optimization: knapsack and subset selection.

Pick the ingredients that make the best dish without overfilling the pot.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Optional, Sequence

from currytruck.exceptions import EmptySpaceError, InfeasibleError
from currytruck.result import Result

# DP table size above which knapsack switches to branch-and-bound
_DP_CELL_LIMIT = 10_000_000


def _knapsack_dp(values: Sequence[float], weights: Sequence[int], capacity: int) -> list[int]:
    n = len(values)
    # keep[i][w] — whether item i is taken at remaining capacity w
    best = [0.0] * (capacity + 1)
    keep = [bytearray(capacity + 1) for _ in range(n)]
    for i in range(n):
        wi, vi = weights[i], values[i]
        if wi > capacity:
            continue
        for w in range(capacity, wi - 1, -1):
            cand = best[w - wi] + vi
            if cand > best[w]:
                best[w] = cand
                keep[i][w] = 1
    chosen = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            chosen.append(i)
            w -= weights[i]
    return sorted(chosen)


def _knapsack_bnb(values: Sequence[float], weights: Sequence[float], capacity: float) -> list[int]:
    # Order by value density; zero-weight items are always taken if valuable
    order = sorted(
        range(len(values)),
        key=lambda i: math.inf if weights[i] == 0 else values[i] / weights[i],
        reverse=True,
    )
    v = [values[i] for i in order]
    w = [weights[i] for i in order]
    n = len(order)

    def bound(level: int, cur_v: float, cur_w: float) -> float:
        # Fractional (LP relaxation) upper bound on the remaining items
        b = cur_v
        room = capacity - cur_w
        for j in range(level, n):
            if v[j] <= 0:
                continue
            if w[j] <= room:
                room -= w[j]
                b += v[j]
            else:
                return b + v[j] * room / w[j]
        return b

    best_v = 0.0
    best_take: list[bool] = [False] * n
    take = [False] * n

    def search(level: int, cur_v: float, cur_w: float) -> None:
        nonlocal best_v, best_take
        if cur_v > best_v:
            best_v = cur_v
            best_take = take.copy()
        if level == n or bound(level, cur_v, cur_w) <= best_v:
            return
        if v[level] > 0 and cur_w + w[level] <= capacity:
            take[level] = True
            search(level + 1, cur_v + v[level], cur_w + w[level])
            take[level] = False
        search(level + 1, cur_v, cur_w)

    search(0, 0.0, 0.0)
    return sorted(order[j] for j in range(n) if best_take[j])


def knapsack(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
) -> Result:
    """Solve the 0/1 knapsack problem — maximize total value within a weight cap.

    Uses exact dynamic programming when all weights and the capacity are
    integers (and the table fits in memory), otherwise exact branch-and-bound.

    Parameters
    ----------
    values:
        Value of each item.
    weights:
        Non-negative weight of each item.
    capacity:
        Maximum total weight.

    Returns
    -------
    Result
        ``x`` is the sorted list of selected item indices, ``score`` the total
        value, ``extra["weight"]`` the total weight used.

    Examples
    --------
    >>> knapsack([60, 100, 120], [10, 20, 30], 50).x
    [1, 2]
    """
    if len(values) != len(weights):
        raise ValueError(f"values and weights differ in length ({len(values)} vs {len(weights)})")
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    all_int = isinstance(capacity, int) and all(isinstance(w, int) for w in weights)
    if all_int and len(values) * (capacity + 1) <= _DP_CELL_LIMIT:
        chosen = _knapsack_dp(values, weights, capacity)  # type: ignore[arg-type]
        method = "knapsack_dp"
    else:
        chosen = _knapsack_bnb(values, weights, capacity)
        method = "knapsack_bnb"

    total_value = sum(values[i] for i in chosen)
    return Result(
        x=chosen,
        score=total_value,
        maximize=True,
        history=[total_value],
        n_evals=0,
        n_iter=1,
        converged=True,
        message="Optimal",
        method=method,
        extra={"weight": sum(weights[i] for i in chosen)},
    )


def _better(a: float, b: float, maximize: bool) -> bool:
    return a > b if maximize else a < b


def best_subset(
    items: Sequence[Any],
    objective: Callable[[list[Any]], float],
    *,
    size: Optional[int] = None,
    min_size: int = 1,
    max_size: Optional[int] = None,
    constraint: Optional[Callable[[list[Any]], bool]] = None,
    maximize: bool = True,
    max_evals: int = 100_000,
) -> Result:
    """Choose the subset of ``items`` that optimizes ``objective``.

    Searches exhaustively when the number of candidate subsets is at most
    ``max_evals``; otherwise falls back to greedy forward selection (start
    empty, repeatedly add the item that improves the score most).

    Parameters
    ----------
    items:
        The pool to choose from.
    objective:
        Called with a list of chosen items; returns a number.
    size:
        Exact subset size. Overrides ``min_size`` / ``max_size``.
    min_size, max_size:
        Inclusive size range. ``max_size`` defaults to ``len(items)``.
    constraint:
        Optional feasibility check — subsets for which it returns ``False``
        are skipped.
    maximize:
        Default ``True`` — higher scores are better.
    max_evals:
        Exhaustive-search budget before switching to greedy.

    Returns
    -------
    Result
        ``x`` is the sorted list of chosen indices into ``items``;
        ``extra["items"]`` holds the chosen items themselves.
    """
    n = len(items)
    if n == 0:
        raise EmptySpaceError("items is empty")
    if size is not None:
        min_size = max_size = size
    if max_size is None:
        max_size = n
    if not 0 <= min_size <= max_size <= n:
        raise ValueError(f"Invalid size range [{min_size}, {max_size}] for {n} items")

    def feasible(idx: Sequence[int]) -> bool:
        return constraint is None or constraint([items[i] for i in idx])

    def score(idx: Sequence[int]) -> float:
        return float(objective([items[i] for i in idx]))

    total = sum(math.comb(n, k) for k in range(min_size, max_size + 1))
    best_idx: Optional[list[int]] = None
    best = 0.0
    history: list[float] = []
    evals = 0

    if total <= max_evals:
        for k in range(min_size, max_size + 1):
            for combo in itertools.combinations(range(n), k):
                if not feasible(combo):
                    continue
                s = score(combo)
                evals += 1
                if best_idx is None or _better(s, best, maximize):
                    best_idx, best = list(combo), s
                history.append(best)
        method, converged, message = "best_subset_exhaustive", True, "Exhaustive search"
    else:
        current: list[int] = []
        cur_score: Optional[float] = None
        if min_size == 0 and feasible(current):
            cur_score = score(current)
            evals += 1
            best_idx, best = [], cur_score
        while len(current) < max_size:
            step_idx, step_score = None, 0.0
            for i in range(n):
                if i in current:
                    continue
                cand = sorted(current + [i])
                if not feasible(cand):
                    continue
                s = score(cand)
                evals += 1
                if step_idx is None or _better(s, step_score, maximize):
                    step_idx, step_score = i, s
            if step_idx is None:
                break
            # past min_size, stop as soon as adding an item no longer helps
            if len(current) >= min_size and cur_score is not None and not _better(
                step_score, cur_score, maximize
            ):
                break
            current = sorted(current + [step_idx])
            cur_score = step_score
            if len(current) >= min_size and (
                best_idx is None or _better(cur_score, best, maximize)
            ):
                best_idx, best = current, cur_score
            history.append(best if best_idx is not None else cur_score)
        method, converged, message = "best_subset_greedy", False, "Greedy forward selection"

    if best_idx is None:
        raise InfeasibleError("No subset satisfies the size range and constraint")

    return Result(
        x=best_idx,
        score=best,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=len(history),
        converged=converged,
        message=message,
        method=method,
        extra={"items": [items[i] for i in best_idx]},
    )
