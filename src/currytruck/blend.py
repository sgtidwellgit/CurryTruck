"""blend — combinatorial optimization: knapsack, subset selection, and routing.

Pick the ingredients that make the best dish without overfilling the pot —
then work out the best way to deliver it.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Optional, Sequence, Union

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


# --- Routing -----------------------------------------------------------------

Matrix = Sequence[Sequence[float]]


def _distance_matrix(distances: Optional[Matrix], points: Optional[Sequence[Sequence[float]]]) -> list[list[float]]:
    if (distances is None) == (points is None):
        raise ValueError("Pass exactly one of distances or points")
    if points is not None:
        pts = [tuple(p) for p in points]
        return [[math.dist(a, b) for b in pts] for a in pts]
    d = [[float(v) for v in row] for row in distances]  # type: ignore[union-attr]
    if any(len(row) != len(d) for row in d):
        raise ValueError("distances must be a square matrix")
    return d


def _tour_length(d: list[list[float]], tour: Sequence[int], closed: bool) -> float:
    total = sum(d[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    if closed and len(tour) > 1:
        total += d[tour[-1]][tour[0]]
    return total


def _held_karp(d: list[list[float]], start: int, closed: bool) -> list[int]:
    n = len(d)
    others = [i for i in range(n) if i != start]
    m = len(others)
    # best[(mask, j)] = (cost, prev) — cheapest path from start visiting mask, ending at others[j]
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for j in range(m):
        best[(1 << j, j)] = (d[start][others[j]], -1)
    for size in range(2, m + 1):
        for subset in itertools.combinations(range(m), size):
            mask = 0
            for j in subset:
                mask |= 1 << j
            for j in subset:
                prev_mask = mask ^ (1 << j)
                best[(mask, j)] = min(
                    (best[(prev_mask, k)][0] + d[others[k]][others[j]], k)
                    for k in subset
                    if k != j
                )
    full = (1 << m) - 1
    end = min(
        range(m),
        key=lambda j: best[(full, j)][0] + (d[others[j]][start] if closed else 0.0),
    )
    path = []
    mask, j = full, end
    while j != -1:
        path.append(others[j])
        _, k = best[(mask, j)]
        mask ^= 1 << j
        j = k
    return [start] + path[::-1]


def _two_opt(d: list[list[float]], tour: list[int], closed: bool) -> list[int]:
    # Reverse segments while that shortens the tour; position 0 stays put
    n = len(tour)
    symmetric = all(d[i][j] == d[j][i] for i in range(len(d)) for j in range(i + 1, len(d)))
    best_len = _tour_length(d, tour, closed)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if symmetric:
                    # only the two edges around the reversed segment change
                    a, b, c = tour[i - 1], tour[i], tour[j]
                    if j + 1 < n or closed:
                        e = tour[(j + 1) % n]
                        delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
                    else:
                        delta = d[a][c] - d[a][b]
                    if delta < -1e-12:
                        tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                        best_len += delta
                        improved = True
                else:
                    cand = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                    length = _tour_length(d, cand, closed)
                    if length < best_len - 1e-12:
                        tour, best_len, improved = cand, length, True
    return tour


def shortest_route(
    distances: Optional[Matrix] = None,
    *,
    points: Optional[Sequence[Sequence[float]]] = None,
    start: int = 0,
    closed: bool = True,
    exact_limit: int = 12,
) -> Result:
    """Find the shortest route visiting every stop once (traveling salesman).

    Up to ``exact_limit`` stops the route is provably optimal (Held-Karp
    dynamic programming). Beyond that, a nearest-neighbor route is improved
    by 2-opt segment reversals until no reversal helps — fast and usually
    within a few percent of optimal.

    Parameters
    ----------
    distances:
        Square matrix of travel costs (may be asymmetric).
    points:
        Coordinates instead of a matrix; Euclidean distance is used.
    start:
        Index of the starting stop (the depot).
    closed:
        Return to ``start`` at the end (a tour). ``False`` gives an open path
        that ends wherever is cheapest.
    exact_limit:
        Largest number of stops solved exactly. Exact cost grows as
        ``n² · 2ⁿ`` — 12 takes well under a second; 16 takes minutes.

    Returns
    -------
    Result
        ``x`` is the visiting order starting with ``start`` (the return leg
        isn't repeated); ``score`` the total length; ``extra["exact"]``
        whether optimality is guaranteed.

    Examples
    --------
    >>> r = shortest_route(points=[(0, 0), (2, 2), (0, 2), (2, 0)])
    >>> r.x, r.score
    ([0, 3, 1, 2], 8.0)
    """
    d = _distance_matrix(distances, points)
    n = len(d)
    if n == 0:
        raise EmptySpaceError("No stops to route")
    if not 0 <= start < n:
        raise IndexError(f"start={start} out of range for {n} stops")

    if n <= 2:
        tour = [start] + [i for i in range(n) if i != start]
        exact = True
    elif n <= exact_limit:
        tour = _held_karp(d, start, closed)
        exact = True
    else:
        tour = [start]
        left = set(range(n)) - {start}
        while left:
            nxt = min(left, key=lambda j: d[tour[-1]][j])
            tour.append(nxt)
            left.remove(nxt)
        tour = _two_opt(d, tour, closed)
        exact = False

    length = _tour_length(d, tour, closed)
    return Result(
        x=tour,
        score=length,
        maximize=False,
        history=[length],
        n_evals=0,
        n_iter=1,
        converged=True,
        message="Optimal (Held-Karp)" if exact else "2-opt local optimum",
        method="shortest_route",
        extra={"exact": exact},
    )


def vehicle_routes(
    distances: Optional[Matrix] = None,
    *,
    points: Optional[Sequence[Sequence[float]]] = None,
    demands: Sequence[float],
    capacity: float,
    depot: int = 0,
    n_vehicles: Optional[int] = None,
    time_windows: Optional[Sequence[Optional[tuple[float, float]]]] = None,
    service_times: Optional[Union[float, Sequence[float]]] = None,
    travel_times: Optional[Matrix] = None,
) -> Result:
    """Plan delivery routes for a fleet of capacity-limited vehicles.

    Every route starts and ends at ``depot``. Uses the Clarke-Wright savings
    heuristic (merge routes where sharing a vehicle saves the most
    distance), checking capacity and time windows on every merge, then
    shortens each route with 2-opt.

    Parameters
    ----------
    distances, points:
        Travel costs, as for :func:`shortest_route`. Minimized in total.
    demands:
        Load to deliver to each stop (the depot's entry is ignored).
    capacity:
        Vehicle capacity.
    depot:
        Index of the depot.
    n_vehicles:
        Fleet size limit. ``None`` means as many as needed.
    time_windows:
        ``(earliest, latest)`` arrival per stop, or ``None`` for no window. A
        vehicle arriving early waits. The depot's window bounds the whole
        working day (leave no earlier than its start, return by its end).
    service_times:
        Time spent at each stop — one number or one per stop (default 0).
    travel_times:
        Travel-time matrix for time windows. Defaults to ``distances``.

    Returns
    -------
    Result
        ``x`` is a list of routes, each a list of stop indices (depot
        excluded); ``score`` the total distance; ``extra`` has per-route
        ``loads``, ``distances`` and ``arrivals``.

    Raises
    ------
    InfeasibleError
        A stop can't be served by any vehicle (demand over capacity, or its
        time window can't be reached), or no plan was found within
        ``n_vehicles``. The latter is a heuristic verdict.

    Examples
    --------
    >>> pts = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
    >>> r = vehicle_routes(points=pts, demands=[0, 1, 1, 1, 1], capacity=2)
    >>> sorted(sorted(route) for route in r.x), r.score
    ([[1, 2], [3, 4]], 8.0)
    """
    d = _distance_matrix(distances, points)
    n = len(d)
    if not 0 <= depot < n:
        raise IndexError(f"depot={depot} out of range for {n} stops")
    if len(demands) != n:
        raise ValueError(f"demands has {len(demands)} entries, expected {n}")
    tt = [[float(v) for v in row] for row in travel_times] if travel_times is not None else d
    if len(tt) != n or any(len(row) != n for row in tt):
        raise ValueError(f"travel_times must be {n}x{n}")
    if time_windows is not None and len(time_windows) != n:
        raise ValueError(f"time_windows has {len(time_windows)} entries, expected {n}")
    if service_times is None:
        service = [0.0] * n
    elif isinstance(service_times, (int, float)):
        service = [float(service_times)] * n
    else:
        service = [float(v) for v in service_times]
        if len(service) != n:
            raise ValueError(f"service_times has {len(service)} entries, expected {n}")
    tw = [
        (float("-inf"), float("inf")) if time_windows is None or time_windows[i] is None
        else (float(time_windows[i][0]), float(time_windows[i][1]))  # type: ignore[index]
        for i in range(n)
    ]

    def schedule(route: Sequence[int]) -> Optional[list[float]]:
        """Arrival times along a route, or None if a window is missed."""
        t = max(tw[depot][0], 0.0) if tw[depot][0] != float("-inf") else 0.0
        prev = depot
        arrivals = []
        for c in route:
            t += tt[prev][c]
            t = max(t, tw[c][0])
            if t > tw[c][1] + 1e-9:
                return None
            arrivals.append(t)
            t += service[c]
            prev = c
        if t + tt[prev][depot] > tw[depot][1] + 1e-9:
            return None
        return arrivals

    def load(route: Sequence[int]) -> float:
        return sum(demands[c] for c in route)

    def feasible(route: Sequence[int]) -> bool:
        return load(route) <= capacity + 1e-9 and schedule(route) is not None

    customers = [i for i in range(n) if i != depot]
    for c in customers:
        if demands[c] > capacity:
            raise InfeasibleError(f"Stop {c} demand {demands[c]} exceeds vehicle capacity {capacity}")
        if schedule([c]) is None:
            raise InfeasibleError(f"Stop {c} can't be reached within its time window")

    routes: dict[int, list[int]] = {c: [c] for c in customers}  # route id -> stops
    route_of = {c: c for c in customers}
    savings = sorted(
        ((d[i][depot] + d[depot][j] - d[i][j], i, j) for i in customers for j in customers if i != j),
        reverse=True,
    )
    for s, i, j in savings:
        if s <= 0:
            break
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        options = []
        if a[-1] == i and b[0] == j:
            options.append(a + b)
        if a[-1] == i and b[-1] == j:
            options.append(a + b[::-1])
        if a[0] == i and b[0] == j:
            options.append(a[::-1] + b)
        if a[0] == i and b[-1] == j:
            options.append(b + a)
        for merged in options:
            if feasible(merged):
                routes[ri] = merged
                del routes[rj]
                for c in merged:
                    route_of[c] = ri
                break

    plan = []
    for r in routes.values():
        best = r
        best_len = _tour_length(d, [depot] + r, True)
        cand = _two_opt(d, [depot] + r, True)[1:]
        if feasible(cand) and _tour_length(d, [depot] + cand, True) < best_len - 1e-12:
            best = cand
        plan.append(best)
    plan.sort(key=lambda r: r[0])

    if n_vehicles is not None and len(plan) > n_vehicles:
        raise InfeasibleError(
            f"Could not fit all stops into {n_vehicles} vehicles (needed {len(plan)}); "
            "the heuristic may miss tighter packings"
        )

    route_lengths = [_tour_length(d, [depot] + r, True) for r in plan]
    total = sum(route_lengths)
    return Result(
        x=plan,
        score=total,
        maximize=False,
        history=[total],
        n_evals=0,
        n_iter=1,
        converged=True,
        message=f"{len(plan)} route(s)",
        method="vehicle_routes",
        extra={
            "loads": [load(r) for r in plan],
            "distances": route_lengths,
            "arrivals": [schedule(r) for r in plan],
        },
    )
