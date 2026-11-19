"""balance — multi-objective optimization with Pareto dominance.

When cheap, fast, and tasty pull in different directions, there is no single
best dish — only the ones nothing else beats on every count. ``nsga2``
evolves the whole set of best trade-offs at once.
"""

from __future__ import annotations

import random
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


def _sbx(a: float, b: float, lo: float, hi: float, eta: float, rng: random.Random) -> tuple[float, float]:
    # Simulated binary crossover: children spread around the parents like
    # single-point crossover does for bit strings
    u = rng.random()
    beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return min(hi, max(lo, c1)), min(hi, max(lo, c2))


def _poly_mutation(x: float, lo: float, hi: float, eta: float, rng: random.Random) -> float:
    if hi == lo:
        return lo
    u = rng.random()
    if u < 0.5:
        delta = (2 * u) ** (1 / (eta + 1)) - 1
    else:
        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
    return min(hi, max(lo, x + delta * (hi - lo)))


def nsga2(
    objectives: Sequence[Callable[[list[float]], float]],
    bounds: Sequence[tuple[float, float]],
    *,
    population: int = 60,
    generations: int = 100,
    maximize: Senses = False,
    crossover_rate: float = 0.9,
    mutation_rate: Optional[float] = None,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, list[tuple[list[float], tuple[float, ...]]]], Optional[bool]]] = None,
) -> list[tuple[list[float], tuple[float, ...]]]:
    """Evolve an approximation of the Pareto front (NSGA-II).

    Each generation, parents and offspring are pooled, sorted into
    non-dominated fronts, and the best fronts survive — ties on the last
    front broken in favor of less crowded points, which keeps the front
    spread out. Offspring come from binary tournaments, simulated binary
    crossover, and polynomial mutation.

    Parameters
    ----------
    objectives:
        Functions scoring a candidate (a list of floats).
    bounds:
        ``[(lo, hi), ...]`` per decision variable.
    population:
        Population size (rounded up to an even number).
    generations:
        Number of generations.
    maximize:
        One bool for all objectives or one per objective.
    crossover_rate:
        Probability a pair of parents is recombined.
    mutation_rate:
        Per-variable mutation probability. Defaults to ``1 / len(bounds)``.
    eta_crossover, eta_mutation:
        Distribution indices — larger keeps children closer to their parents.
    seed:
        Seed for reproducible runs.
    callback:
        ``callback(generation, front)`` after each generation, where
        ``front`` is the current list of ``(x, scores)``; return ``True`` to
        stop early.

    Returns
    -------
    list of ``(x, scores)`` on the final non-dominated front, sorted by the
    first objective.

    Examples
    --------
    Trade off two parabolas — the true front is every ``x`` in ``[0, 2]``:

    >>> front = nsga2([lambda v: v[0] ** 2, lambda v: (v[0] - 2) ** 2], [(-5, 5)], seed=0)
    >>> all(-0.05 <= x[0] <= 2.05 for x, _ in front), len(front) > 20
    (True, True)
    """
    if not objectives:
        raise ValueError("objectives is empty")
    if not bounds:
        raise ValueError("bounds is empty")
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    for lo, hi in bounds:
        if lo > hi:
            raise ValueError(f"Lower bound {lo} exceeds upper bound {hi}")
    if population < 4:
        raise ValueError(f"population must be >= 4, got {population}")
    population += population % 2
    k = len(objectives)
    senses = _senses(maximize, k)
    dims = len(bounds)
    p_mut = 1.0 / dims if mutation_rate is None else mutation_rate
    rng = random.Random(seed)

    def evaluate(x: list[float]) -> tuple[float, ...]:
        return tuple(float(f(x)) for f in objectives)

    def survivors(pool: list[list[float]], scores: list[tuple[float, ...]]) -> tuple[list[int], list[int], list[float]]:
        """Indices of the best ``population`` members, with their ranks and crowding."""
        ranks = pareto_rank(scores, senses)
        chosen: list[int] = []
        crowd = [0.0] * len(pool)
        for r in range(max(ranks) + 1):
            front = [i for i in range(len(pool)) if ranks[i] == r]
            dist = crowding_distance([scores[i] for i in front])
            for i, dv in zip(front, dist):
                crowd[i] = dv
            if len(chosen) + len(front) <= population:
                chosen.extend(front)
            else:
                order = sorted(range(len(front)), key=lambda p: dist[p], reverse=True)
                chosen.extend(front[p] for p in order[: population - len(chosen)])
                break
        return chosen, ranks, crowd

    pop = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(population)]
    pop_scores = [evaluate(x) for x in pop]
    _, ranks, crowd = survivors(pop, pop_scores)
    pop_rank = ranks
    pop_crowd = crowd

    def front_of(xs: list[list[float]], ss: list[tuple[float, ...]], rs: list[int]) -> list[tuple[list[float], tuple[float, ...]]]:
        seen = set()
        out = []
        for x, s, r in zip(xs, ss, rs):
            if r == 0 and tuple(x) not in seen:
                seen.add(tuple(x))
                out.append((list(x), s))
        return sorted(out, key=lambda p: p[1][0])

    for gen in range(1, generations + 1):
        def tournament() -> list[float]:
            a, b = rng.randrange(population), rng.randrange(population)
            if pop_rank[a] != pop_rank[b]:
                return pop[a] if pop_rank[a] < pop_rank[b] else pop[b]
            return pop[a] if pop_crowd[a] >= pop_crowd[b] else pop[b]

        offspring: list[list[float]] = []
        while len(offspring) < population:
            p1, p2 = tournament(), tournament()
            c1, c2 = list(p1), list(p2)
            if rng.random() < crossover_rate:
                for d, (lo, hi) in enumerate(bounds):
                    if rng.random() < 0.5:
                        c1[d], c2[d] = _sbx(p1[d], p2[d], lo, hi, eta_crossover, rng)
            for child in (c1, c2):
                for d, (lo, hi) in enumerate(bounds):
                    if rng.random() < p_mut:
                        child[d] = _poly_mutation(child[d], lo, hi, eta_mutation, rng)
            offspring.extend([c1, c2])

        pool = pop + offspring
        pool_scores = pop_scores + [evaluate(x) for x in offspring]
        chosen, ranks, crowd = survivors(pool, pool_scores)
        pop = [pool[i] for i in chosen]
        pop_scores = [pool_scores[i] for i in chosen]
        # ranks within the survivors are unchanged by dropping worse fronts
        pop_rank = [ranks[i] for i in chosen]
        pop_crowd = [crowd[i] for i in chosen]
        if callback is not None and callback(gen, front_of(pop, pop_scores, pop_rank)):
            break

    return front_of(pop, pop_scores, pop_rank)
