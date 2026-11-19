"""heat — global optimization: simulated annealing, genetic algorithms,
particle swarms, and tabu search.

Turn up the heat to escape local optima. The population and annealing
methods take a ``heat`` level (1-5): higher heat means more exploration — a
hotter starting temperature for annealing, a higher mutation rate for the
genetic algorithm, more inertia for the particle swarm.
"""

from __future__ import annotations

import math
import random
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Hashable, Optional, Sequence

from currytruck.exceptions import InvalidHeatLevel
from currytruck.result import Result

Callback = Callable[[int, Any, float], Optional[bool]]

# heat -> multiplier on the estimated starting temperature
_SA_TEMP = {1: 0.1, 2: 0.5, 3: 1.0, 4: 2.0, 5: 5.0}
# heat -> per-gene mutation probability
_GA_MUTATION = {1: 0.01, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.4}
# heat -> particle inertia weight (3 is the standard constriction value)
_PSO_INERTIA = {1: 0.4, 2: 0.6, 3: 0.7298, 4: 0.85, 5: 0.95}


def _check_heat(heat: int) -> None:
    if heat not in _SA_TEMP:
        raise InvalidHeatLevel(f"heat must be an integer 1-5, got {heat!r}")


def _make_pool(workers: Optional[int], backend: str) -> Optional[Executor]:
    if backend not in ("thread", "process"):
        raise ValueError(f"backend must be 'thread' or 'process', got {backend!r}")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers is None or workers == 1:
        return None
    return (ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor)(max_workers=workers)


def _evaluate(objective: Callable[[Any], float], items: list[Any], pool: Optional[Executor]) -> list[float]:
    if pool is None:
        return [float(objective(i)) for i in items]
    return [float(v) for v in pool.map(objective, items)]


def swap_neighbor(x: Sequence[Any], rng: random.Random) -> list[Any]:
    """Return a copy of ``x`` with two random positions swapped.

    Pass as ``neighbor=`` to :func:`simulated_annealing` or
    :func:`tabu_search` to search over orderings (delivery routes, task
    sequences, ...).
    """
    y = list(x)
    if len(y) >= 2:
        i, j = rng.sample(range(len(y)), 2)
        y[i], y[j] = y[j], y[i]
    return y


def reverse_neighbor(x: Sequence[Any], rng: random.Random) -> list[Any]:
    """Return a copy of ``x`` with a random segment reversed (a 2-opt move).

    Usually a stronger move than :func:`swap_neighbor` for routing problems.
    """
    y = list(x)
    if len(y) >= 2:
        i, j = sorted(rng.sample(range(len(y)), 2))
        y[i : j + 1] = reversed(y[i : j + 1])
    return y


def _gaussian_neighbor(
    bounds: Optional[Sequence[tuple[float, float]]], step: Optional[float]
) -> Callable[[Sequence[float], random.Random], list[float]]:
    def neighbor(x: Sequence[float], rng: random.Random) -> list[float]:
        y = list(x)
        i = rng.randrange(len(y))
        if bounds is not None:
            lo, hi = bounds[i]
            s = step if step is not None else 0.1 * (hi - lo)
            y[i] = min(hi, max(lo, y[i] + rng.gauss(0, s)))
        else:
            s = step if step is not None else 0.1 * max(1.0, abs(y[i]))
            y[i] = y[i] + rng.gauss(0, s)
        return y

    return neighbor


def _default_neighbor(
    x0: Any,
    neighbor: Optional[Callable[[Any, random.Random], Any]],
    bounds: Optional[Sequence[tuple[float, float]]],
    step: Optional[float],
) -> Callable[[Any, random.Random], Any]:
    if neighbor is not None:
        return neighbor
    if bounds is not None and len(bounds) != len(x0):
        raise ValueError("bounds must have one (lo, hi) pair per dimension")
    return _gaussian_neighbor(bounds, step)


def simulated_annealing(
    objective: Callable[[Any], float],
    x0: Any,
    *,
    neighbor: Optional[Callable[[Any, random.Random], Any]] = None,
    bounds: Optional[Sequence[tuple[float, float]]] = None,
    step: Optional[float] = None,
    heat: int = 3,
    max_iter: int = 10_000,
    maximize: bool = False,
    seed: Optional[int] = None,
    callback: Optional[Callback] = None,
) -> Result:
    """Search for a global optimum by simulated annealing.

    Parameters
    ----------
    objective:
        Called with a candidate solution; returns a number.
    x0:
        Starting solution.
    neighbor:
        ``neighbor(x, rng) -> new_x`` producing a nearby candidate. Must not
        mutate ``x``. Defaults to perturbing one coordinate of a numeric
        vector with Gaussian noise. Use :func:`swap_neighbor` or
        :func:`reverse_neighbor` for orderings.
    bounds:
        ``[(lo, hi), ...]`` per dimension for the default neighbor; values
        are clipped to the box.
    step:
        Gaussian step size for the default neighbor. Defaults to 10% of each
        bound's width, or 10% of the coordinate's magnitude if unbounded.
    heat:
        1-5. Scales the starting temperature — higher heat accepts more
        uphill moves early on.
    max_iter:
        Number of annealing steps. Temperature cools geometrically to 1e-4 of
        its starting value over this many steps.
    maximize:
        If ``True``, higher scores are better.
    seed:
        Seed for reproducible runs.
    callback:
        ``callback(iteration, best_x, best_score)`` after each step; return
        ``True`` to stop early.
    """
    _check_heat(heat)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    rng = random.Random(seed)
    neighbor = _default_neighbor(x0, neighbor, bounds, step)

    sign = -1.0 if maximize else 1.0

    current = x0
    cur_cost = sign * float(objective(current))
    evals = 1

    # Estimate a starting temperature from the typical size of a move
    deltas = []
    for _ in range(20):
        cand = neighbor(current, rng)
        deltas.append(abs(sign * float(objective(cand)) - cur_cost))
        evals += 1
    scale = sum(deltas) / len(deltas) or 1.0
    t0 = scale * _SA_TEMP[heat]
    alpha = (1e-4) ** (1.0 / max_iter)

    best, best_cost = current, cur_cost
    history = []
    temp = t0
    accepted = 0
    message = f"Completed {max_iter} annealing steps"
    it = 0
    for it in range(1, max_iter + 1):
        cand = neighbor(current, rng)
        cand_cost = sign * float(objective(cand))
        evals += 1
        delta = cand_cost - cur_cost
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current, cur_cost = cand, cand_cost
            accepted += 1
            if cur_cost < best_cost:
                best, best_cost = current, cur_cost
        history.append(sign * best_cost)
        temp *= alpha
        if callback is not None and callback(it, best, sign * best_cost):
            message = "Stopped by callback"
            break

    return Result(
        x=best,
        score=sign * best_cost,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=it,
        converged=False,
        message=message,
        method="simulated_annealing",
        extra={"acceptance_rate": accepted / it, "start_temperature": t0},
    )


def _key(x: Any) -> Hashable:
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return x


def tabu_search(
    objective: Callable[[Any], float],
    x0: Any,
    *,
    neighbor: Optional[Callable[[Any, random.Random], Any]] = None,
    bounds: Optional[Sequence[tuple[float, float]]] = None,
    step: Optional[float] = None,
    n_candidates: int = 20,
    tenure: int = 10,
    max_iter: int = 1_000,
    patience: Optional[int] = None,
    key: Optional[Callable[[Any], Hashable]] = None,
    maximize: bool = False,
    seed: Optional[int] = None,
    callback: Optional[Callback] = None,
) -> Result:
    """Local search with a short-term memory that forbids revisiting solutions.

    Each iteration samples ``n_candidates`` neighbors and moves to the best
    one that is not tabu — even if it is worse than the current solution,
    which is how tabu search climbs out of local optima. A tabu candidate is
    still accepted if it beats the best solution found so far (aspiration).

    Best suited to discrete problems — orderings with :func:`swap_neighbor`
    / :func:`reverse_neighbor`, assignments, subsets.

    Parameters
    ----------
    objective:
        Called with a candidate solution; returns a number.
    x0:
        Starting solution.
    neighbor:
        ``neighbor(x, rng) -> new_x``; defaults as for
        :func:`simulated_annealing`.
    bounds, step:
        Settings for the default numeric neighbor.
    n_candidates:
        Neighbors sampled per iteration.
    tenure:
        How many recent solutions stay tabu.
    max_iter:
        Iteration cap.
    patience:
        Stop after this many iterations without improving the best.
    key:
        Maps a solution to a hashable value for the tabu list. Defaults to
        ``tuple(x)`` for lists/tuples, else ``x`` itself.
    maximize:
        If ``True``, higher scores are better.
    seed:
        Seed for reproducible runs.
    callback:
        ``callback(iteration, best_x, best_score)``; return ``True`` to stop.

    Examples
    --------
    >>> r = tabu_search(lambda p: sum(abs(v - i) for i, v in enumerate(p)),
    ...                 [3, 1, 2, 0], neighbor=swap_neighbor, seed=0, max_iter=50)
    >>> r.x, r.score
    ([0, 1, 2, 3], 0.0)
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if tenure < 0:
        raise ValueError(f"tenure must be >= 0, got {tenure}")
    rng = random.Random(seed)
    neighbor = _default_neighbor(x0, neighbor, bounds, step)
    key = key or _key
    sign = -1.0 if maximize else 1.0

    current = x0
    cur_cost = sign * float(objective(current))
    evals = 1
    best, best_cost = current, cur_cost
    tabu: deque[Hashable] = deque([key(current)], maxlen=tenure)
    history = []
    stale = 0
    converged = False
    message = f"Reached max_iter={max_iter}"
    it = 0

    for it in range(1, max_iter + 1):
        cands = [neighbor(current, rng) for _ in range(n_candidates)]
        costs = [sign * float(objective(c)) for c in cands]
        evals += n_candidates
        order = sorted(range(n_candidates), key=lambda i: costs[i])
        pick = None
        for i in order:
            if key(cands[i]) not in tabu or costs[i] < best_cost:
                pick = i
                break
        if pick is None:  # everything tabu — take the best anyway
            pick = order[0]
        current, cur_cost = cands[pick], costs[pick]
        tabu.append(key(current))
        if cur_cost < best_cost:
            best, best_cost = current, cur_cost
            stale = 0
        else:
            stale += 1
        history.append(sign * best_cost)
        if callback is not None and callback(it, best, sign * best_cost):
            message = "Stopped by callback"
            break
        if patience is not None and stale >= patience:
            converged, message = True, f"No improvement for {patience} iterations"
            break

    return Result(
        x=best,
        score=sign * best_cost,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=it,
        converged=converged,
        message=message,
        method="tabu_search",
    )


def _order_crossover(a: list[Any], b: list[Any], rng: random.Random) -> list[Any]:
    # OX1: copy a slice from parent a, fill the rest in parent b's order
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2)) if n >= 2 else (0, n)
    child: list[Any] = [None] * n
    child[i:j] = a[i:j]
    taken = list(a[i:j])
    fill = []
    for gene in b:
        if gene in taken:
            taken.remove(gene)  # handles duplicate genes correctly
        else:
            fill.append(gene)
    k = 0
    for pos in list(range(j, n)) + list(range(0, i)):
        child[pos] = fill[k]
        k += 1
    return child


def genetic_algorithm(
    objective: Callable[[list[Any]], float],
    bounds: Optional[Sequence[tuple[float, float]]] = None,
    *,
    permutation: Optional[Sequence[Any]] = None,
    integer: bool = False,
    population: int = 50,
    generations: int = 200,
    heat: int = 3,
    crossover_rate: float = 0.9,
    elite: int = 2,
    tournament: int = 3,
    patience: Optional[int] = None,
    maximize: bool = False,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    backend: str = "thread",
    callback: Optional[Callback] = None,
) -> Result:
    """Evolve a population of candidate solutions.

    Works in one of two modes — pass exactly one of ``bounds`` or
    ``permutation``:

    - **Vector mode** (``bounds=[(lo, hi), ...]``): each individual is a list
      of numbers within the bounds. Uniform crossover, Gaussian mutation.
      ``integer=True`` rounds genes to ints.
    - **Permutation mode** (``permutation=items``): each individual is an
      ordering of ``items``. Order crossover (OX1), swap mutation. Good for
      routing and sequencing.

    Parameters
    ----------
    objective:
        Called with an individual (a list); returns a number.
    population:
        Individuals per generation.
    generations:
        Maximum number of generations.
    heat:
        1-5. Sets the per-gene mutation probability
        (1: 1%, 2: 5%, 3: 10%, 4: 20%, 5: 40%).
    crossover_rate:
        Probability two selected parents are recombined rather than copied.
    elite:
        Number of best individuals carried unchanged into the next generation.
    tournament:
        Tournament size for parent selection.
    patience:
        Stop early after this many generations without improvement.
    maximize:
        If ``True``, higher scores are better.
    seed:
        Seed for reproducible runs. Breeding happens in the calling thread, so
        a seeded run gives the same result with any ``workers``.
    workers:
        Evaluate each generation's offspring in parallel with this many
        workers.
    backend:
        ``"thread"`` or ``"process"`` (the objective must then be picklable —
        a module-level function, not a lambda).
    callback:
        ``callback(generation, best_x, best_score)`` after each generation;
        return ``True`` to stop early.
    """
    _check_heat(heat)
    if (bounds is None) == (permutation is None):
        raise ValueError("Pass exactly one of bounds or permutation")
    if population < 2:
        raise ValueError(f"population must be >= 2, got {population}")
    if not 0 <= elite < population:
        raise ValueError(f"elite must be in [0, population), got {elite}")

    rng = random.Random(seed)
    mut_rate = _GA_MUTATION[heat]

    if bounds is not None:
        bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        for lo, hi in bounds:
            if lo > hi:
                raise ValueError(f"Lower bound {lo} exceeds upper bound {hi}")

        def fix(ind: list[Any]) -> list[Any]:
            return [round(g) for g in ind] if integer else ind

        def spawn() -> list[Any]:
            return fix([rng.uniform(lo, hi) for lo, hi in bounds])

        def crossover(a: list[Any], b: list[Any]) -> list[Any]:
            return [x if rng.random() < 0.5 else y for x, y in zip(a, b)]

        def mutate(ind: list[Any]) -> list[Any]:
            out = []
            for g, (lo, hi) in zip(ind, bounds):
                if rng.random() < mut_rate:
                    g = min(hi, max(lo, g + rng.gauss(0, 0.1 * (hi - lo) or 1.0)))
                out.append(g)
            return fix(out)

    else:
        items = list(permutation)  # type: ignore[arg-type]

        def spawn() -> list[Any]:
            return rng.sample(items, len(items))

        def crossover(a: list[Any], b: list[Any]) -> list[Any]:
            return _order_crossover(a, b, rng)

        def mutate(ind: list[Any]) -> list[Any]:
            out = list(ind)
            for i in range(len(out)):
                if rng.random() < mut_rate and len(out) >= 2:
                    j = rng.randrange(len(out))
                    out[i], out[j] = out[j], out[i]
            return out

    sign = -1.0 if maximize else 1.0
    pool = _make_pool(workers, backend)
    evals = 0

    def costs_of(inds: list[list[Any]]) -> list[float]:
        nonlocal evals
        evals += len(inds)
        return [sign * v for v in _evaluate(objective, inds, pool)]

    try:
        pop = [spawn() for _ in range(population)]
        costs = costs_of(pop)

        def select() -> list[Any]:
            contenders = rng.sample(range(population), min(tournament, population))
            return pop[min(contenders, key=lambda i: costs[i])]

        best_i = min(range(population), key=lambda i: costs[i])
        best, best_cost = pop[best_i], costs[best_i]
        history = [sign * best_cost]
        stale = 0
        converged = False
        message = f"Reached generations={generations}"
        gen = 0

        for gen in range(1, generations + 1):
            ranked = sorted(range(population), key=lambda i: costs[i])
            elites = [pop[i] for i in ranked[:elite]]
            elite_costs = [costs[i] for i in ranked[:elite]]
            children = []
            while len(elites) + len(children) < population:
                a, b = select(), select()
                child = crossover(a, b) if rng.random() < crossover_rate else list(a)
                children.append(mutate(child))
            pop = elites + children
            costs = elite_costs + costs_of(children)

            gen_best = min(range(population), key=lambda i: costs[i])
            if costs[gen_best] < best_cost:
                best, best_cost = pop[gen_best], costs[gen_best]
                stale = 0
            else:
                stale += 1
            history.append(sign * best_cost)
            if callback is not None and callback(gen, best, sign * best_cost):
                message = "Stopped by callback"
                break
            if patience is not None and stale >= patience:
                converged = True
                message = f"No improvement for {patience} generations"
                break
    finally:
        if pool is not None:
            pool.shutdown()

    return Result(
        x=best,
        score=sign * best_cost,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=gen,
        converged=converged,
        message=message,
        method="genetic_algorithm",
        extra={"mutation_rate": mut_rate},
    )


def particle_swarm(
    objective: Callable[[list[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    swarm: int = 30,
    max_iter: int = 200,
    heat: int = 3,
    cognitive: float = 1.49618,
    social: float = 1.49618,
    patience: Optional[int] = None,
    maximize: bool = False,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    backend: str = "thread",
    callback: Optional[Callback] = None,
) -> Result:
    """Particle swarm optimization over a box.

    Each particle is pulled toward its own best position (``cognitive``) and
    the swarm's best position (``social``), while keeping some of its
    previous velocity (inertia, set by ``heat``). Velocities are clamped to
    20% of each dimension's range; particles stop at the box walls.

    Parameters
    ----------
    objective:
        Called with a list of floats; returns a number.
    bounds:
        ``[(lo, hi), ...]`` per dimension.
    swarm:
        Number of particles.
    max_iter:
        Number of iterations.
    heat:
        1-5. Inertia weight (1: 0.4, 2: 0.6, 3: 0.7298, 4: 0.85, 5: 0.95) —
        higher keeps particles roaming longer.
    cognitive, social:
        Acceleration coefficients toward personal and global bests.
    patience:
        Stop after this many iterations without improving the global best.
    maximize:
        If ``True``, higher scores are better.
    seed:
        Seed for reproducible runs (same result with any ``workers``).
    workers, backend:
        Evaluate the swarm in parallel, as for :func:`genetic_algorithm`.
    callback:
        ``callback(iteration, best_x, best_score)``; return ``True`` to stop.

    Examples
    --------
    >>> r = particle_swarm(lambda v: (v[0] - 1) ** 2 + (v[1] + 2) ** 2, [(-5, 5), (-5, 5)], seed=0)
    >>> [round(v, 3) for v in r.x]
    [1.0, -2.0]
    """
    _check_heat(heat)
    if swarm < 1:
        raise ValueError(f"swarm must be >= 1, got {swarm}")
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    for lo, hi in bounds:
        if lo > hi:
            raise ValueError(f"Lower bound {lo} exceeds upper bound {hi}")
    if not bounds:
        raise ValueError("bounds is empty")

    rng = random.Random(seed)
    w = _PSO_INERTIA[heat]
    vmax = [0.2 * (hi - lo) for lo, hi in bounds]
    sign = -1.0 if maximize else 1.0
    pool = _make_pool(workers, backend)
    evals = 0

    def costs_of(xs: list[list[float]]) -> list[float]:
        nonlocal evals
        evals += len(xs)
        return [sign * v for v in _evaluate(objective, xs, pool)]

    try:
        pos = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(swarm)]
        vel = [[rng.uniform(-vm, vm) for vm in vmax] for _ in range(swarm)]
        costs = costs_of(pos)
        pbest = [list(p) for p in pos]
        pbest_cost = list(costs)
        g = min(range(swarm), key=lambda i: costs[i])
        gbest, gbest_cost = list(pos[g]), costs[g]
        history = [sign * gbest_cost]
        stale = 0
        converged = False
        message = f"Reached max_iter={max_iter}"
        it = 0

        for it in range(1, max_iter + 1):
            for i in range(swarm):
                for d, (lo, hi) in enumerate(bounds):
                    v = (
                        w * vel[i][d]
                        + cognitive * rng.random() * (pbest[i][d] - pos[i][d])
                        + social * rng.random() * (gbest[d] - pos[i][d])
                    )
                    v = max(-vmax[d], min(vmax[d], v))
                    x = pos[i][d] + v
                    if x < lo or x > hi:
                        x = min(hi, max(lo, x))
                        v = 0.0
                    pos[i][d], vel[i][d] = x, v
            costs = costs_of([list(p) for p in pos])
            improved = False
            for i in range(swarm):
                if costs[i] < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = list(pos[i]), costs[i]
                    if costs[i] < gbest_cost:
                        gbest, gbest_cost = list(pos[i]), costs[i]
                        improved = True
            stale = 0 if improved else stale + 1
            history.append(sign * gbest_cost)
            if callback is not None and callback(it, gbest, sign * gbest_cost):
                message = "Stopped by callback"
                break
            if patience is not None and stale >= patience:
                converged, message = True, f"No improvement for {patience} iterations"
                break
    finally:
        if pool is not None:
            pool.shutdown()

    return Result(
        x=gbest,
        score=sign * gbest_cost,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=it,
        converged=converged,
        message=message,
        method="particle_swarm",
        extra={"inertia": w},
    )
