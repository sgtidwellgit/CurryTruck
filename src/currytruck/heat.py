"""heat — global optimization: simulated annealing and genetic algorithms.

Turn up the heat to escape local optima. Both algorithms take a ``heat``
level (1-5): higher heat means more exploration — a hotter starting
temperature for annealing, a higher mutation rate for the genetic algorithm.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional, Sequence

from currytruck.exceptions import InvalidHeatLevel
from currytruck.result import Result

# heat -> multiplier on the estimated starting temperature
_SA_TEMP = {1: 0.1, 2: 0.5, 3: 1.0, 4: 2.0, 5: 5.0}
# heat -> per-gene mutation probability
_GA_MUTATION = {1: 0.01, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.4}


def _check_heat(heat: int) -> None:
    if heat not in _SA_TEMP:
        raise InvalidHeatLevel(f"heat must be an integer 1-5, got {heat!r}")


def _better(a: float, b: float, maximize: bool) -> bool:
    return a > b if maximize else a < b


def swap_neighbor(x: Sequence[Any], rng: random.Random) -> list[Any]:
    """Return a copy of ``x`` with two random positions swapped.

    Pass as ``neighbor=`` to :func:`simulated_annealing` to search over
    orderings (delivery routes, task sequences, ...).
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
    """
    _check_heat(heat)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    rng = random.Random(seed)
    if neighbor is None:
        if bounds is not None and len(bounds) != len(x0):
            raise ValueError("bounds must have one (lo, hi) pair per dimension")
        neighbor = _gaussian_neighbor(bounds, step)

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
    for _ in range(max_iter):
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

    return Result(
        x=best,
        score=sign * best_cost,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=max_iter,
        converged=False,
        message=f"Completed {max_iter} annealing steps",
        method="simulated_annealing",
        extra={"acceptance_rate": accepted / max_iter, "start_temperature": t0},
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
        Seed for reproducible runs.
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
    evals = 0

    def cost(ind: list[Any]) -> float:
        nonlocal evals
        evals += 1
        return sign * float(objective(ind))

    pop = [spawn() for _ in range(population)]
    costs = [cost(ind) for ind in pop]

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
        next_pop = [pop[i] for i in ranked[:elite]]
        next_costs = [costs[i] for i in ranked[:elite]]
        while len(next_pop) < population:
            a, b = select(), select()
            child = crossover(a, b) if rng.random() < crossover_rate else list(a)
            child = mutate(child)
            next_pop.append(child)
            next_costs.append(cost(child))
        pop, costs = next_pop, next_costs

        gen_best = min(range(population), key=lambda i: costs[i])
        if costs[gen_best] < best_cost:
            best, best_cost = pop[gen_best], costs[gen_best]
            stale = 0
        else:
            stale += 1
        history.append(sign * best_cost)
        if patience is not None and stale >= patience:
            converged = True
            message = f"No improvement for {patience} generations"
            break

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
