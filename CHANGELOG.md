# Changelog

All notable changes to CurryTruck are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [2026.11.18]

The full roadmap from the first release.

### Added

- `portion.milp` — mixed-integer linear programming by branch-and-bound on
  `linprog`, with a `max_nodes` budget
- `portion.portfolio` — mean-variance allocation (accelerated projected
  gradient) with per-asset weight caps and an optional `max_volatility` cap
- `recipe` — AC-3 arc-consistency preprocessing (`ac3=True` by default), a
  `max_nodes` search budget, and `Constraint(partial=...)` checks that prune
  global constraints before all their variables are assigned
- `recipe.count_at_most` / `count_at_least` global constraints, and matching
  `Recipe.at_most()` / `Recipe.at_least()` builder methods
- `recipe.schedule_shifts` — employee scheduling template: coverage,
  availability, min/max shifts, conflicting shifts, preference maximization
- `simmer.adam` and `simmer.rmsprop`; `gradient_descent(line_search=True)`
  backtracking (Armijo) line search
- `heat.tabu_search` and `heat.particle_swarm` (`heat=` sets inertia)
- `blend.shortest_route` — traveling salesman: exact Held-Karp up to 12
  stops, nearest-neighbor + 2-opt beyond
- `blend.vehicle_routes` — capacitated vehicle routing with time windows and
  service times (Clarke-Wright savings + 2-opt)
- `balance.nsga2` — NSGA-II multi-objective genetic algorithm
- `serve.plot_convergence` and `serve.plot_pareto` — need the new optional
  `plot` extra (`pip install currytruck[plot]`)
- `callback=` early-stopping hook on every iterative optimizer
  (`callback(iteration, best_x, best_score)`; return `True` to stop)
- `workers=` / `backend=` parallel evaluation for `grid_search`,
  `random_search`, `genetic_algorithm`, and `particle_swarm`; seeded runs
  give identical results with or without workers
- `SearchBudgetExceeded` exception (a `CurryTruckError` and `RuntimeError`)
- A bare `pytest` now runs from a fresh checkout without installing the
  package

### Changed

- Versions are now the release date (`YYYY.MM.DD`)

## [2026.11.16]

First release — the full planned menu.

### Added

- `spice` — `grid_search`, `random_search` (int/float ranges, choice lists,
  custom sampler callables)
- `blend` — exact 0/1 `knapsack` (dynamic programming for integer weights,
  branch-and-bound otherwise); `best_subset` (exhaustive up to `max_evals`,
  then greedy forward selection)
- `simmer` — `gradient_descent` with momentum, box bounds (projected descent),
  analytic or central-difference `numerical_gradient`; divergence is reported
  in the result rather than raised
- `heat` — `simulated_annealing` (auto-calibrated starting temperature,
  pluggable neighbor moves: `swap_neighbor`, `reverse_neighbor`) and
  `genetic_algorithm` (vector mode with optional integer genes, permutation
  mode with order crossover); both take `heat=1..5`
- `recipe` — `solve_csp` backtracking search with MRV ordering and forward
  checking; optional objective for constraint optimization; `Recipe` builder,
  `Constraint`, `all_different`
- `taste` — `penalized`, `weighted_sum`, `minmax_normalize`, `counted`,
  `score_all`; benchmark functions `sphere`, `rosenbrock`, `rastrigin`,
  `ackley`
- `portion` — pure Python two-phase simplex `linprog` (inequalities,
  equalities, arbitrary per-variable bounds, Bland's rule fallback against
  cycling); `allocate` budget-splitting helper
- `balance` — `dominates`, `pareto_front`, `pareto_rank`, `crowding_distance`,
  `pareto_search`
- `serve` — `rank`, `best`, `convergence`, `report`, `leaderboard`
- Shared `Result` dataclass returned by every optimizer
- `CurryTruckError` hierarchy: `InvalidHeatLevel`, `InfeasibleError`,
  `UnboundedError`, `EmptySpaceError` (each also a `ValueError`)
- `py.typed` marker (PEP 561)
- CI: GitHub Actions running pytest (plus doctests) across Python 3.9–3.13,
  with Windows and macOS legs
