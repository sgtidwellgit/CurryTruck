# Changelog

All notable changes to CurryTruck are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.1.0] - 2026-11-16

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
