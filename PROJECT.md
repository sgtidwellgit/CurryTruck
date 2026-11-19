# CurryTruck — Project Document

> **Current version:** 2026.11.18 | **PyPI:** `pip install currytruck` | **Python:** ≥ 3.9 | **Dependencies:** none required (`[plot]` adds matplotlib)

**Tagline:** Finding the best solution, one ingredient at a time.

---

## What CurryTruck Is

CurryTruck is a general-purpose mathematical optimization library that helps
Python developers find optimal or near-optimal solutions to complex problems
involving constraints, resources, and competing objectives. It is pure
Python with no required runtime dependencies.

See [README.md](README.md) for usage and [ARCHITECTURE.md](ARCHITECTURE.md)
for layout, dependency map, and conventions.

---

## Current Menu

| Module | Functionality | Public API |
|---|---|---|
| 🌶️ `spice` | Parameter optimization | `grid_search`, `random_search` |
| 🥘 `blend` | Combinatorial optimization | `knapsack`, `best_subset`, `shortest_route`, `vehicle_routes` |
| 🍲 `simmer` | Iterative optimization | `gradient_descent`, `adam`, `rmsprop`, `numerical_gradient` |
| 🔥 `heat` | Global optimization | `simulated_annealing`, `genetic_algorithm`, `particle_swarm`, `tabu_search`, `swap_neighbor`, `reverse_neighbor` |
| 📖 `recipe` | Constraint satisfaction | `solve_csp`, `Recipe`, `Constraint`, `all_different`, `count_at_most`, `count_at_least`, `schedule_shifts` |
| 👅 `taste` | Objective evaluation | `penalized`, `weighted_sum`, `minmax_normalize`, `counted`, `score_all`, benchmark functions |
| 🍽️ `portion` | Resource allocation | `linprog`, `milp`, `allocate`, `portfolio` |
| ⚖️ `balance` | Multi-objective optimization | `dominates`, `pareto_front`, `pareto_rank`, `crowding_distance`, `pareto_search`, `nsga2` |
| 🚚 `serve` | Results and reporting | `rank`, `best`, `convergence`, `report`, `leaderboard`, `plot_convergence`, `plot_pareto` |

Cross-cutting: `callback=` on every iterative optimizer; `workers=` /
`backend=` on `grid_search`, `random_search`, `genetic_algorithm`,
`particle_swarm`; `max_nodes=` on `solve_csp`, `schedule_shifts`, `milp`.

### Scope decisions

**2026.11.16**

- **Zero required dependencies.** numpy would speed up `linprog` and the
  population methods, but every algorithm here runs fine at the problem
  sizes a pure Python library is reasonable for, and an install with
  nothing to resolve is worth more than the speed. Revisit with an optional
  `[fast]` extra if real workloads demand it.
- **`linprog` is a dense tableau simplex** — correct (cross-checked against
  scipy/HiGHS on 3,000 random problems, including infeasible and unbounded
  ones), but intended for small-to-medium problems (hundreds of variables),
  not industrial LPs.
- **`best_subset` greedy fallback is a heuristic** — `result.converged` is
  `False` when it is used, so callers can tell.

**2026.11.18**

- **Templates live in the module whose solver they use** — routing in
  `blend`, scheduling in `recipe`, portfolios in `portion` — rather than in
  a new templates module, so no menu module imports another.
- **`milp` is depth-first branch-and-bound on `linprog`.** Verified against
  scipy's HiGHS MILP on 600 random mixed-integer problems; it agreed on all
  of them once HiGHS presolve was disabled — with presolve on, scipy 1.15
  returned a suboptimal "optimal" answer on 3. Fine for tens of integer
  variables; no cutting planes or presolve of its own.
- **`portfolio` caps volatility by bisecting on risk aversion** rather than
  adding a general QP solver — volatility falls monotonically as risk
  aversion rises, so this finds the exact capped optimum with the same
  projected-gradient solver. Verified against scipy SLSQP on 230 random
  problems (never worse). A 30-asset capped solve takes about 3 seconds.
- **`vehicle_routes` is a heuristic** (Clarke-Wright + 2-opt). Its
  `n_vehicles` verdict is heuristic too — the error message says so.
- **`schedule_shifts` with `preferences` searches within `max_nodes`** and
  returns the best schedule found; `converged` tells you whether it proved
  optimality.
- **`nsga2` returns the front, not a `Result`** — matching `pareto_search`.
  A single `score` doesn't describe a trade-off curve.
- **Plotting returns the Axes and never calls `plt.show()`** — callers decide
  whether to show, save, or compose figures. matplotlib is imported only
  when a `plot_*` function runs, so the core install stays dependency-free.
- **`workers=` only where evaluation is batched.** Annealing and tabu search
  are inherently sequential (each step depends on the last); gradient
  methods' cost is in the gradient, which a user-supplied `grad=` can
  parallelize itself.

---

## Tests

One test file per module, mirroring the source layout, plus
`test_package.py`. Every docstring example is also a doctest
(`addopts = "--doctest-modules"`, `testpaths = ["tests", "src"]`,
`pythonpath = ["src"]` so a bare `pytest` works from a fresh checkout).

```bash
pip install -e ".[dev,plot]"
pytest               # 335 tests as of 2026.11.18
```

The plotting tests skip themselves if matplotlib isn't installed.

CI (`.github/workflows/tests.yml`) runs on Python 3.9–3.13 on Ubuntu
(3.9 on ubuntu-24.04, since ubuntu-latest has no 3.9 build), plus Windows
and macOS on 3.9 and 3.13.

---

## Build & Publish

```bash
python -m build            # produces dist/*.whl and dist/*.tar.gz
twine check dist/*         # verify package metadata
twine upload dist/*        # publish to PyPI
```

Versions are the release date, `YYYY.MM.DD` (e.g. `2026.11.16`). A second
release on the same day appends a counter: `2026.11.16.1`. Don't zero-pad
the month or day — PyPI normalizes `2026.01.05` to `2026.1.5` anyway.

Bump the version in exactly two places before each release:
- `pyproject.toml` → `version = "YYYY.MM.DD"`
- `src/currytruck/__init__.py` → `__version__ = "YYYY.MM.DD"`

Move the `[Unreleased]` section of `CHANGELOG.md` under a new
`[YYYY.MM.DD]` heading as part of the same release.

The first upload went out as `0.1.0` before the date scheme was adopted;
date versions sort above it, so `pip install currytruck` picks them up.

---

## The Food Truck Fleet

| Package | Focus |
|---|---|
| **thaitruck** | Batch DataFrame cleaning, merging, profiling, caching |
| **sushitruck** | Streaming ingestion, REST API connectors, file reading |
| **ramentruck** | ML/AI toolkit — training, tuning, explainability |
| **currytruck** | Mathematical optimization |

Each package is fully independent — none imports from another.

---

## Roadmap

The original roadmap (integer programming, AC-3, Adam/RMSProp, line search,
tabu search, particle swarm, routing / scheduling / portfolio templates,
NSGA-II, parallel evaluation, callbacks, plotting) shipped in 2026.11.18.
Next candidates:

### Priority 1: Solver depth

- **`milp` cutting planes and best-bound node selection** — fewer nodes on
  harder integer programs
- **Revised / sparse simplex** for `linprog` — larger LPs
- **Or-opt and inter-route moves** (relocate, swap between trucks) for
  `vehicle_routes`
- **Maintaining arc consistency (MAC)** during `recipe` search, not just
  before it

### Priority 2: Problem templates

- **Bin packing / cutting stock** (manufacturing waste)
- **Job-shop scheduling** (machines, task precedence, makespan)
- **Assignment problem** (Hungarian algorithm — workers to tasks, 1:1)

### Priority 3: Ergonomics

- **`[fast]` optional numpy extra** for `linprog`, `portfolio`, and the
  population methods
- **Checkpoint / resume** for long `genetic_algorithm` and `nsga2` runs
- **Hypervolume indicator** in `balance` for comparing fronts

---

*Last updated: 2026-11-18*
