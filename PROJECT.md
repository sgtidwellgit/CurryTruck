# CurryTruck — Project Document

> **Current version:** 2026.11.16 | **PyPI:** `pip install currytruck` | **Python:** ≥ 3.9 | **Dependencies:** none

**Tagline:** Finding the best solution, one ingredient at a time.

---

## What CurryTruck Is

CurryTruck is a general-purpose mathematical optimization library that helps
Python developers find optimal or near-optimal solutions to complex problems
involving constraints, resources, and competing objectives. It is pure
Python with no runtime dependencies.

See [README.md](README.md) for usage and [ARCHITECTURE.md](ARCHITECTURE.md)
for layout, dependency map, and conventions.

---

## Current Menu

| Module | Functionality | Public API |
|---|---|---|
| 🌶️ `spice` | Parameter optimization | `grid_search`, `random_search` |
| 🥘 `blend` | Combinatorial optimization | `knapsack`, `best_subset` |
| 🍲 `simmer` | Iterative optimization | `gradient_descent`, `numerical_gradient` |
| 🔥 `heat` | Global optimization | `simulated_annealing`, `genetic_algorithm`, `swap_neighbor`, `reverse_neighbor` |
| 📖 `recipe` | Constraint satisfaction | `solve_csp`, `Recipe`, `Constraint`, `all_different` |
| 👅 `taste` | Objective evaluation | `penalized`, `weighted_sum`, `minmax_normalize`, `counted`, `score_all`, benchmark functions |
| 🍽️ `portion` | Resource allocation | `linprog`, `allocate` |
| ⚖️ `balance` | Multi-objective optimization | `dominates`, `pareto_front`, `pareto_rank`, `crowding_distance`, `pareto_search` |
| 🚚 `serve` | Results and reporting | `rank`, `best`, `convergence`, `report`, `leaderboard` |

### Scope decisions (2026.11.16)

- **Zero dependencies.** numpy would speed up `linprog` and the population
  methods, but every algorithm here runs fine at the problem sizes a pure
  Python library is reasonable for, and an install with nothing to resolve
  is worth more than the speed. Revisit with an optional `[fast]` extra if
  real workloads demand it.
- **`linprog` is a dense tableau simplex** — correct (cross-checked against
  scipy/HiGHS on 3,000 random problems, including infeasible and unbounded
  ones), but intended for small-to-medium problems (hundreds of variables),
  not industrial LPs.
- **`linprog` is continuous only.** Integer programming is on the roadmap.
- **`recipe` checks n-ary constraints only once all their variables are
  assigned** (forward checking prunes the last free variable). Arc
  consistency (AC-3) is on the roadmap.
- **`best_subset` greedy fallback is a heuristic** — `result.converged` is
  `False` when it is used, so callers can tell.

---

## Tests

One test file per module, mirroring the source layout, plus
`test_package.py`. Every docstring example is also a doctest
(`addopts = "--doctest-modules"`, `testpaths = ["tests", "src"]`).

```bash
pip install -e ".[dev]"
pytest               # 193 tests as of 2026.11.16
```

CI (`.github/workflows/tests.yml`) runs on Python 3.9–3.13 on Ubuntu, plus
Windows and macOS on 3.9 and 3.13.

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

### Priority 1: Solver depth

- **Integer / mixed-integer programming** in `portion` — branch-and-bound on
  top of `linprog`
- **AC-3 arc consistency** in `recipe`, and a `max_nodes` search budget
- **Adam / RMSProp** optimizers in `simmer`, plus line search
- **Tabu search** and **particle swarm** in `heat`

### Priority 2: Problem templates

Ready-made helpers for the headline use cases:

- `serve`-ready **delivery routing** (TSP / VRP with time windows)
- **Shift scheduling** builder on top of `recipe` (availability, max hours,
  coverage requirements)
- **Portfolio allocation** with risk caps (mean-variance needs a QP solver —
  a design decision on whether to add one)

### Priority 3: Multi-objective search

- **NSGA-II** — a genetic algorithm in `balance` that evolves the whole
  Pareto front, using the existing `pareto_rank` and `crowding_distance`

### Priority 4: Ergonomics

- **Parallel evaluation** (`workers=`) for `spice` and `genetic_algorithm`
- **Callbacks / early stopping** hooks shared across iterative optimizers
- **Plotting** of convergence curves and Pareto fronts in `serve` — needs an
  optional `matplotlib` extra and a return-shape decision

---

*Last updated: 2026-11-16*
