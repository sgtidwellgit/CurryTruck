# CurryTruck — Architecture

## Project Type

Pure Python library. No framework, no server, no CLI, **no required runtime
dependencies** (matplotlib is an optional extra for plotting). Distributed as a PyPI wheel. Consumers `import` functions or
modules directly.

---

## Repository Layout

```
CurryTruck/
├── pyproject.toml              # build config, metadata, pytest config
├── README.md
├── ARCHITECTURE.md
├── PROJECT.md
├── CHANGELOG.md
├── LICENSE                     # MIT
├── .github/workflows/tests.yml # CI
├── src/
│   └── currytruck/
│       ├── __init__.py         # public re-exports + __version__
│       ├── py.typed            # PEP 561 marker
│       ├── spice.py            # grid search, random search
│       ├── blend.py            # knapsack, subset selection, TSP, vehicle routing
│       ├── simmer.py           # gradient descent, Adam, RMSProp
│       ├── heat.py             # annealing, genetic algorithm, particle swarm, tabu
│       ├── recipe.py           # constraint satisfaction, shift scheduling
│       ├── taste.py            # penalties, scoring, benchmark functions
│       ├── portion.py          # LP, MILP, allocation, portfolios
│       ├── balance.py          # Pareto dominance, fronts, NSGA-II
│       ├── serve.py            # rankings, convergence stats, reports, plots
│       ├── result.py           # Result dataclass (shared leaf module)
│       └── exceptions.py       # CurryTruckError hierarchy (shared leaf module)
└── tests/
    ├── test_spice.py
    ├── test_blend.py
    ├── test_simmer.py
    ├── test_heat.py
    ├── test_recipe.py
    ├── test_taste.py
    ├── test_portion.py
    ├── test_balance.py
    ├── test_serve.py
    └── test_package.py         # exports, version, exception hierarchy
```

---

## Module Dependency Map

The nine menu modules are **independent of each other** — none imports
another menu module. Each depends only on the standard library and the two
shared leaf modules.

```
spice       → stdlib (concurrent.futures, functools, itertools, random), result, exceptions
blend       → stdlib (itertools, math), result, exceptions
simmer      → stdlib (math), result
heat        → stdlib (collections, concurrent.futures, math, random), result, exceptions
recipe      → stdlib (collections, dataclasses), result, exceptions
taste       → stdlib (functools, math)
portion     → stdlib (math), result, exceptions
balance     → stdlib (random), exceptions
serve       → result, exceptions [+ matplotlib, imported lazily, for plot_* only]
result      → stdlib only (leaf)
exceptions  → stdlib only (leaf)
```

`result` and `exceptions` are shared leaves — the "one return type, errors
live in one place" pattern — not a violation of module independence.

Small helpers (e.g. the `_better(a, b, maximize)` comparison, and the
thread/process pool set-up behind `workers=`) are duplicated per module
rather than shared, preserving that independence. Likewise the shift
scheduler lives in `recipe` and solves with `recipe`'s own search, and
`milp` branches on `portion`'s own `linprog`.

---

## Shared Conventions

- **Every optimizer returns a `Result`** with `x`, `score`, `maximize`,
  `history`, `n_evals`, `n_iter`, `converged`, `message`, `method`, `extra`.
- **Minimize by default.** Pass `maximize=True` to flip. (`knapsack` and
  `best_subset` are naturally maximizing and default accordingly.)
- **`history` is best-so-far** for stochastic and enumerative searches, and
  the iterate's score trajectory for `gradient_descent`.
- **Reproducibility** — stochastic functions take `seed=` and use a private
  `random.Random`, never the global RNG.
- **Inputs are never mutated.**
- **`heat` (1–5)** appears only in the `heat` module (annealing, genetic
  algorithm, particle swarm); higher is more exploratory. Out-of-range
  values raise `InvalidHeatLevel`.
- **`callback(iteration, best_x, best_score)`** on every iterative optimizer;
  returning `True` stops the run (`converged=False`, message
  `"Stopped by callback"`). `nsga2` passes `(generation, front)` instead.
- **`workers=` / `backend=`** — parallel evaluation where the work comes in
  batches (`spice`, `genetic_algorithm`, `particle_swarm`). All random
  draws happen in the calling thread, so results don't depend on `workers`.
  `backend="process"` needs a picklable (module-level) objective.
- **Budgets** — exact searches (`solve_csp`, `milp`) take `max_nodes`. Out of
  budget with a solution in hand: return it with `converged=False`. Without
  one: raise `SearchBudgetExceeded`.
- **Exceptions** each subclass `CurryTruckError` and a built-in:
  `ValueError` for bad or unsolvable input, `RuntimeError` for
  `SearchBudgetExceeded`.

---

## Algorithms

| Function | Method | Exact? |
|---|---|---|
| `grid_search` | Cartesian product | Yes, over the given grid |
| `random_search` | Uniform sampling | No |
| `knapsack` | DP (int weights, ≤10M table cells) / branch-and-bound with LP-relaxation bound | Yes |
| `best_subset` | Exhaustive ≤ `max_evals`, else greedy forward selection | Exhaustive: yes; greedy: no |
| `shortest_route` | Held-Karp DP (≤ `exact_limit` stops), else nearest neighbor + 2-opt | Exact: yes; 2-opt: no |
| `vehicle_routes` | Clarke-Wright savings (capacity + time-window checked), per-route 2-opt | No |
| `gradient_descent` | First-order descent, classical momentum or Armijo line search, box projection | Local optimum |
| `adam` / `rmsprop` | Adaptive per-coordinate step sizes | Local optimum |
| `simulated_annealing` | Metropolis acceptance, geometric cooling to 1e-4·T0 | No |
| `genetic_algorithm` | Tournament selection, elitism, uniform / OX1 crossover | No |
| `particle_swarm` | Inertia-weight PSO, velocity clamping, wall-stopping | No |
| `tabu_search` | Sampled neighborhood, tabu list with aspiration | No |
| `solve_csp` | AC-3, backtracking, MRV, forward checking, partial global checks | Yes (within `max_nodes`) |
| `schedule_shifts` | `solve_csp` model; preference-ordered values, symmetry breaking | Yes (within `max_nodes`) |
| `linprog` | Two-phase tableau simplex, Dantzig pivoting with Bland fallback | Yes |
| `milp` | Depth-first branch-and-bound on `linprog`, most-fractional branching | Yes (within `max_nodes`) |
| `portfolio` | FISTA projected gradient on the capped simplex; volatility cap by bisection on risk aversion | Yes (convex) |
| `pareto_front` / `pareto_rank` | Pairwise dominance, O(n²) per front | Yes |
| `nsga2` | Non-dominated sorting, crowding distance, SBX + polynomial mutation | No |

---

## Build System

| Tool | Role |
|---|---|
| `hatchling` | Build backend (PEP 517) |
| `python -m build` | Produces `.tar.gz` + `.whl` in `dist/` |
| `twine upload dist/*` | Publishes to PyPI |

The sdist uses an explicit include allowlist. Version is declared in
`pyproject.toml` and mirrored in `src/currytruck/__init__.py`;
`tests/test_package.py` fails if they drift.
