# CurryTruck — Architecture

## Project Type

Pure Python library. No framework, no server, no CLI, **no runtime
dependencies**. Distributed as a PyPI wheel. Consumers `import` functions or
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
│       ├── blend.py            # knapsack, subset selection
│       ├── simmer.py           # gradient descent
│       ├── heat.py             # simulated annealing, genetic algorithm
│       ├── recipe.py           # constraint satisfaction
│       ├── taste.py            # penalties, scoring, benchmark functions
│       ├── portion.py          # linear programming, allocation
│       ├── balance.py          # Pareto dominance and fronts
│       ├── serve.py            # rankings, convergence stats, text reports
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
spice       → stdlib (itertools, random), result, exceptions
blend       → stdlib (itertools, math), result, exceptions
simmer      → stdlib (math), result
heat        → stdlib (math, random), result, exceptions
recipe      → stdlib (dataclasses), result, exceptions
taste       → stdlib (functools, math)
portion     → result, exceptions
balance     → exceptions
serve       → result, exceptions
result      → stdlib only (leaf)
exceptions  → stdlib only (leaf)
```

`result` and `exceptions` are shared leaves — the "one return type, errors
live in one place" pattern — not a violation of module independence.

Small helpers (e.g. the `_better(a, b, maximize)` comparison) are duplicated
per module rather than shared, preserving that independence.

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
- **`heat` (1–5)** appears only in the `heat` module; higher is more
  exploratory. Out-of-range values raise `InvalidHeatLevel`.
- **Exceptions** each subclass both `CurryTruckError` and `ValueError`.

---

## Algorithms

| Function | Method | Exact? |
|---|---|---|
| `grid_search` | Cartesian product | Yes, over the given grid |
| `random_search` | Uniform sampling | No |
| `knapsack` | DP (int weights, ≤10M table cells) / branch-and-bound with LP-relaxation bound | Yes |
| `best_subset` | Exhaustive ≤ `max_evals`, else greedy forward selection | Exhaustive: yes; greedy: no |
| `gradient_descent` | First-order descent, classical momentum, box projection | Local optimum |
| `simulated_annealing` | Metropolis acceptance, geometric cooling to 1e-4·T0 | No |
| `genetic_algorithm` | Tournament selection, elitism, uniform / OX1 crossover | No |
| `solve_csp` | Backtracking, MRV, forward checking | Yes |
| `linprog` | Two-phase tableau simplex, Dantzig pivoting with Bland fallback | Yes |
| `pareto_front` / `pareto_rank` | Pairwise dominance, O(n²) per front | Yes |

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
