# CurryTruck

<p align="center">
  <a href="https://github.com/sgtidwellgit/CurryTruck/actions/workflows/tests.yml">
    <img src="https://github.com/sgtidwellgit/CurryTruck/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
</p>

**Finding the best solution, one ingredient at a time.**

You've got shifts to fill, routes to drive, a budget to split, and a model
with eleven knobs. Every choice pulls against another.

*CurryTruck.*

```python
pip install currytruck
```

CurryTruck is a general-purpose optimization toolkit for Python. It helps you
find optimal or near-optimal solutions to problems involving constraints,
limited resources, and competing objectives — in **pure Python, with zero
dependencies**.

---

## The Menu

| Module | Dish | Algorithms |
|---|---|---|
| 🌶️ `spice` | Parameter optimization | Grid search, random search |
| 🥘 `blend` | Combinatorial optimization | 0/1 knapsack, subset selection |
| 🍲 `simmer` | Iterative optimization | Gradient descent (with momentum, bounds) |
| 🔥 `heat` | Global optimization | Simulated annealing, genetic algorithms |
| 📖 `recipe` | Constraint satisfaction | Backtracking + forward checking |
| 👅 `taste` | Objective evaluation | Penalties, weighted scores, benchmark functions |
| 🍽️ `portion` | Resource allocation | Linear programming (two-phase simplex) |
| ⚖️ `balance` | Multi-objective optimization | Pareto fronts, non-dominated sorting |
| 🚚 `serve` | Results and reporting | Rankings, convergence statistics |

Every optimizer returns the same `Result` object — `x` (best solution),
`score`, `history`, `n_evals`, `converged`, `message` — so results from
different algorithms can be compared and reported the same way.

Import functions directly, or use the modules as namespaces:

```python
from currytruck import simulated_annealing, linprog
from currytruck import heat, portion   # heat.simulated_annealing, portion.linprog
```

---

### 🌶️ `spice` — Parameter Optimization

*Season to taste.* The objective is called with each parameter set as keyword
arguments.

```python
from currytruck import grid_search, random_search

r = grid_search(train_and_score, {"depth": [3, 5, 8], "lr": [0.01, 0.1]}, maximize=True)
r.x        # {'depth': 5, 'lr': 0.1}

r = random_search(
    train_and_score,
    {
        "depth": (2, 12),            # int range   -> uniform integer
        "lr": (0.001, 0.3),          # float range -> uniform float
        "kernel": ["rbf", "linear"], # list        -> random choice
    },
    n_iter=200, seed=42, maximize=True,
)
```

### 🥘 `blend` — Combinatorial Optimization

```python
from currytruck import knapsack, best_subset

# Which ingredients fit in a 50kg truck for the most value?
r = knapsack(values=[60, 100, 120], weights=[10, 20, 30], capacity=50)
r.x       # [1, 2]  (indices)
r.score   # 220

# Best 3-person team by combined skill, with at most one senior
r = best_subset(people, objective=team_skill, size=3,
                constraint=lambda team: sum(p.senior for p in team) <= 1)
```

`knapsack` is exact — dynamic programming for integer weights, branch-and-bound
for real-valued ones. `best_subset` is exhaustive up to `max_evals` candidate
subsets, then falls back to greedy forward selection.

### 🍲 `simmer` — Gradient Descent

*Low and slow.*

```python
from currytruck import gradient_descent

r = gradient_descent(loss, x0=[0.0, 0.0], lr=0.05, momentum=0.9)
r = gradient_descent(loss, x0, grad=my_gradient)         # analytic gradient
r = gradient_descent(loss, x0, bounds=[(0, 1), (0, None)])  # projected onto a box
```

Without `grad=`, a central-difference numerical gradient is used.

### 🔥 `heat` — Global Optimization

*Turn up the heat to escape local optima.*

```python
from currytruck import simulated_annealing, genetic_algorithm, reverse_neighbor

# Continuous
r = simulated_annealing(f, x0=[4.0, 4.0], bounds=[(-5, 5), (-5, 5)], heat=4, seed=1)

# Delivery route — search over orderings
r = simulated_annealing(route_length, x0=list(range(n_stops)), neighbor=reverse_neighbor)

# Genetic algorithm: vector mode or permutation mode
r = genetic_algorithm(f, bounds=[(-5, 5)] * 10, population=80, generations=300)
r = genetic_algorithm(route_length, permutation=range(n_stops), patience=50)
```

### 🔥 The Heat Guide

`heat` takes a `heat` level (1–5). Higher heat means more exploration.

| Heat | Annealing start temp | GA mutation rate | Vibe |
|---|---|---|---|
| 1 | × 0.1 | 1% | Mild. Barely leaves home. |
| 2 | × 0.5 | 5% | A little warmth. |
| 3 | × 1.0 | 10% | Medium. The default. |
| 4 | × 2.0 | 20% | Getting spicy. Wanders far. |
| 5 | × 5.0 | 40% | Napalm. Explores everything. |

### 📖 `recipe` — Constraint Satisfaction

```python
from currytruck import Recipe

shifts = ["morning", "evening", "night"]
r = (
    Recipe()
    .add_variable("ann", shifts)
    .add_variable("bob", shifts)
    .add_variable("cat", shifts)
    .all_different(["ann", "bob", "cat"])
    .add_constraint("ann", lambda s: s != "night")
)
r.solve().x
# Find the best schedule, not just a valid one:
r.solve(objective=lambda a: a["bob"] == "morning", maximize=True).x
```

Or functionally: `solve_csp(variables, constraints, objective=..., all_solutions=True)`.

### 👅 `taste` — Objective Evaluation

```python
from currytruck import penalized, weighted_sum, counted
from currytruck.taste import rastrigin, rosenbrock   # benchmark functions

# Turn a constrained problem into an unconstrained one: g(x) <= 0
f = penalized(cost, [lambda x: x[0] + x[1] - 10], weight=1e4)

score = weighted_sum({"cost": 120, "delay": 3}, {"cost": 1.0, "delay": 25.0})
f = counted(expensive_objective)   # f.calls tracks evaluations
```

### 🍽️ `portion` — Linear Programming

```python
from currytruck import linprog, allocate

# maximize 3x + 5y  st  x <= 4,  2y <= 12,  3x + 2y <= 18
r = linprog([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
r.x, r.score     # [2.0, 6.0], 36.0

# Split a budget across options
r = allocate(100_000, returns={"ads": 1.4, "seo": 1.2, "events": 1.6},
             maximum={"events": 20_000}, minimum={"seo": 10_000})
r.x   # {'ads': 70000.0, 'seo': 10000.0, 'events': 20000.0}
```

Supports `A_eq`/`b_eq` equalities and per-variable `(lo, hi)` bounds (`None` =
unbounded). Raises `InfeasibleError` or `UnboundedError` when appropriate.

### ⚖️ `balance` — Multi-Objective Optimization

```python
from currytruck import pareto_search

front = pareto_search(
    [lambda p: p.risk, lambda p: p.expected_return],
    portfolios,
    maximize=[False, True],   # minimize risk, maximize return
)
```

Also: `dominates`, `pareto_front`, `pareto_rank` (non-dominated sorting),
`crowding_distance`.

### 🚚 `serve` — Results and Reporting

```python
from currytruck import serve

runs = [simulated_annealing(f, x0, seed=s) for s in range(5)]
print(serve.leaderboard(runs))
print(serve.report(serve.best(runs)))
serve.convergence(runs[0])   # initial/final, improvement, iteration_of_best, plateau
```

---

## Real-World Uses

| Problem | Reach for |
|---|---|
| Employee scheduling | `recipe` |
| Delivery route optimization | `heat` with `reverse_neighbor` or `permutation=` |
| Resource allocation across projects | `portion.allocate`, `blend.knapsack` |
| Investment allocation under risk limits | `portion.linprog`, `balance` |
| Manufacturing waste / efficiency | `portion.linprog`, `taste.penalized` + `heat` |
| ML hyperparameters, custom objectives | `spice`, `simmer` |
| Game AI and tactics | `heat`, `blend.best_subset` |

---

## Exceptions

All exceptions inherit from `CurryTruckError` and from `ValueError`, so
existing `except ValueError` code keeps working:
`InvalidHeatLevel`, `InfeasibleError`, `UnboundedError`, `EmptySpaceError`.

---

## The Food Truck Fleet

| Package | Focus |
|---|---|
| [thaitruck](https://pypi.org/project/thaitruck/) | Batch DataFrame cleaning, merging, profiling, caching |
| [sushitruck](https://pypi.org/project/sushitruck/) | Streaming ingestion and connectors |
| [ramentruck](https://pypi.org/project/ramentruck/) | ML/AI toolkit |
| **currytruck** | Mathematical optimization |

---

## License

MIT © [Babble Baz](https://www.babblebaz.com)
