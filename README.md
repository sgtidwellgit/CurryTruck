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
pip install currytruck[plot]   # + convergence and Pareto plots (matplotlib)
```

CurryTruck is a general-purpose optimization toolkit for Python. It helps you
find optimal or near-optimal solutions to problems involving constraints,
limited resources, and competing objectives — in **pure Python, with zero
required dependencies**.

---

## The Menu

| Module | Dish | Algorithms |
|---|---|---|
| 🌶️ `spice` | Parameter optimization | Grid search, random search (parallel) |
| 🥘 `blend` | Combinatorial optimization | Knapsack, subset selection, traveling salesman, vehicle routing |
| 🍲 `simmer` | Iterative optimization | Gradient descent (momentum, line search), Adam, RMSProp |
| 🔥 `heat` | Global optimization | Simulated annealing, genetic algorithms, particle swarm, tabu search |
| 📖 `recipe` | Constraint satisfaction | Backtracking + AC-3 + forward checking, shift scheduling |
| 👅 `taste` | Objective evaluation | Penalties, weighted scores, benchmark functions |
| 🍽️ `portion` | Resource allocation | Linear & mixed-integer programming, mean-variance portfolios |
| ⚖️ `balance` | Multi-objective optimization | Pareto fronts, non-dominated sorting, NSGA-II |
| 🚚 `serve` | Results and reporting | Rankings, convergence statistics, plots |

Every optimizer returns the same `Result` object — `x` (best solution),
`score`, `history`, `n_evals`, `converged`, `message` — so results from
different algorithms can be compared and reported the same way.

Import functions directly, or use the modules as namespaces:

```python
from currytruck import simulated_annealing, linprog
from currytruck import heat, portion   # heat.simulated_annealing, portion.linprog
```

### Shared controls

- **`maximize=True`** — everything minimizes by default.
- **`seed=`** — every stochastic method is reproducible.
- **`callback=`** — every iterative optimizer calls
  `callback(iteration, best_x, best_score)`; return `True` to stop early.
- **`workers=`** — `grid_search`, `random_search`, `genetic_algorithm`, and
  `particle_swarm` can evaluate in parallel (`backend="thread"` or
  `"process"`). A seeded run gives the same answer with any number of workers.

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
    workers=8,                       # train 8 models at a time
)
```

### 🥘 `blend` — Combinatorial Optimization

```python
from currytruck import knapsack, best_subset, shortest_route, vehicle_routes

# Which ingredients fit in a 50kg truck for the most value?
r = knapsack(values=[60, 100, 120], weights=[10, 20, 30], capacity=50)
r.x       # [1, 2]  (indices)
r.score   # 220

# Best 3-person team by combined skill, with at most one senior
r = best_subset(people, objective=team_skill, size=3,
                constraint=lambda team: sum(p.senior for p in team) <= 1)

# Visit every stop once and come home
r = shortest_route(points=[(0, 0), (2, 2), (0, 2), (2, 0)])
r.x, r.score   # [0, 3, 1, 2], 8.0

# Plan a fleet: capacity, delivery time windows, time spent at each stop
r = vehicle_routes(
    points=stops,                 # stops[0] is the depot
    demands=crates_per_stop,
    capacity=40,
    time_windows=windows,         # (earliest, latest) per stop, or None
    service_times=5,
)
r.x                    # [[3, 7, 1], [2, 5], ...] — one list per truck
r.extra["arrivals"]    # arrival time at each stop
```

`knapsack` is exact — dynamic programming for integer weights,
branch-and-bound for real-valued ones. `shortest_route` is exact up to 12
stops (Held-Karp) and uses nearest-neighbor + 2-opt beyond. `vehicle_routes`
uses Clarke-Wright savings with 2-opt clean-up.

### 🍲 `simmer` — Gradient-Based Optimization

*Low and slow.*

```python
from currytruck import gradient_descent, adam, rmsprop

r = gradient_descent(loss, x0=[0.0, 0.0], lr=0.05, momentum=0.9)
r = gradient_descent(loss, x0, lr=1.0, line_search=True)   # robust to a bad lr
r = gradient_descent(loss, x0, bounds=[(0, 1), (0, None)])  # projected onto a box
r = adam(loss, x0, lr=0.01)        # badly scaled problems
r = rmsprop(loss, x0, grad=my_gradient)
```

Without `grad=`, a central-difference numerical gradient is used.

### 🔥 `heat` — Global Optimization

*Turn up the heat to escape local optima.*

```python
from currytruck import (simulated_annealing, genetic_algorithm, particle_swarm,
                        tabu_search, reverse_neighbor)

# Continuous
r = simulated_annealing(f, x0=[4.0, 4.0], bounds=[(-5, 5), (-5, 5)], heat=4, seed=1)
r = particle_swarm(f, bounds=[(-5, 5)] * 10, swarm=40, seed=1)

# Delivery route — search over orderings
r = simulated_annealing(route_length, x0=list(range(n_stops)), neighbor=reverse_neighbor)
r = tabu_search(route_length, x0=list(range(n_stops)), neighbor=reverse_neighbor)

# Genetic algorithm: vector mode or permutation mode
r = genetic_algorithm(f, bounds=[(-5, 5)] * 10, population=80, generations=300)
r = genetic_algorithm(route_length, permutation=range(n_stops), patience=50)
```

### 🔥 The Heat Guide

Annealing, the genetic algorithm, and the particle swarm take a `heat` level
(1–5). Higher heat means more exploration.

| Heat | Annealing start temp | GA mutation rate | Swarm inertia | Vibe |
|---|---|---|---|---|
| 1 | × 0.1 | 1% | 0.4 | Mild. Barely leaves home. |
| 2 | × 0.5 | 5% | 0.6 | A little warmth. |
| 3 | × 1.0 | 10% | 0.73 | Medium. The default. |
| 4 | × 2.0 | 20% | 0.85 | Getting spicy. Wanders far. |
| 5 | × 5.0 | 40% | 0.95 | Napalm. Explores everything. |

### 📖 `recipe` — Constraint Satisfaction

```python
from currytruck import Recipe, schedule_shifts

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

# Or use the ready-made scheduler
r = schedule_shifts(
    employees=["ann", "bob", "cat", "dan"],
    shifts=["mon-am", "mon-pm", "tue-am", "tue-pm"],
    coverage=2,                                   # people per shift
    availability={"ann": ["mon-am", "tue-am"]},   # mornings only
    max_shifts=2,
    conflicts=[("mon-am", "mon-pm"), ("tue-am", "tue-pm")],   # no doubles
    preferences={"bob": {"mon-pm": 5}},
)
r.x   # {'mon-am': [...], 'mon-pm': [...], ...}
```

Global rules like `count_at_most` / `count_at_least` prune as soon as they
can't be met, and `max_nodes=` caps the search on big problems.

### 👅 `taste` — Objective Evaluation

```python
from currytruck import penalized, weighted_sum, counted
from currytruck.taste import rastrigin, rosenbrock   # benchmark functions

# Turn a constrained problem into an unconstrained one: g(x) <= 0
f = penalized(cost, [lambda x: x[0] + x[1] - 10], weight=1e4)

score = weighted_sum({"cost": 120, "delay": 3}, {"cost": 1.0, "delay": 25.0})
f = counted(expensive_objective)   # f.calls tracks evaluations
```

### 🍽️ `portion` — Linear, Integer & Portfolio Optimization

```python
from currytruck import linprog, milp, allocate, portfolio

# maximize 3x + 5y  st  x <= 4,  2y <= 12,  3x + 2y <= 18
r = linprog([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
r.x, r.score     # [2.0, 6.0], 36.0

# Same, but whole units only
r = milp([5, 4], A_ub=[[6, 4], [1, 2]], b_ub=[24, 6], maximize=True)
r.x, r.score     # [4, 0], 20.0

# Split a budget across options
r = allocate(100_000, returns={"ads": 1.4, "seo": 1.2, "events": 1.6},
             maximum={"events": 20_000}, minimum={"seo": 10_000})
r.x   # {'ads': 70000.0, 'seo': 10000.0, 'events': 20000.0}

# Invest: highest expected return with volatility capped at 10%
r = portfolio({"bonds": 0.03, "stocks": 0.08}, [[0.0025, 0.0], [0.0, 0.04]],
              max_volatility=0.10, max_weight={"stocks": 0.6})
r.x                    # {'bonds': 0.517..., 'stocks': 0.483...}
r.extra["volatility"]  # 0.1
```

`linprog`/`milp` support `A_eq`/`b_eq` equalities and per-variable `(lo, hi)`
bounds (`None` = unbounded), and raise `InfeasibleError` or
`UnboundedError` when appropriate. `milp(integers=[...])` picks which
variables must be whole numbers (default: all).

### ⚖️ `balance` — Multi-Objective Optimization

```python
from currytruck import pareto_search, nsga2

# Compare known candidates
front = pareto_search(
    [lambda p: p.risk, lambda p: p.expected_return],
    portfolios,
    maximize=[False, True],   # minimize risk, maximize return
)

# Or evolve the whole trade-off curve over a design space
front = nsga2([cost, weight], bounds=[(0, 1)] * 6, generations=200, seed=1)
for x, (c, w) in front:
    ...
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

# With currytruck[plot]
ax = serve.plot_convergence(runs, log=True)
ax.figure.savefig("convergence.png")
serve.plot_pareto(points, front=pareto_front(points))
```

---

## Real-World Uses

| Problem | Reach for |
|---|---|
| Employee scheduling | `recipe.schedule_shifts` |
| Delivery route optimization | `blend.shortest_route`, `blend.vehicle_routes` |
| Resource allocation across projects | `portion.allocate`, `portion.milp`, `blend.knapsack` |
| Investment allocation under risk limits | `portion.portfolio`, `balance` |
| Manufacturing waste / efficiency | `portion.milp`, `taste.penalized` + `heat` |
| ML hyperparameters, custom objectives | `spice`, `simmer.adam` |
| Game AI and tactics | `heat`, `blend.best_subset` |

---

## Exceptions

All exceptions inherit from `CurryTruckError`. `InvalidHeatLevel`,
`InfeasibleError`, `UnboundedError`, and `EmptySpaceError` are also
`ValueError`s, so existing `except ValueError` code keeps working.
`SearchBudgetExceeded` (a node budget ran out before any solution was found)
is a `RuntimeError` — the problem may well be solvable with a bigger budget.

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
