"""CurryTruck — finding the best solution, one ingredient at a time."""

from currytruck import balance, blend, heat, portion, recipe, serve, simmer, spice, taste
from currytruck.result import Result
from currytruck.spice import grid_search, random_search
from currytruck.blend import knapsack, best_subset, shortest_route, vehicle_routes
from currytruck.simmer import gradient_descent, adam, rmsprop, numerical_gradient
from currytruck.heat import (
    simulated_annealing,
    genetic_algorithm,
    particle_swarm,
    tabu_search,
    swap_neighbor,
    reverse_neighbor,
)
from currytruck.recipe import (
    Recipe,
    Constraint,
    solve_csp,
    all_different,
    count_at_most,
    count_at_least,
    schedule_shifts,
)
from currytruck.taste import weighted_sum, minmax_normalize, penalized, counted, score_all
from currytruck.portion import linprog, milp, allocate, portfolio
from currytruck.balance import (
    dominates,
    pareto_front,
    pareto_rank,
    crowding_distance,
    pareto_search,
    nsga2,
)
from currytruck.serve import (
    rank,
    best,
    convergence,
    report,
    leaderboard,
    plot_convergence,
    plot_pareto,
)
from currytruck.exceptions import (
    CurryTruckError,
    InvalidHeatLevel,
    InfeasibleError,
    UnboundedError,
    EmptySpaceError,
    SearchBudgetExceeded,
)

__version__ = "2026.11.18"

__all__ = [
    # modules
    "spice",
    "blend",
    "simmer",
    "heat",
    "recipe",
    "taste",
    "portion",
    "balance",
    "serve",
    # result container
    "Result",
    # spice
    "grid_search",
    "random_search",
    # blend
    "knapsack",
    "best_subset",
    "shortest_route",
    "vehicle_routes",
    # simmer
    "gradient_descent",
    "adam",
    "rmsprop",
    "numerical_gradient",
    # heat
    "simulated_annealing",
    "genetic_algorithm",
    "particle_swarm",
    "tabu_search",
    "swap_neighbor",
    "reverse_neighbor",
    # recipe
    "Recipe",
    "Constraint",
    "solve_csp",
    "all_different",
    "count_at_most",
    "count_at_least",
    "schedule_shifts",
    # taste
    "weighted_sum",
    "minmax_normalize",
    "penalized",
    "counted",
    "score_all",
    # portion
    "linprog",
    "milp",
    "allocate",
    "portfolio",
    # balance
    "dominates",
    "pareto_front",
    "pareto_rank",
    "crowding_distance",
    "pareto_search",
    "nsga2",
    # serve
    "rank",
    "best",
    "convergence",
    "report",
    "leaderboard",
    "plot_convergence",
    "plot_pareto",
    # exceptions
    "CurryTruckError",
    "InvalidHeatLevel",
    "InfeasibleError",
    "UnboundedError",
    "EmptySpaceError",
    "SearchBudgetExceeded",
]
