"""CurryTruck — finding the best solution, one ingredient at a time."""

from currytruck import balance, blend, heat, portion, recipe, serve, simmer, spice, taste
from currytruck.result import Result
from currytruck.spice import grid_search, random_search
from currytruck.blend import knapsack, best_subset
from currytruck.simmer import gradient_descent, numerical_gradient
from currytruck.heat import simulated_annealing, genetic_algorithm, swap_neighbor, reverse_neighbor
from currytruck.recipe import Recipe, Constraint, solve_csp, all_different
from currytruck.taste import weighted_sum, minmax_normalize, penalized, counted, score_all
from currytruck.portion import linprog, allocate
from currytruck.balance import dominates, pareto_front, pareto_rank, crowding_distance, pareto_search
from currytruck.serve import rank, best, convergence, report, leaderboard
from currytruck.exceptions import (
    CurryTruckError,
    InvalidHeatLevel,
    InfeasibleError,
    UnboundedError,
    EmptySpaceError,
)

__version__ = "2026.11.16"

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
    # simmer
    "gradient_descent",
    "numerical_gradient",
    # heat
    "simulated_annealing",
    "genetic_algorithm",
    "swap_neighbor",
    "reverse_neighbor",
    # recipe
    "Recipe",
    "Constraint",
    "solve_csp",
    "all_different",
    # taste
    "weighted_sum",
    "minmax_normalize",
    "penalized",
    "counted",
    "score_all",
    # portion
    "linprog",
    "allocate",
    # balance
    "dominates",
    "pareto_front",
    "pareto_rank",
    "crowding_distance",
    "pareto_search",
    # serve
    "rank",
    "best",
    "convergence",
    "report",
    "leaderboard",
    # exceptions
    "CurryTruckError",
    "InvalidHeatLevel",
    "InfeasibleError",
    "UnboundedError",
    "EmptySpaceError",
]
