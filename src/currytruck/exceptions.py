"""exceptions — package-specific exception types for cleaner downstream handling."""


class CurryTruckError(Exception):
    """Base class for all CurryTruck exceptions."""


class InvalidHeatLevel(CurryTruckError, ValueError):
    """Raised when a heat parameter is outside its valid range (1-5)."""


class InfeasibleError(CurryTruckError, ValueError):
    """Raised when a problem has no solution that satisfies its constraints."""


class UnboundedError(CurryTruckError, ValueError):
    """Raised by portion.linprog when the objective can improve without limit."""


class EmptySpaceError(CurryTruckError, ValueError):
    """Raised when a search space or candidate set has nothing to evaluate."""


class SearchBudgetExceeded(CurryTruckError, RuntimeError):
    """Raised when a search hits its node budget before finding any solution.

    Unlike the other exceptions this is a ``RuntimeError``, not a
    ``ValueError`` — the problem may well be solvable with a larger budget.
    """
