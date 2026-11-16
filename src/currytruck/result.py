"""result — the shared Result container every optimizer returns (leaf module)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Result:
    """Outcome of a single optimization run.

    Attributes
    ----------
    x:
        The best solution found. Its shape depends on the optimizer — a dict of
        parameters for ``spice``, a list of floats for ``simmer``/``heat``, a
        list of selected indices for ``blend``, a variable assignment for
        ``recipe``.
    score:
        Objective value at ``x``.
    maximize:
        Whether higher ``score`` is better. Used by ``serve`` to rank results.
    history:
        Score after each iteration / evaluation, for convergence reporting —
        best-so-far for searches, the current iterate's score for
        ``simmer.gradient_descent``.
    n_evals:
        Number of objective evaluations performed.
    n_iter:
        Number of iterations (generations, steps, ...) performed.
    converged:
        ``True`` if the run stopped because a convergence criterion was met
        (or the search was exhaustive), ``False`` if it hit an iteration cap.
    message:
        Human-readable stop reason.
    method:
        Name of the algorithm that produced this result.
    extra:
        Algorithm-specific details (e.g. all evaluated trials, slack values).
    """

    x: Any
    score: float
    maximize: bool = False
    history: list[float] = field(default_factory=list)
    n_evals: int = 0
    n_iter: int = 0
    converged: bool = True
    message: str = ""
    method: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def better_than(self, other: Result) -> bool:
        """Return ``True`` if this result's score beats ``other``'s."""
        if self.maximize:
            return self.score > other.score
        return self.score < other.score
