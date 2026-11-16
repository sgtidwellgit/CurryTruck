"""recipe — constraint satisfaction by backtracking search.

Every recipe has rules: this goes before that, these two never share a pan.
Declare variables with their possible values and the rules between them;
``recipe`` finds assignments that follow every rule — and, given an
objective, the best one.

Search is depth-first backtracking with minimum-remaining-values variable
ordering and forward checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from currytruck.exceptions import InfeasibleError
from currytruck.result import Result


@dataclass(frozen=True)
class Constraint:
    """A rule over one or more variables.

    ``predicate`` receives the variables' values positionally, in the order
    given by ``variables``, and returns ``True`` if the combination is allowed.
    """

    variables: tuple[str, ...]
    predicate: Callable[..., bool]
    name: str = ""

    def check(self, assignment: Mapping[str, Any]) -> bool:
        return bool(self.predicate(*(assignment[v] for v in self.variables)))


ConstraintLike = Union[Constraint, tuple[Sequence[str], Callable[..., bool]]]


def all_different(variables: Sequence[str]) -> list[Constraint]:
    """Pairwise ``!=`` constraints — no two of ``variables`` share a value."""
    vs = list(variables)
    return [
        Constraint((vs[i], vs[j]), lambda a, b: a != b, name=f"all_different({vs[i]},{vs[j]})")
        for i in range(len(vs))
        for j in range(i + 1, len(vs))
    ]


def _as_constraint(c: ConstraintLike) -> Constraint:
    if isinstance(c, Constraint):
        return c
    variables, predicate = c
    if isinstance(variables, str):
        variables = (variables,)
    return Constraint(tuple(variables), predicate)


def _search(
    domains: dict[str, list[Any]],
    constraints: list[Constraint],
    counter: list[int],
) -> Iterator[dict[str, Any]]:
    by_var: dict[str, list[Constraint]] = {v: [] for v in domains}
    for c in constraints:
        for v in set(c.variables):
            by_var[v].append(c)

    assignment: dict[str, Any] = {}

    def consistent(var: str) -> bool:
        return all(
            c.check(assignment)
            for c in by_var[var]
            if all(v in assignment for v in c.variables)
        )

    def forward_check(var: str, doms: dict[str, list[Any]]) -> Optional[dict[str, list[Any]]]:
        new = dict(doms)
        for c in by_var[var]:
            free = {v for v in c.variables if v not in assignment}
            if len(free) != 1:
                continue
            (u,) = free
            kept = []
            for val in new[u]:
                assignment[u] = val
                if c.check(assignment):
                    kept.append(val)
            del assignment[u]
            if not kept:
                return None
            new[u] = kept
        return new

    def backtrack(doms: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
        if len(assignment) == len(doms):
            yield dict(assignment)
            return
        var = min((v for v in doms if v not in assignment), key=lambda v: len(doms[v]))
        for val in doms[var]:
            counter[0] += 1
            assignment[var] = val
            if consistent(var):
                pruned = forward_check(var, doms)
                if pruned is not None:
                    yield from backtrack(pruned)
            del assignment[var]

    # Unary constraints can be applied to the domains up front
    for c in constraints:
        if len(set(c.variables)) == 1:
            (v,) = set(c.variables)
            domains[v] = [x for x in domains[v] if c.check({v: x})]
            if not domains[v]:
                return
    yield from backtrack(domains)


def solve_csp(
    variables: Mapping[str, Iterable[Any]],
    constraints: Sequence[ConstraintLike] = (),
    *,
    objective: Optional[Callable[[dict[str, Any]], float]] = None,
    maximize: bool = False,
    all_solutions: bool = False,
    limit: Optional[int] = None,
) -> Result:
    """Find an assignment of values to variables that satisfies every constraint.

    Parameters
    ----------
    variables:
        ``{name: domain}`` — the values each variable may take.
    constraints:
        :class:`Constraint` objects, or ``(variable_names, predicate)``
        tuples. The predicate gets the variables' values positionally.
    objective:
        Optional score for a full assignment. When given, every solution (up
        to ``limit``) is enumerated and the best one returned.
    maximize:
        With ``objective``, prefer higher scores.
    all_solutions:
        Collect every solution (up to ``limit``) in ``extra["solutions"]``.
    limit:
        Stop after this many solutions.

    Returns
    -------
    Result
        ``x`` is the chosen assignment dict; ``score`` is its objective value
        (``0.0`` when no objective is given); ``n_evals`` counts search nodes;
        ``converged`` is ``True`` if the search space was exhausted (or the
        first solution was all that was asked for).

    Raises
    ------
    InfeasibleError
        If no assignment satisfies the constraints.

    Examples
    --------
    >>> r = solve_csp(
    ...     {"a": [1, 2, 3], "b": [1, 2, 3]},
    ...     [(("a", "b"), lambda a, b: a + b == 4), (("a", "b"), lambda a, b: a < b)],
    ... )
    >>> r.x
    {'a': 1, 'b': 3}
    """
    domains = {name: list(dom) for name, dom in variables.items()}
    cons = [_as_constraint(c) for c in constraints]
    for c in cons:
        for v in c.variables:
            if v not in domains:
                raise KeyError(f"Constraint refers to unknown variable {v!r}")

    counter = [0]
    enumerate_all = objective is not None or all_solutions
    solutions: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_score = 0.0
    history: list[float] = []
    exhausted = True

    for sol in _search(domains, cons, counter):
        score = float(objective(sol)) if objective is not None else 0.0
        if all_solutions:
            solutions.append(sol)
        if best is None or (
            objective is not None and (score > best_score if maximize else score < best_score)
        ):
            best, best_score = sol, score
        history.append(best_score)
        if not enumerate_all:
            break
        if limit is not None and len(history) >= limit:
            exhausted = False
            break

    if best is None:
        raise InfeasibleError("No assignment satisfies all constraints")

    extra: dict[str, Any] = {"n_solutions": len(history)}
    if all_solutions:
        extra["solutions"] = solutions
    return Result(
        x=best,
        score=best_score,
        maximize=maximize,
        history=history,
        n_evals=counter[0],
        n_iter=len(history),
        converged=exhausted,
        message="Search exhausted" if exhausted and enumerate_all else (
            "First solution" if exhausted else f"Stopped at limit={limit}"
        ),
        method="solve_csp",
        extra=extra,
    )


class Recipe:
    """Builder for a constraint problem.

    Examples
    --------
    >>> r = Recipe()
    >>> r.add_variable("x", range(5)).add_variable("y", range(5))
    Recipe(variables=['x', 'y'], constraints=0)
    >>> r.add_constraint(["x", "y"], lambda x, y: x + y == 6).all_different(["x", "y"])
    Recipe(variables=['x', 'y'], constraints=2)
    >>> r.solve(objective=lambda s: s["x"], maximize=True).x
    {'x': 4, 'y': 2}
    """

    def __init__(self) -> None:
        self.variables: dict[str, list[Any]] = {}
        self.constraints: list[Constraint] = []

    def add_variable(self, name: str, domain: Iterable[Any]) -> Recipe:
        if name in self.variables:
            raise ValueError(f"Variable {name!r} already defined")
        self.variables[name] = list(domain)
        return self

    def add_constraint(
        self, variables: Union[str, Sequence[str]], predicate: Callable[..., bool], name: str = ""
    ) -> Recipe:
        if isinstance(variables, str):
            variables = (variables,)
        self.constraints.append(Constraint(tuple(variables), predicate, name))
        return self

    def all_different(self, variables: Sequence[str]) -> Recipe:
        self.constraints.extend(all_different(variables))
        return self

    def solve(self, **kwargs: Any) -> Result:
        """Solve the recipe; keyword arguments go to :func:`solve_csp`."""
        return solve_csp(self.variables, self.constraints, **kwargs)

    def __repr__(self) -> str:
        return f"Recipe(variables={list(self.variables)}, constraints={len(self.constraints)})"
