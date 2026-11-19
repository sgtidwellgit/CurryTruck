"""recipe — constraint satisfaction by backtracking search.

Every recipe has rules: this goes before that, these two never share a pan.
Declare variables with their possible values and the rules between them;
``recipe`` finds assignments that follow every rule — and, given an
objective, the best one.

Search is depth-first backtracking with AC-3 arc-consistency preprocessing,
minimum-remaining-values variable ordering, and forward checking.
:func:`schedule_shifts` is a ready-made employee scheduling template.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from currytruck.exceptions import InfeasibleError, SearchBudgetExceeded
from currytruck.result import Result


@dataclass(frozen=True)
class Constraint:
    """A rule over one or more variables.

    ``predicate`` receives the variables' values positionally, in the order
    given by ``variables``, and returns ``True`` if the combination is allowed.

    ``partial``, if given, is called with ``{variable: value}`` for just the
    variables assigned so far and returns ``False`` as soon as the
    constraint can no longer be satisfied. Without it, a constraint is only
    checked once all of its variables are assigned — fine for small rules,
    slow for global ones spanning many variables.
    """

    variables: tuple[str, ...]
    predicate: Callable[..., bool]
    name: str = ""
    partial: Optional[Callable[[Mapping[str, Any]], bool]] = None

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


def count_at_most(variables: Sequence[str], value: Any, n: int) -> Constraint:
    """At most ``n`` of ``variables`` take ``value``. Prunes partial assignments.

    >>> r = solve_csp({v: "ab" for v in "xyz"}, [count_at_most("xyz", "a", 1)], all_solutions=True)
    >>> r.extra["n_solutions"]
    4
    """
    vs = tuple(variables)
    return Constraint(
        vs,
        lambda *vals: sum(v == value for v in vals) <= n,
        name=f"count_at_most({value!r}, {n})",
        partial=lambda assigned: sum(v == value for v in assigned.values()) <= n,
    )


def count_at_least(variables: Sequence[str], value: Any, n: int) -> Constraint:
    """At least ``n`` of ``variables`` take ``value``. Prunes partial assignments."""
    vs = tuple(variables)
    return Constraint(
        vs,
        lambda *vals: sum(v == value for v in vals) >= n,
        name=f"count_at_least({value!r}, {n})",
        partial=lambda assigned: (
            sum(v == value for v in assigned.values()) + len(vs) - len(assigned) >= n
        ),
    )


def _as_constraint(c: ConstraintLike) -> Constraint:
    if isinstance(c, Constraint):
        return c
    variables, predicate = c
    if isinstance(variables, str):
        variables = (variables,)
    return Constraint(tuple(variables), predicate)


def _ac3(domains: dict[str, list[Any]], constraints: list[Constraint]) -> bool:
    """Prune values with no support across binary constraints. False if a domain empties."""
    arcs: dict[str, list[tuple[str, Constraint]]] = {v: [] for v in domains}
    for c in constraints:
        if len(c.variables) == 2 and c.variables[0] != c.variables[1]:
            a, b = c.variables
            arcs[a].append((b, c))
            arcs[b].append((a, c))
    queue = deque((x, y, c) for x in arcs for y, c in arcs[x])
    while queue:
        x, y, c = queue.popleft()
        kept = []
        for vx in domains[x]:
            for vy in domains[y]:
                if c.check({x: vx, y: vy}):
                    kept.append(vx)
                    break
        if len(kept) < len(domains[x]):
            domains[x] = kept
            if not kept:
                return False
            queue.extend((z, x, c2) for z, c2 in arcs[x] if z != y)
    return True


class _BudgetHit(Exception):
    pass


def _search(
    domains: dict[str, list[Any]],
    constraints: list[Constraint],
    counter: list[int],
    max_nodes: Optional[int],
    ac3: bool,
) -> Iterator[dict[str, Any]]:
    by_var: dict[str, list[Constraint]] = {v: [] for v in domains}
    for c in constraints:
        for v in set(c.variables):
            by_var[v].append(c)

    assignment: dict[str, Any] = {}

    def consistent(var: str) -> bool:
        for c in by_var[var]:
            if all(v in assignment for v in c.variables):
                if not c.check(assignment):
                    return False
            elif c.partial is not None:
                if not c.partial({v: assignment[v] for v in c.variables if v in assignment}):
                    return False
        return True

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
            if max_nodes is not None and counter[0] >= max_nodes:
                raise _BudgetHit
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
    if ac3 and not _ac3(domains, constraints):
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
    max_nodes: Optional[int] = None,
    ac3: bool = True,
) -> Result:
    """Find an assignment of values to variables that satisfies every constraint.

    Parameters
    ----------
    variables:
        ``{name: domain}`` — the values each variable may take. Values are
        tried in the order given, so put likely-good values first.
    constraints:
        :class:`Constraint` objects, or ``(variable_names, predicate)``
        tuples. The predicate gets the variables' values positionally.
    objective:
        Optional score for a full assignment. When given, every solution (up
        to ``limit`` / ``max_nodes``) is enumerated and the best one returned.
    maximize:
        With ``objective``, prefer higher scores.
    all_solutions:
        Collect every solution (up to ``limit``) in ``extra["solutions"]``.
    limit:
        Stop after this many solutions.
    max_nodes:
        Search-node budget. If it runs out, the best solution so far is
        returned with ``converged=False``; with no solution yet,
        ``SearchBudgetExceeded`` is raised.
    ac3:
        Run AC-3 arc consistency over binary constraints before searching.

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
    SearchBudgetExceeded
        If ``max_nodes`` ran out before any solution was found.

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
    if max_nodes is not None and max_nodes < 1:
        raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")

    counter = [0]
    enumerate_all = objective is not None or all_solutions
    solutions: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_score = 0.0
    history: list[float] = []
    exhausted = True
    stop_reason = ""

    try:
        for sol in _search(domains, cons, counter, max_nodes, ac3):
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
                exhausted, stop_reason = False, f"Stopped at limit={limit}"
                break
    except _BudgetHit:
        exhausted, stop_reason = False, f"Best found within max_nodes={max_nodes}"

    if best is None:
        if not exhausted:
            raise SearchBudgetExceeded(f"No solution found within max_nodes={max_nodes}")
        raise InfeasibleError("No assignment satisfies all constraints")

    if exhausted:
        message = "Search exhausted" if enumerate_all else "First solution"
    else:
        message = stop_reason
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
        message=message,
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

    def at_most(self, variables: Sequence[str], value: Any, n: int) -> Recipe:
        self.constraints.append(count_at_most(variables, value, n))
        return self

    def at_least(self, variables: Sequence[str], value: Any, n: int) -> Recipe:
        self.constraints.append(count_at_least(variables, value, n))
        return self

    def solve(self, **kwargs: Any) -> Result:
        """Solve the recipe; keyword arguments go to :func:`solve_csp`."""
        return solve_csp(self.variables, self.constraints, **kwargs)

    def __repr__(self) -> str:
        return f"Recipe(variables={list(self.variables)}, constraints={len(self.constraints)})"


def schedule_shifts(
    employees: Sequence[str],
    shifts: Sequence[str],
    *,
    coverage: Union[int, Mapping[str, int]] = 1,
    availability: Optional[Mapping[str, Iterable[str]]] = None,
    max_shifts: Optional[Union[int, Mapping[str, int]]] = None,
    min_shifts: Optional[Union[int, Mapping[str, int]]] = None,
    conflicts: Iterable[tuple[str, str]] = (),
    preferences: Optional[Mapping[str, Mapping[str, float]]] = None,
    max_nodes: int = 100_000,
) -> Result:
    """Assign employees to shifts.

    Parameters
    ----------
    employees:
        Employee names.
    shifts:
        Shift names (e.g. ``"mon-am"``).
    coverage:
        Staff needed per shift — one number, or ``{shift: n}`` (default 1).
    availability:
        ``{employee: shifts they can work}``. Employees not listed are
        available for every shift.
    max_shifts, min_shifts:
        Per-employee shift count limits — one number, or ``{employee: n}``.
    conflicts:
        Pairs of shifts no one may work both of (e.g. overlapping times, or
        a closing shift followed by an opening one).
    preferences:
        ``{employee: {shift: score}}``. When given, the schedule with the
        highest total preference is sought, and each shift tries its most
        preferred employees first.
    max_nodes:
        Search budget. With preferences, the best schedule found within the
        budget is returned (``converged`` is ``False`` if the search was cut
        short); without them, the first valid schedule is returned.

    Returns
    -------
    Result
        ``x`` is ``{shift: [employees]}``; ``score`` the total preference
        (``0.0`` without preferences); ``extra["load"]`` the number of shifts
        per employee.

    Raises
    ------
    InfeasibleError
        No schedule satisfies the rules.
    SearchBudgetExceeded
        No valid schedule was found within ``max_nodes``.

    Examples
    --------
    >>> r = schedule_shifts(
    ...     ["ann", "bob", "cat"], ["mon", "tue", "wed"],
    ...     availability={"ann": ["mon"]}, max_shifts=1,
    ... )
    >>> r.x
    {'mon': ['ann'], 'tue': ['bob'], 'wed': ['cat']}
    """
    emps = list(employees)
    index = {e: i for i, e in enumerate(emps)}
    if len(index) != len(emps):
        raise ValueError("employees contains duplicates")
    shift_set = set(shifts)

    def need(s: str) -> int:
        k = coverage.get(s, 1) if isinstance(coverage, Mapping) else coverage
        if k < 0:
            raise ValueError(f"coverage for {s!r} must be >= 0")
        return k

    def limit(spec: Optional[Union[int, Mapping[str, int]]], e: str) -> Optional[int]:
        if spec is None:
            return None
        if isinstance(spec, Mapping):
            return spec.get(e)
        return spec

    available: dict[str, set[str]] = {e: set(shifts) for e in emps}
    if availability:
        for e, ok in availability.items():
            if e not in index:
                raise KeyError(f"Unknown employee {e!r}")
            ok = set(ok)
            unknown = ok - shift_set
            if unknown:
                raise KeyError(f"Unknown shift(s) {sorted(unknown)} for {e!r}")
            available[e] = ok
    prefs = preferences or {}
    for e in prefs:
        if e not in index:
            raise KeyError(f"Unknown employee {e!r}")

    def pref(e: str, s: str) -> float:
        return float(prefs.get(e, {}).get(s, 0.0))

    variables: dict[str, list[str]] = {}
    slots_of: dict[str, list[str]] = {}
    for s in shifts:
        cands = [e for e in emps if s in available[e]]
        # most-preferred first so early solutions are already good ones
        cands.sort(key=lambda e: (-pref(e, s), index[e]))
        slots_of[s] = [f"{s}#{k}" for k in range(need(s))]
        for var in slots_of[s]:
            variables[var] = cands

    cons: list[Constraint] = []
    for s in shifts:
        slots = slots_of[s]
        # distinct people per shift; ordered by index to skip symmetric duplicates
        for a, b in zip(slots, slots[1:]):
            cons.append(Constraint((a, b), lambda x, y: index[x] < index[y], name=f"order({s})"))
    for s1, s2 in conflicts:
        for s in (s1, s2):
            if s not in shift_set:
                raise KeyError(f"Unknown shift {s!r} in conflicts")
        for a in slots_of[s1]:
            for b in slots_of[s2]:
                cons.append(Constraint((a, b), lambda x, y: x != y, name=f"conflict({s1},{s2})"))
    all_slots = [v for s in shifts for v in slots_of[s]]
    for e in emps:
        hi = limit(max_shifts, e)
        lo = limit(min_shifts, e)
        if hi is not None:
            cons.append(count_at_most(all_slots, e, hi))
        if lo is not None:
            cons.append(count_at_least(all_slots, e, lo))

    def total_pref(sol: dict[str, Any]) -> float:
        return sum(pref(sol[v], v.rsplit("#", 1)[0]) for v in sol)

    res = solve_csp(
        variables,
        cons,
        objective=total_pref if preferences else None,
        maximize=True,
        max_nodes=max_nodes,
    )
    sol = res.x
    schedule = {s: [sol[v] for v in slots_of[s]] for s in shifts}
    load = {e: sum(e in staff for staff in schedule.values()) for e in emps}
    res.x = schedule
    res.method = "schedule_shifts"
    res.extra["load"] = load
    return res
