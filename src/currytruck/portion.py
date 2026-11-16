"""portion — resource allocation by linear programming.

Divide a fixed pot among hungry customers. ``linprog`` is a pure Python
two-phase simplex solver; ``allocate`` is a convenience wrapper for the
common "split a budget across options" problem.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from currytruck.exceptions import InfeasibleError, UnboundedError
from currytruck.result import Result

_EPS = 1e-9
# consecutive degenerate pivots before switching to Bland's anti-cycling rule
_DEGENERATE_LIMIT = 50

Bound = tuple[Optional[float], Optional[float]]


def _pivot(T: list[list[float]], row: int, col: int) -> None:
    pr = T[row]
    p = pr[col]
    for k in range(len(pr)):
        pr[k] /= p
    for i, r in enumerate(T):
        if i != row:
            f = r[col]
            if f != 0.0:
                for k in range(len(r)):
                    r[k] -= f * pr[k]


def _simplex(
    T: list[list[float]],
    basis: list[int],
    cost: list[float],
    allowed: list[bool],
    max_iter: int,
) -> int:
    """Run primal simplex in place. Returns iterations used; raises UnboundedError."""
    n = len(cost)
    degenerate = 0
    for it in range(max_iter):
        cb = [cost[b] for b in basis]
        bland = degenerate >= _DEGENERATE_LIMIT
        enter, best_r = -1, -_EPS
        for j in range(n):
            if not allowed[j]:
                continue
            r = cost[j] - sum(cb[i] * T[i][j] for i in range(len(T)))
            if r < best_r:
                enter, best_r = j, r
                if bland:
                    break
        if enter < 0:
            return it
        leave, best_ratio = -1, float("inf")
        for i, row in enumerate(T):
            a = row[enter]
            if a > _EPS:
                ratio = row[-1] / a
                if ratio < best_ratio - _EPS or (
                    abs(ratio - best_ratio) <= _EPS and basis[i] < basis[leave]
                ):
                    leave, best_ratio = i, ratio
        if leave < 0:
            raise UnboundedError("Objective is unbounded")
        degenerate = degenerate + 1 if best_ratio <= _EPS else 0
        _pivot(T, leave, enter)
        basis[leave] = enter
    raise RuntimeError(f"Simplex did not finish within {max_iter} iterations")


def linprog(
    c: Sequence[float],
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Optional[Sequence[Bound]] = None,
    *,
    maximize: bool = False,
    max_iter: int = 10_000,
) -> Result:
    """Solve a linear program with the two-phase simplex method.

    Minimizes (or maximizes) ``c @ x`` subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lo_i <= x_i <= hi_i

    Parameters
    ----------
    c:
        Objective coefficients.
    A_ub, b_ub:
        Inequality constraints (rows of coefficients, right-hand sides).
    A_eq, b_eq:
        Equality constraints.
    bounds:
        ``(lo, hi)`` per variable; ``None`` means unbounded on that side.
        Defaults to ``(0, None)`` — non-negative — for every variable.
    maximize:
        Maximize instead of minimize.
    max_iter:
        Pivot cap per phase.

    Returns
    -------
    Result
        ``x`` is the list of variable values; ``score`` the objective value;
        ``extra["slack"]`` holds ``b_ub - A_ub @ x``.

    Raises
    ------
    InfeasibleError
        No ``x`` satisfies the constraints.
    UnboundedError
        The objective can improve without limit.

    Examples
    --------
    Maximize ``3x + 2y`` subject to ``x + y <= 4`` and ``x + 3y <= 6``:

    >>> r = linprog([3, 2], A_ub=[[1, 1], [1, 3]], b_ub=[4, 6], maximize=True)
    >>> [round(v, 6) for v in r.x], round(r.score, 6)
    ([4.0, 0.0], 12.0)
    """
    n = len(c)
    A_ub = [list(map(float, r)) for r in (A_ub or [])]
    b_ub = [float(v) for v in (b_ub or [])]
    A_eq = [list(map(float, r)) for r in (A_eq or [])]
    b_eq = [float(v) for v in (b_eq or [])]
    if len(A_ub) != len(b_ub) or len(A_eq) != len(b_eq):
        raise ValueError("Constraint matrices and right-hand sides differ in length")
    for row in A_ub + A_eq:
        if len(row) != n:
            raise ValueError(f"Constraint row has {len(row)} coefficients, expected {n}")
    if bounds is None:
        bounds = [(0.0, None)] * n
    if len(bounds) != n:
        raise ValueError(f"bounds has {len(bounds)} entries, expected {n}")

    # --- Map each original variable onto non-negative standard-form variables.
    # x_i = offset_i + sum(coef * y_k) over its parts.
    parts: list[list[tuple[int, float]]] = []
    offsets: list[float] = []
    extra_rows: list[tuple[list[tuple[int, float]], float]] = []  # y-part <= rhs
    ny = 0
    for lo, hi in bounds:
        if lo is not None and hi is not None and lo > hi:
            raise InfeasibleError(f"Bound lo={lo} exceeds hi={hi}")
        if lo is not None:
            parts.append([(ny, 1.0)])
            offsets.append(float(lo))
            if hi is not None:
                extra_rows.append(([(ny, 1.0)], float(hi) - float(lo)))
            ny += 1
        elif hi is not None:
            parts.append([(ny, -1.0)])
            offsets.append(float(hi))
            ny += 1
        else:
            parts.append([(ny, 1.0), (ny + 1, -1.0)])
            offsets.append(0.0)
            ny += 2

    def to_y(row: Sequence[float], rhs: float) -> tuple[list[float], float]:
        coeffs = [0.0] * ny
        for i, a in enumerate(row):
            for k, s in parts[i]:
                coeffs[k] += a * s
            rhs -= a * offsets[i]
        return coeffs, rhs

    # rows: (coeffs over y, sense, rhs) with sense in {"<=", "="}
    rows: list[tuple[list[float], str, float]] = []
    for row, rhs in zip(A_ub, b_ub):
        co, r = to_y(row, rhs)
        rows.append((co, "<=", r))
    for row, rhs in zip(A_eq, b_eq):
        co, r = to_y(row, rhs)
        rows.append((co, "=", r))
    for terms, rhs in extra_rows:
        co = [0.0] * ny
        for k, s in terms:
            co[k] = s
        rows.append((co, "<=", rhs))

    # --- Build the phase-1 tableau. Columns: y | slacks | artificials | rhs
    n_slack = sum(1 for _, s, _ in rows if s == "<=")
    # normalize to rhs >= 0; a flipped "<=" becomes ">=" (surplus + artificial)
    norm = []
    for co, s, r in rows:
        if r < 0:
            norm.append(([-v for v in co], ">=" if s == "<=" else "=", -r))
        else:
            norm.append((co, s, r))
    n_art = sum(1 for _, s, _ in norm if s != "<=")
    width = ny + n_slack + n_art
    T: list[list[float]] = []
    basis: list[int] = []
    slack_col = ny
    art_col = ny + n_slack
    for co, s, r in norm:
        row = co + [0.0] * (n_slack + n_art) + [r]
        if s == "<=":
            row[slack_col] = 1.0
            basis.append(slack_col)
            slack_col += 1
        elif s == ">=":
            row[slack_col] = -1.0
            slack_col += 1
            row[art_col] = 1.0
            basis.append(art_col)
            art_col += 1
        else:
            row[art_col] = 1.0
            basis.append(art_col)
            art_col += 1
        T.append(row)

    allowed = [True] * width
    iters = 0
    if n_art:
        phase1_cost = [0.0] * (ny + n_slack) + [1.0] * n_art
        iters += _simplex(T, basis, phase1_cost, allowed, max_iter)
        infeas = sum(T[i][-1] for i, b in enumerate(basis) if b >= ny + n_slack)
        if infeas > 1e-7:
            raise InfeasibleError("No solution satisfies all constraints")
        # Drive remaining (zero-valued) artificials out of the basis
        for i in range(len(T) - 1, -1, -1):
            if basis[i] < ny + n_slack:
                continue
            for j in range(ny + n_slack):
                if abs(T[i][j]) > _EPS:
                    _pivot(T, i, j)
                    basis[i] = j
                    break
            else:  # redundant constraint
                del T[i]
                del basis[i]
        for j in range(ny + n_slack, width):
            allowed[j] = False

    sign = -1.0 if maximize else 1.0
    cy = [0.0] * width
    for i, ci in enumerate(c):
        for k, s in parts[i]:
            cy[k] += sign * ci * s
    iters += _simplex(T, basis, cy, allowed, max_iter)

    y = [0.0] * width
    for i, b in enumerate(basis):
        y[b] = T[i][-1]
    x = [offsets[i] + sum(s * y[k] for k, s in parts[i]) for i in range(n)]
    x = [0.0 if abs(v) < 1e-12 else v for v in x]
    value = sum(ci * xi for ci, xi in zip(c, x))
    slack = [b - sum(a * xi for a, xi in zip(row, x)) for row, b in zip(A_ub, b_ub)]

    return Result(
        x=x,
        score=value,
        maximize=maximize,
        history=[value],
        n_evals=0,
        n_iter=iters,
        converged=True,
        message="Optimal",
        method="linprog_simplex",
        extra={"slack": slack},
    )


def allocate(
    budget: float,
    returns: Mapping[str, float],
    *,
    costs: Optional[Mapping[str, float]] = None,
    minimum: Optional[Mapping[str, float]] = None,
    maximum: Optional[Mapping[str, float]] = None,
) -> Result:
    """Split a budget across options to maximize total return.

    Maximizes ``sum(returns[k] * x[k])`` subject to
    ``sum(costs[k] * x[k]) <= budget`` and ``minimum[k] <= x[k] <= maximum[k]``.

    Parameters
    ----------
    budget:
        Total resource available.
    returns:
        Return per unit allocated to each option.
    costs:
        Resource consumed per unit (default ``1`` for every option).
    minimum, maximum:
        Per-option allocation floor / cap (default ``0`` / unbounded).

    Returns
    -------
    Result
        ``x`` is ``{option: amount}``; ``score`` the total return;
        ``extra["unused"]`` the leftover budget.

    Examples
    --------
    >>> r = allocate(100, {"ads": 1.5, "seo": 1.2}, maximum={"ads": 60})
    >>> r.x, r.score
    ({'ads': 60.0, 'seo': 40.0}, 138.0)
    """
    names = list(returns)
    costs = costs or {}
    minimum = minimum or {}
    maximum = maximum or {}
    for mapping in (costs, minimum, maximum):
        for k in mapping:
            if k not in returns:
                raise KeyError(f"Unknown option {k!r}")
    res = linprog(
        [returns[k] for k in names],
        A_ub=[[costs.get(k, 1.0) for k in names]],
        b_ub=[budget],
        bounds=[(minimum.get(k, 0.0), maximum.get(k)) for k in names],
        maximize=True,
    )
    res.x = {k: round(v, 12) for k, v in zip(names, res.x)}
    res.score = round(res.score, 12)
    res.method = "allocate"
    res.extra["unused"] = res.extra.pop("slack")[0]
    return res
