"""portion — resource allocation by linear and integer programming.

Divide a fixed pot among hungry customers. ``linprog`` is a pure Python
two-phase simplex solver and ``milp`` adds integer variables on top of it by
branch-and-bound. ``allocate`` splits a budget across options;
``portfolio`` does mean-variance investment allocation with risk caps.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Union

from currytruck.exceptions import InfeasibleError, SearchBudgetExceeded, UnboundedError
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


def milp(
    c: Sequence[float],
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Optional[Sequence[Bound]] = None,
    *,
    integers: Optional[Sequence[int]] = None,
    maximize: bool = False,
    max_nodes: int = 10_000,
    tol: float = 1e-6,
) -> Result:
    """Solve a mixed-integer linear program by branch-and-bound.

    Same problem form as :func:`linprog`, with some variables required to
    take integer values. Each node solves an LP relaxation; fractional
    integer variables are branched on (most fractional first), and nodes
    whose relaxation can't beat the best integer solution are pruned.

    Parameters
    ----------
    c, A_ub, b_ub, A_eq, b_eq, bounds, maximize:
        As for :func:`linprog`.
    integers:
        Indices of the integer variables. ``None`` (default) means all.
        For 0/1 variables, give them bounds ``(0, 1)``.
    max_nodes:
        Branch-and-bound node budget. If it runs out after an integer
        solution was found, the best one is returned with
        ``converged=False``; if none was found, ``SearchBudgetExceeded`` is
        raised.
    tol:
        How close to an integer a value must be to count as integral.

    Returns
    -------
    Result
        ``x`` (integer variables as ``int``), ``score``, ``extra["nodes"]``.

    Raises
    ------
    InfeasibleError
        No integer-feasible solution exists.
    UnboundedError
        The LP relaxation is unbounded.
    SearchBudgetExceeded
        ``max_nodes`` ran out before any integer solution was found.

    Examples
    --------
    Maximize ``5x + 4y`` subject to ``6x + 4y <= 24``, ``x + 2y <= 6``,
    integer ``x, y`` (the LP optimum is fractional: ``(3, 1.5)``):

    >>> r = milp([5, 4], A_ub=[[6, 4], [1, 2]], b_ub=[24, 6], maximize=True)
    >>> r.x, r.score
    ([4, 0], 20.0)
    """
    n = len(c)
    if bounds is None:
        bounds = [(0.0, None)] * n
    if len(bounds) != n:
        raise ValueError(f"bounds has {len(bounds)} entries, expected {n}")
    int_idx = list(range(n)) if integers is None else sorted(set(integers))
    for i in int_idx:
        if not 0 <= i < n:
            raise IndexError(f"Integer variable index {i} out of range for {n} variables")
    if max_nodes < 1:
        raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")

    def improves(value: float, incumbent: float) -> bool:
        return value > incumbent + tol if maximize else value < incumbent - tol

    best_x: Optional[list[float]] = None
    best_val = 0.0
    history: list[float] = []
    nodes = 0
    root = True
    stack: list[list[Bound]] = [list(bounds)]
    exhausted = True

    while stack:
        if nodes >= max_nodes:
            exhausted = False
            break
        node_bounds = stack.pop()
        nodes += 1
        try:
            relax = linprog(c, A_ub, b_ub, A_eq, b_eq, node_bounds, maximize=maximize)
        except InfeasibleError:
            if root:
                raise
            continue
        except UnboundedError:
            if root:
                raise UnboundedError("LP relaxation is unbounded")
            continue
        finally:
            root = False
        if best_x is not None and not improves(relax.score, best_val):
            continue  # bound: this subtree can't beat the incumbent

        frac_i, frac_dist = -1, 0.0
        for i in int_idx:
            v = relax.x[i]
            dist = abs(v - round(v))
            if dist > tol and dist > frac_dist:
                frac_i, frac_dist = i, dist
        if frac_i < 0:
            x = [float(round(v)) if i in int_idx else v for i, v in enumerate(relax.x)]
            best_x = x
            best_val = sum(ci * xi for ci, xi in zip(c, x))
            history.append(best_val)
            continue

        v = relax.x[frac_i]
        lo, hi = node_bounds[frac_i]
        down = list(node_bounds)
        down[frac_i] = (lo, float(math.floor(v)))
        up = list(node_bounds)
        up[frac_i] = (float(math.ceil(v)), hi)
        # depth-first; explore the branch nearer the relaxed value first
        if v - math.floor(v) >= 0.5:
            stack.extend([down, up])
        else:
            stack.extend([up, down])

    if best_x is None:
        if not exhausted:
            raise SearchBudgetExceeded(f"No integer solution found within max_nodes={max_nodes}")
        raise InfeasibleError("No integer solution satisfies all constraints")

    int_set = set(int_idx)
    x_out = [int(v) if i in int_set else v for i, v in enumerate(best_x)]
    return Result(
        x=x_out,
        score=float(best_val),
        maximize=maximize,
        history=history,
        n_evals=nodes,
        n_iter=nodes,
        converged=exhausted,
        message="Optimal" if exhausted else f"Best found within max_nodes={max_nodes}",
        method="milp_branch_and_bound",
        extra={"nodes": nodes},
    )


def _project_capped_simplex(v: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> list[float]:
    # Euclidean projection onto {sum(w) == 1, lo <= w <= hi}: find the shift
    # t with sum(clip(v - t, lo, hi)) == 1 by bisection
    def total(t: float) -> float:
        return sum(min(h, max(l, x - t)) for x, l, h in zip(v, lo, hi))

    a = min(x - h for x, h in zip(v, hi))
    b = max(x - l for x, l in zip(v, lo))
    for _ in range(200):
        mid = (a + b) / 2
        if total(mid) > 1:
            a = mid
        else:
            b = mid
        if b - a < 1e-15:
            break
    t = (a + b) / 2
    return [min(h, max(l, x - t)) for x, l, h in zip(v, lo, hi)]


def _mean_variance(
    mu: list[float],
    cov: list[list[float]],
    lam: float,
    lo: list[float],
    hi: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], int]:
    n = len(mu)
    if lam == 0:
        # Pure return maximization: fill the best-returning assets first
        w = list(lo)
        room = 1.0 - sum(lo)
        for i in sorted(range(n), key=lambda i: mu[i], reverse=True):
            add = min(hi[i] - lo[i], room)
            w[i] += add
            room -= add
        return w, 0
    # FISTA (accelerated projected gradient) on f(w) = lam*w'Σw - mu'w
    def grad(v: list[float]) -> list[float]:
        return [2 * lam * sum(cov[i][j] * v[j] for j in range(n)) - mu[i] for i in range(n)]

    def value(v: list[float]) -> float:
        return lam * sum(v[i] * cov[i][j] * v[j] for i in range(n) for j in range(n)) - sum(
            m * x for m, x in zip(mu, v)
        )

    lipschitz = 2 * lam * max(sum(abs(v) for v in row) for row in cov) or 1.0
    step = 1.0 / lipschitz
    w = _project_capped_simplex([1.0 / n] * n, lo, hi)
    fw = value(w)
    y, t = list(w), 1.0
    it = 0
    for it in range(1, max_iter + 1):
        gy = grad(y)
        w_new = _project_capped_simplex([y[i] - step * gy[i] for i in range(n)], lo, hi)
        f_new = value(w_new)
        if f_new > fw:
            # adaptive restart: momentum overshot, so drop it and step from w
            t = 1.0
            gw = grad(w)
            w_new = _project_capped_simplex([w[i] - step * gw[i] for i in range(n)], lo, hi)
            f_new = value(w_new)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = [w_new[i] + (t - 1) / t_new * (w_new[i] - w[i]) for i in range(n)]
        w, fw, t = w_new, f_new, t_new
        # optimal iff a projected gradient step from w goes nowhere
        gw = grad(w)
        probe = _project_capped_simplex([w[i] - step * gw[i] for i in range(n)], lo, hi)
        if math.sqrt(sum((a - b) ** 2 for a, b in zip(probe, w))) < tol:
            break
    return w, it


def portfolio(
    expected_returns: Mapping[str, float],
    covariance: Union[Sequence[Sequence[float]], Mapping[str, Mapping[str, float]]],
    *,
    risk_aversion: float = 1.0,
    max_volatility: Optional[float] = None,
    min_weight: Union[float, Mapping[str, float]] = 0.0,
    max_weight: Union[float, Mapping[str, float]] = 1.0,
    tol: float = 1e-10,
    max_iter: int = 20_000,
) -> Result:
    """Mean-variance portfolio allocation with weight and risk caps.

    Finds fully-invested weights (summing to 1) that maximize
    ``expected_return - risk_aversion * variance``, within per-asset weight
    bounds. With ``max_volatility``, instead finds the highest-return
    portfolio whose volatility (standard deviation) stays at or below the
    cap.

    Parameters
    ----------
    expected_returns:
        ``{asset: expected return}``.
    covariance:
        Covariance matrix — a nested list in the order of
        ``expected_returns``, or ``{asset: {asset: cov}}``.
    risk_aversion:
        Trade-off between return and variance (``>= 0``). ``0`` maximizes
        return alone.
    max_volatility:
        Volatility cap. Overrides ``risk_aversion``.
    min_weight, max_weight:
        Per-asset weight bounds — one number for every asset, or
        ``{asset: bound}`` (missing assets get 0 / 1). Use ``max_weight`` to
        cap concentration.
    tol, max_iter:
        Solver convergence controls.

    Returns
    -------
    Result
        ``x`` is ``{asset: weight}``; ``score`` the expected return;
        ``extra`` has ``volatility``, ``variance``, ``risk_aversion``.

    Raises
    ------
    InfeasibleError
        The weight bounds can't sum to 1, or even the minimum-variance
        portfolio exceeds ``max_volatility``.

    Examples
    --------
    >>> r = portfolio({"bonds": 0.03, "stocks": 0.08},
    ...               [[0.0025, 0.0], [0.0, 0.04]], max_volatility=0.10)
    >>> {k: round(v, 3) for k, v in r.x.items()}
    {'bonds': 0.517, 'stocks': 0.483}
    >>> round(r.extra["volatility"], 6)
    0.1
    """
    names = list(expected_returns)
    n = len(names)
    if n == 0:
        raise ValueError("expected_returns is empty")
    if risk_aversion < 0:
        raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
    if max_volatility is not None and max_volatility < 0:
        raise ValueError(f"max_volatility must be >= 0, got {max_volatility}")
    mu = [float(expected_returns[k]) for k in names]

    if isinstance(covariance, Mapping):
        cov = [[float(covariance[a][b]) for b in names] for a in names]
    else:
        cov = [[float(v) for v in row] for row in covariance]
    if len(cov) != n or any(len(row) != n for row in cov):
        raise ValueError(f"covariance must be {n}x{n}")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(cov[i][j] - cov[j][i]) > 1e-9 * max(1.0, abs(cov[i][j])):
                raise ValueError("covariance must be symmetric")

    def per_asset(spec: Union[float, Mapping[str, float]], default: float) -> list[float]:
        if isinstance(spec, Mapping):
            for k in spec:
                if k not in expected_returns:
                    raise KeyError(f"Unknown asset {k!r}")
            return [float(spec.get(k, default)) for k in names]
        return [float(spec)] * n

    lo = per_asset(min_weight, 0.0)
    hi = per_asset(max_weight, 1.0)
    if any(l > h for l, h in zip(lo, hi)):
        raise InfeasibleError("A min_weight exceeds its max_weight")
    if sum(lo) > 1 + 1e-12 or sum(hi) < 1 - 1e-12:
        raise InfeasibleError("Weight bounds can't sum to 1")

    def variance(w: list[float]) -> float:
        return sum(w[i] * cov[i][j] * w[j] for i in range(n) for j in range(n))

    iters = 0
    if max_volatility is None:
        lam = risk_aversion
        w, iters = _mean_variance(mu, cov, lam, lo, hi, tol, max_iter)
    else:
        cap = max_volatility**2
        w, iters = _mean_variance(mu, cov, 0.0, lo, hi, tol, max_iter)
        lam = 0.0
        if variance(w) > cap * (1 + 1e-9):
            # Volatility falls as risk aversion rises: bisect (in log space)
            # for the smallest risk aversion that meets the cap
            lam_hi = 1.0
            w_hi, k = _mean_variance(mu, cov, lam_hi, lo, hi, tol, max_iter)
            iters += k
            while variance(w_hi) > cap * (1 + 1e-9):
                lam_hi *= 10
                if lam_hi > 1e12:
                    raise InfeasibleError(
                        f"Even the minimum-variance portfolio exceeds max_volatility={max_volatility}"
                    )
                w_hi, k = _mean_variance(mu, cov, lam_hi, lo, hi, tol, max_iter)
                iters += k
            lam_lo = lam_hi / 10 if lam_hi > 1.0 else 0.0
            for _ in range(60):
                mid = math.sqrt(lam_lo * lam_hi) if lam_lo > 0 else lam_hi / 2
                w_mid, k = _mean_variance(mu, cov, mid, lo, hi, tol, max_iter)
                iters += k
                if variance(w_mid) > cap:
                    lam_lo = mid
                else:
                    lam_hi, w_hi = mid, w_mid
                if lam_hi - lam_lo <= 1e-9 * lam_hi:
                    break
            w, lam = w_hi, lam_hi

    w = [min(h, max(l, v)) for v, l, h in zip(w, lo, hi)]
    var = max(0.0, variance(w))
    ret = sum(m * v for m, v in zip(mu, w))
    return Result(
        x=dict(zip(names, w)),
        score=ret,
        maximize=True,
        history=[ret],
        n_evals=0,
        n_iter=iters,
        converged=True,
        message="Optimal",
        method="portfolio",
        extra={"volatility": math.sqrt(var), "variance": var, "risk_aversion": lam},
    )
