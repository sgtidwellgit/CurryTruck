"""simmer — iterative optimization by gradient descent, Adam, and RMSProp.

Low and slow: take small steps downhill until the pot stops changing.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from currytruck.result import Result

Bounds = Optional[Sequence[tuple[Optional[float], Optional[float]]]]
Callback = Callable[[int, list[float], float], Optional[bool]]
# Update rule: (gradient, iteration) -> step to add to x
StepRule = Callable[[list[float], int], list[float]]


def numerical_gradient(
    f: Callable[[list[float]], float],
    x: Sequence[float],
    *,
    eps: float = 1e-6,
) -> list[float]:
    """Central-difference estimate of the gradient of ``f`` at ``x``.

    Examples
    --------
    >>> [round(g, 4) for g in numerical_gradient(lambda v: v[0] ** 2 + 3 * v[1], [2.0, 0.0])]
    [4.0, 3.0]
    """
    x = list(x)
    grad = []
    for i in range(len(x)):
        h = eps * max(1.0, abs(x[i]))
        orig = x[i]
        x[i] = orig + h
        f_plus = f(x)
        x[i] = orig - h
        f_minus = f(x)
        x[i] = orig
        grad.append((f_plus - f_minus) / (2 * h))
    return grad


def _clip(x: list[float], bounds: Bounds) -> list[float]:
    if bounds is None:
        return x
    out = []
    for xi, (lo, hi) in zip(x, bounds):
        if lo is not None and xi < lo:
            xi = lo
        if hi is not None and xi > hi:
            xi = hi
        out.append(xi)
    return out


def _descend(
    f: Callable[[list[float]], float],
    x0: Sequence[float],
    *,
    grad: Optional[Callable[[list[float]], Sequence[float]]],
    make_rule: Callable[[int], StepRule],
    line_search: bool,
    max_iter: int,
    tol: float,
    bounds: Bounds,
    maximize: bool,
    callback: Optional[Callback],
    method: str,
) -> Result:
    if bounds is not None and len(bounds) != len(x0):
        raise ValueError("bounds must have one (lo, hi) pair per dimension")

    sign = -1.0 if maximize else 1.0
    evals = 0

    def obj(v: list[float]) -> float:
        nonlocal evals
        evals += 1
        return sign * f(v)

    def gradient(v: list[float]) -> list[float]:
        nonlocal evals
        if grad is not None:
            return [sign * g for g in grad(v)]
        evals += 2 * len(v)
        return numerical_gradient(lambda u: sign * f(u), v)

    x = _clip([float(v) for v in x0], bounds)
    rule = make_rule(len(x))
    fx = obj(x)
    history = [sign * fx]
    converged = False
    message = f"Reached max_iter={max_iter}"
    diverged = "Diverged (non-finite or overflowing values) — try a smaller lr"
    it = 0

    for it in range(1, max_iter + 1):
        try:
            g = gradient(x)
        except OverflowError:
            message = diverged
            break
        if not all(math.isfinite(gi) for gi in g):
            message = diverged
            break
        gnorm = math.sqrt(sum(gi * gi for gi in g))
        if gnorm < tol:
            converged, message = True, "Gradient norm below tol"
            break
        step = rule(g, it)
        try:
            new_x = _clip([xi + si for xi, si in zip(x, step)], bounds)
            new_fx = obj(new_x)
            if line_search:
                # Backtracking (Armijo): halve the step until f drops by at least
                # a small fraction of what the linear model predicts
                slope = sum(gi * si for gi, si in zip(g, step))
                shrink = 1.0
                while new_fx > fx + 1e-4 * shrink * slope and shrink > 1e-10:
                    shrink *= 0.5
                    new_x = _clip([xi + shrink * si for xi, si in zip(x, step)], bounds)
                    new_fx = obj(new_x)
        except OverflowError:
            message = diverged
            break
        moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_x, x)))
        x, fx = new_x, new_fx
        history.append(sign * fx)
        if not math.isfinite(fx):
            message = diverged
            break
        if moved < tol:
            converged, message = True, "Step size below tol"
            break
        if callback is not None and callback(it, x, sign * fx):
            message = "Stopped by callback"
            break

    return Result(
        x=x,
        score=sign * fx,
        maximize=maximize,
        history=history,
        n_evals=evals,
        n_iter=it,
        converged=converged,
        message=message,
        method=method,
    )


def gradient_descent(
    f: Callable[[list[float]], float],
    x0: Sequence[float],
    *,
    grad: Optional[Callable[[list[float]], Sequence[float]]] = None,
    lr: float = 0.01,
    momentum: float = 0.0,
    line_search: bool = False,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    bounds: Bounds = None,
    maximize: bool = False,
    callback: Optional[Callback] = None,
) -> Result:
    """Minimize (or maximize) a differentiable function by gradient descent.

    Parameters
    ----------
    f:
        Objective, called with a list of floats.
    x0:
        Starting point.
    grad:
        Gradient function. If omitted, a central-difference numerical
        gradient is used (two evaluations of ``f`` per dimension per step).
    lr:
        Learning rate (step size). With ``line_search=True``, the initial
        step tried each iteration.
    momentum:
        Classical momentum coefficient in ``[0, 1)``. ``0`` is plain descent.
    line_search:
        Backtracking (Armijo) line search: halve the step until it gives a
        sufficient decrease. Makes descent robust to a too-large ``lr``.
        Can't be combined with ``momentum``.
    max_iter:
        Iteration cap.
    tol:
        Stop when the gradient norm, or the step size, drops below ``tol``.
    bounds:
        Optional ``[(lo, hi), ...]`` per dimension; ``None`` means unbounded
        on that side. Iterates are clipped to the box (projected descent).
    maximize:
        Ascend instead of descend.
    callback:
        ``callback(iteration, x, score)`` after each step; return ``True``
        to stop early.

    Examples
    --------
    >>> r = gradient_descent(lambda v: (v[0] - 3) ** 2, [0.0], lr=0.1)
    >>> round(r.x[0], 4)
    3.0
    """
    if not 0 <= momentum < 1:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if line_search and momentum:
        raise ValueError("line_search can't be combined with momentum")

    def make_rule(n: int) -> StepRule:
        velocity = [0.0] * n

        def rule(g: list[float], it: int) -> list[float]:
            nonlocal velocity
            velocity = [momentum * vi - lr * gi for vi, gi in zip(velocity, g)]
            return velocity

        return rule

    return _descend(
        f, x0, grad=grad, make_rule=make_rule, line_search=line_search, max_iter=max_iter,
        tol=tol, bounds=bounds, maximize=maximize, callback=callback, method="gradient_descent",
    )


def adam(
    f: Callable[[list[float]], float],
    x0: Sequence[float],
    *,
    grad: Optional[Callable[[list[float]], Sequence[float]]] = None,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    bounds: Bounds = None,
    maximize: bool = False,
    callback: Optional[Callback] = None,
) -> Result:
    """Minimize (or maximize) with the Adam optimizer.

    Per-coordinate step sizes adapted from running averages of the gradient
    and its square, with bias correction. Robust to badly scaled problems.
    Parameters as for :func:`gradient_descent`, plus Adam's ``beta1``,
    ``beta2`` and ``eps``.

    Adam's steps don't shrink to zero near an optimum the way plain descent
    does, so it usually runs to ``max_iter``; the result is still close.

    Examples
    --------
    >>> r = adam(lambda v: (v[0] - 3) ** 2 + 100 * (v[1] + 1) ** 2, [0.0, 0.0], lr=0.05, max_iter=3000)
    >>> [round(v, 2) for v in r.x]
    [3.0, -1.0]
    """
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("beta1 and beta2 must be in [0, 1)")

    def make_rule(n: int) -> StepRule:
        m = [0.0] * n
        v = [0.0] * n

        def rule(g: list[float], it: int) -> list[float]:
            step = []
            for i, gi in enumerate(g):
                m[i] = beta1 * m[i] + (1 - beta1) * gi
                v[i] = beta2 * v[i] + (1 - beta2) * gi * gi
                m_hat = m[i] / (1 - beta1**it)
                v_hat = v[i] / (1 - beta2**it)
                step.append(-lr * m_hat / (math.sqrt(v_hat) + eps))
            return step

        return rule

    return _descend(
        f, x0, grad=grad, make_rule=make_rule, line_search=False, max_iter=max_iter, tol=tol,
        bounds=bounds, maximize=maximize, callback=callback, method="adam",
    )


def rmsprop(
    f: Callable[[list[float]], float],
    x0: Sequence[float],
    *,
    grad: Optional[Callable[[list[float]], Sequence[float]]] = None,
    lr: float = 0.01,
    rho: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    bounds: Bounds = None,
    maximize: bool = False,
    callback: Optional[Callback] = None,
) -> Result:
    """Minimize (or maximize) with RMSProp.

    Scales each coordinate's step by a running average of its squared
    gradient. Parameters as for :func:`gradient_descent`, plus the decay
    rate ``rho`` and ``eps``.
    """
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not 0 <= rho < 1:
        raise ValueError(f"rho must be in [0, 1), got {rho}")

    def make_rule(n: int) -> StepRule:
        v = [0.0] * n

        def rule(g: list[float], it: int) -> list[float]:
            step = []
            for i, gi in enumerate(g):
                v[i] = rho * v[i] + (1 - rho) * gi * gi
                step.append(-lr * gi / (math.sqrt(v[i]) + eps))
            return step

        return rule

    return _descend(
        f, x0, grad=grad, make_rule=make_rule, line_search=False, max_iter=max_iter, tol=tol,
        bounds=bounds, maximize=maximize, callback=callback, method="rmsprop",
    )
