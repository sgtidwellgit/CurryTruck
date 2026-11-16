"""simmer — iterative optimization by gradient descent.

Low and slow: take small steps downhill until the pot stops changing.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from currytruck.result import Result


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


def _clip(x: list[float], bounds: Optional[Sequence[tuple[Optional[float], Optional[float]]]]) -> list[float]:
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


def gradient_descent(
    f: Callable[[list[float]], float],
    x0: Sequence[float],
    *,
    grad: Optional[Callable[[list[float]], Sequence[float]]] = None,
    lr: float = 0.01,
    momentum: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    bounds: Optional[Sequence[tuple[Optional[float], Optional[float]]]] = None,
    maximize: bool = False,
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
        Learning rate (step size).
    momentum:
        Classical momentum coefficient in ``[0, 1)``. ``0`` is plain descent.
    max_iter:
        Iteration cap.
    tol:
        Stop when the gradient norm, or the step size, drops below ``tol``.
    bounds:
        Optional ``[(lo, hi), ...]`` per dimension; ``None`` means unbounded
        on that side. Iterates are clipped to the box (projected descent).
    maximize:
        Ascend instead of descend.

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
    if bounds is not None and len(bounds) != len(x0):
        raise ValueError("bounds must have one (lo, hi) pair per dimension")

    sign = -1.0 if maximize else 1.0

    def obj(v: list[float]) -> float:
        return sign * f(v)

    evals = 0

    def gradient(v: list[float]) -> list[float]:
        nonlocal evals
        if grad is not None:
            return [sign * g for g in grad(v)]
        evals += 2 * len(v)
        return numerical_gradient(obj, v)

    x = _clip([float(v) for v in x0], bounds)
    velocity = [0.0] * len(x)
    fx = obj(x)
    evals += 1
    history = [sign * fx]
    converged = False
    message = f"Reached max_iter={max_iter}"
    it = 0

    diverged = "Diverged (non-finite or overflowing values) — try a smaller lr"
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
        velocity = [momentum * vi - lr * gi for vi, gi in zip(velocity, g)]
        new_x = _clip([xi + vi for xi, vi in zip(x, velocity)], bounds)
        try:
            new_fx = obj(new_x)
        except OverflowError:
            message = diverged
            break
        evals += 1
        step = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_x, x)))
        x, fx = new_x, new_fx
        history.append(sign * fx)
        if not math.isfinite(fx):
            message = diverged
            break
        if step < tol:
            converged, message = True, "Step size below tol"
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
        method="gradient_descent",
    )
