"""serve — results and reporting: rankings and convergence statistics.

Plate it up: compare runs, see how fast each one got there, and print a
readable summary.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from currytruck.exceptions import EmptySpaceError
from currytruck.result import Result


def rank(results: Sequence[Result], *, top: Optional[int] = None) -> list[Result]:
    """Sort results best first.

    All results must agree on ``maximize`` — comparing a minimization run
    with a maximization run is refused rather than guessed at.
    """
    if not results:
        return []
    senses = {r.maximize for r in results}
    if len(senses) > 1:
        raise ValueError("Cannot rank results that mix maximize=True and maximize=False")
    ordered = sorted(results, key=lambda r: r.score, reverse=senses.pop())
    return ordered[:top] if top is not None else ordered


def best(results: Sequence[Result]) -> Result:
    """The single best result."""
    if not results:
        raise EmptySpaceError("No results to choose from")
    return rank(results)[0]


def convergence(result: Result, *, tol: float = 1e-9) -> dict[str, Any]:
    """Convergence statistics from a result's ``history``.

    Returns
    -------
    dict with keys:

    - ``initial`` / ``final`` — first and last recorded score
    - ``improvement`` — how much the score improved (always ``>= 0`` for a
      monotone history)
    - ``improvement_pct`` — improvement relative to ``|initial|`` (``None``
      if ``initial`` is 0)
    - ``iterations`` — length of the history
    - ``iteration_of_best`` — first index where the final best was reached
    - ``plateau`` — trailing iterations with no improvement beyond ``tol``
    - ``n_evals``, ``converged``, ``message`` — copied from the result

    Examples
    --------
    >>> r = Result(x=None, score=1.0, history=[5.0, 3.0, 1.0, 1.0, 1.0])
    >>> s = convergence(r)
    >>> s["improvement"], s["iteration_of_best"], s["plateau"]
    (4.0, 2, 2)
    """
    h = result.history
    out: dict[str, Any] = {
        "n_evals": result.n_evals,
        "converged": result.converged,
        "message": result.message,
        "iterations": len(h),
    }
    if not h:
        out.update(initial=None, final=None, improvement=None, improvement_pct=None,
                   iteration_of_best=None, plateau=0)
        return out
    initial, final = h[0], h[-1]
    improvement = (final - initial) if result.maximize else (initial - final)

    def better(a: float, b: float) -> bool:
        return a > b + tol if result.maximize else a < b - tol

    target = max(h) if result.maximize else min(h)
    first_best = next(i for i, v in enumerate(h) if not better(target, v))
    plateau = 0
    for i in range(len(h) - 1, 0, -1):
        if better(h[i], h[i - 1]):
            break
        plateau += 1

    out.update(
        initial=initial,
        final=final,
        improvement=improvement,
        improvement_pct=(100.0 * improvement / abs(initial)) if initial else None,
        iteration_of_best=first_best,
        plateau=plateau,
    )
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    s = repr(v)
    return s if len(s) <= 80 else s[:77] + "..."


def report(result: Result) -> str:
    """A short human-readable summary of a run.

    >>> print(report(Result(x=[1.0], score=0.5, history=[2.0, 0.5], method="demo", n_evals=2)))
    demo (minimize)
      best score : 0.5
      solution   : [1.0]
      evaluations: 2
      iterations : 0
      converged  : True
      improvement: 1.5 (75% from start)
    """
    stats = convergence(result)
    lines = [
        f"{result.method or 'result'} ({'maximize' if result.maximize else 'minimize'})",
        f"  best score : {_fmt(result.score)}",
        f"  solution   : {_fmt(result.x)}",
        f"  evaluations: {result.n_evals}",
        f"  iterations : {result.n_iter}",
        f"  converged  : {result.converged}",
    ]
    if result.message:
        lines.append(f"  message    : {result.message}")
    if stats["improvement"] is not None:
        pct = stats["improvement_pct"]
        tail = f" ({pct:.0f}% from start)" if pct is not None else ""
        lines.append(f"  improvement: {_fmt(stats['improvement'])}{tail}")
    return "\n".join(lines)


def leaderboard(results: Sequence[Result], *, top: Optional[int] = None) -> str:
    """A ranked text table comparing several runs.

    >>> a = Result(x=[0], score=2.0, method="a", n_evals=10)
    >>> b = Result(x=[1], score=1.0, method="b", n_evals=20)
    >>> print(leaderboard([a, b]))
    rank  method  score  evals  converged
       1  b           1     20  True
       2  a           2     10  True
    """
    ranked = rank(results, top=top)
    rows = [("rank", "method", "score", "evals", "converged")]
    for i, r in enumerate(ranked, 1):
        rows.append((str(i), r.method or "-", _fmt(r.score), str(r.n_evals), str(r.converged)))
    widths = [max(len(row[c]) for row in rows) for c in range(5)]
    right = {0, 2, 3}
    lines = []
    for row in rows:
        cells = [
            cell.rjust(widths[c]) if c in right and row is not rows[0] else cell.ljust(widths[c])
            for c, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)
