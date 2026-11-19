import itertools
import random

import pytest
from currytruck import (
    linprog,
    milp,
    allocate,
    portfolio,
    InfeasibleError,
    UnboundedError,
    SearchBudgetExceeded,
)


class TestLinprog:
    def test_textbook_max(self):
        # max 3x + 5y st x <= 4, 2y <= 12, 3x + 2y <= 18  -> (2, 6), 36
        r = linprog([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
        assert r.x == pytest.approx([2, 6])
        assert r.score == pytest.approx(36)
        assert r.extra["slack"] == pytest.approx([2, 0, 0])

    def test_min_with_ge_constraints(self):
        # diet problem: min 2a + 3b st a + b >= 4, a + 3b >= 6
        r = linprog([2, 3], A_ub=[[-1, -1], [-1, -3]], b_ub=[-4, -6])
        assert r.x == pytest.approx([3, 1])
        assert r.score == pytest.approx(9)

    def test_equality(self):
        r = linprog([1, 2, 3], A_eq=[[1, 1, 1]], b_eq=[10])
        assert r.x == pytest.approx([10, 0, 0])

    def test_redundant_equalities(self):
        r = linprog([1, 1], A_eq=[[1, 1], [2, 2]], b_eq=[5, 10])
        assert r.score == pytest.approx(5)

    def test_bounds_lower_and_upper(self):
        r = linprog([1, 1], bounds=[(2, 5), (-3, 1)])
        assert r.x == pytest.approx([2, -3])

    def test_upper_only_bound(self):
        r = linprog([-1], bounds=[(None, 7)])
        assert r.x == pytest.approx([7])

    def test_free_variable(self):
        # min x st x >= -4 (as -x <= 4), x free
        r = linprog([1], A_ub=[[-1]], b_ub=[4], bounds=[(None, None)])
        assert r.x == pytest.approx([-4])

    def test_infeasible(self):
        with pytest.raises(InfeasibleError):
            linprog([1, 1], A_ub=[[1, 1], [-1, -1]], b_ub=[1, -2])

    def test_infeasible_bounds(self):
        with pytest.raises(InfeasibleError):
            linprog([1], bounds=[(3, 1)])

    def test_unbounded(self):
        with pytest.raises(UnboundedError):
            linprog([1, 1], A_ub=[[1, -1]], b_ub=[1], maximize=True)

    def test_degenerate_does_not_cycle(self):
        # Beale's classic cycling example (cycles under Dantzig's rule without anti-cycling)
        c = [-0.75, 150, -0.02, 6]
        A = [[0.25, -60, -0.04, 9], [0.5, -90, -0.02, 3], [0, 0, 1, 0]]
        b = [0, 0, 1]
        r = linprog(c, A_ub=A, b_ub=b)
        assert r.score == pytest.approx(-0.05)

    def test_transportation_problem(self):
        # 2 warehouses (supply 20, 30) -> 3 stores (demand 10, 25, 15)
        cost = [8, 6, 10, 9, 12, 13]
        A_eq = [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]]
        b_eq = [20, 30, 10, 25, 15]
        r = linprog(cost, A_eq=A_eq, b_eq=b_eq)
        assert r.score == pytest.approx(465)
        assert all(v >= -1e-9 for v in r.x)

    def test_shape_errors(self):
        with pytest.raises(ValueError):
            linprog([1, 2], A_ub=[[1]], b_ub=[1])
        with pytest.raises(ValueError):
            linprog([1], A_ub=[[1]], b_ub=[])
        with pytest.raises(ValueError):
            linprog([1], bounds=[(0, 1), (0, 1)])


class TestAllocate:
    def test_caps(self):
        r = allocate(100, {"ads": 1.5, "seo": 1.2}, maximum={"ads": 60})
        assert r.x == {"ads": 60.0, "seo": 40.0}
        assert r.extra["unused"] == pytest.approx(0)

    def test_minimums_and_costs(self):
        r = allocate(100, {"a": 3, "b": 1}, costs={"a": 2, "b": 1}, minimum={"b": 20})
        assert r.x == pytest.approx({"a": 40, "b": 20})
        assert r.score == pytest.approx(140)

    def test_leftover_budget(self):
        r = allocate(100, {"a": 1}, maximum={"a": 30})
        assert r.extra["unused"] == pytest.approx(70)

    def test_infeasible_minimums(self):
        with pytest.raises(InfeasibleError):
            allocate(10, {"a": 1, "b": 1}, minimum={"a": 8, "b": 8})

    def test_unknown_option(self):
        with pytest.raises(KeyError):
            allocate(10, {"a": 1}, maximum={"zzz": 1})


def brute_integer(c, A_ub, b_ub, ranges, maximize):
    best = None
    for x in itertools.product(*ranges):
        if all(sum(a * v for a, v in zip(row, x)) <= b + 1e-9 for row, b in zip(A_ub, b_ub)):
            val = sum(ci * v for ci, v in zip(c, x))
            if best is None or (val > best if maximize else val < best):
                best = val
    return best


class TestMilp:
    def test_docstring_problem(self):
        r = milp([5, 4], A_ub=[[6, 4], [1, 2]], b_ub=[24, 6], maximize=True)
        assert r.x == [4, 0] and r.score == 20
        assert all(isinstance(v, int) for v in r.x)

    @pytest.mark.parametrize("seed", range(15))
    def test_matches_brute_force(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 4)
        c = [rng.randint(-6, 6) for _ in range(n)]
        A = [[rng.randint(-3, 6) for _ in range(n)] for _ in range(rng.randint(1, 3))]
        b = [rng.randint(3, 20) for _ in A]
        bounds = [(0, 5)] * n
        maximize = bool(seed % 2)
        expected = brute_integer(c, A, b, [range(6)] * n, maximize)
        if expected is None:
            with pytest.raises(InfeasibleError):
                milp(c, A_ub=A, b_ub=b, bounds=bounds, maximize=maximize)
        else:
            r = milp(c, A_ub=A, b_ub=b, bounds=bounds, maximize=maximize)
            assert r.score == pytest.approx(expected)

    def test_binary_knapsack(self):
        values, weights = [60, 100, 120], [10, 20, 30]
        r = milp(values, A_ub=[weights], b_ub=[50], bounds=[(0, 1)] * 3, maximize=True)
        assert r.x == [0, 1, 1] and r.score == 220

    def test_mixed_integer(self):
        # x integer, y continuous: max x + y st 2x + 2y <= 5, x <= 1.5
        r = milp([1, 1], A_ub=[[2, 2], [1, 0]], b_ub=[5, 1.5], integers=[0], maximize=True)
        assert r.x[0] == 1 and isinstance(r.x[0], int)
        assert r.x[1] == pytest.approx(1.5)
        assert r.score == pytest.approx(2.5)

    def test_equality(self):
        r = milp([1, 1], A_eq=[[2, 3]], b_eq=[12])
        assert r.score == pytest.approx(4)  # x=0, y=4

    def test_integer_infeasible(self):
        # 2x == 3 has no integer solution
        with pytest.raises(InfeasibleError):
            milp([1], A_eq=[[2]], b_eq=[3])

    def test_unbounded(self):
        with pytest.raises(UnboundedError):
            milp([1, 1], maximize=True)

    def test_budget_with_incumbent(self):
        rng = random.Random(1)
        n = 12
        values = [rng.randint(10, 60) for _ in range(n)]
        weights = [rng.randint(5, 30) for _ in range(n)]
        kw = dict(A_ub=[weights], b_ub=[sum(weights) // 2], bounds=[(0, 1)] * n, maximize=True)
        full = milp(values, **kw)
        assert full.converged and full.extra["nodes"] > 2
        cut = milp(values, **kw, max_nodes=full.extra["nodes"] - 1)
        assert not cut.converged
        assert cut.extra["nodes"] == full.extra["nodes"] - 1
        assert cut.score <= full.score

    def test_budget_without_incumbent(self):
        with pytest.raises(SearchBudgetExceeded):
            milp([1, 1], A_eq=[[2, 2]], b_eq=[3], bounds=[(0, 10)] * 2, max_nodes=1)

    def test_bad_index(self):
        with pytest.raises(IndexError):
            milp([1], integers=[3])


COV = [[0.04, 0.006, 0.0], [0.006, 0.09, 0.01], [0.0, 0.01, 0.0025]]
MU = {"stocks": 0.08, "growth": 0.12, "bonds": 0.03}


def port_var(w, names=("stocks", "growth", "bonds")):
    return sum(w[a] * COV[i][j] * w[b] for i, a in enumerate(names) for j, b in enumerate(names))


class TestPortfolio:
    def test_fully_invested_and_in_bounds(self):
        r = portfolio(MU, COV, risk_aversion=2.0)
        assert sum(r.x.values()) == pytest.approx(1.0)
        assert all(-1e-12 <= v <= 1 + 1e-12 for v in r.x.values())
        assert r.extra["variance"] == pytest.approx(port_var(r.x))

    def test_optimal_against_grid(self):
        lam = 3.0
        r = portfolio(MU, COV, risk_aversion=lam)
        utility = r.score - lam * r.extra["variance"]
        best = -1e9
        steps = 100
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                w = {"stocks": i / steps, "growth": j / steps, "bonds": 1 - (i + j) / steps}
                u = sum(MU[k] * w[k] for k in w) - lam * port_var(w)
                best = max(best, u)
        assert utility >= best - 1e-6

    def test_zero_risk_aversion_takes_best_return(self):
        r = portfolio(MU, COV, risk_aversion=0)
        assert r.x["growth"] == pytest.approx(1.0)

    def test_max_weight_caps_concentration(self):
        r = portfolio(MU, COV, risk_aversion=0, max_weight=0.5)
        assert r.x == pytest.approx({"stocks": 0.5, "growth": 0.5, "bonds": 0.0})

    def test_per_asset_bounds(self):
        r = portfolio(MU, COV, risk_aversion=1, min_weight={"bonds": 0.3}, max_weight={"growth": 0.2})
        assert r.x["bonds"] >= 0.3 - 1e-9 and r.x["growth"] <= 0.2 + 1e-9

    def test_volatility_cap(self):
        r = portfolio(MU, COV, max_volatility=0.12)
        assert r.extra["volatility"] == pytest.approx(0.12, abs=1e-6)
        # no portfolio on a grid with vol <= cap has a higher return
        steps = 100
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                w = {"stocks": i / steps, "growth": j / steps, "bonds": 1 - (i + j) / steps}
                if port_var(w) <= 0.12 ** 2:
                    assert sum(MU[k] * w[k] for k in w) <= r.score + 1e-6

    def test_loose_volatility_cap_is_max_return(self):
        r = portfolio(MU, COV, max_volatility=1.0)
        assert r.x["growth"] == pytest.approx(1.0)

    def test_volatility_cap_too_tight(self):
        with pytest.raises(InfeasibleError):
            portfolio(MU, COV, max_volatility=0.01)

    def test_mapping_covariance(self):
        names = list(MU)
        cov_map = {a: {b: COV[i][j] for j, b in enumerate(names)} for i, a in enumerate(names)}
        a = portfolio(MU, COV, risk_aversion=2)
        b = portfolio(MU, cov_map, risk_aversion=2)
        assert a.x == pytest.approx(b.x)

    def test_infeasible_bounds(self):
        with pytest.raises(InfeasibleError):
            portfolio(MU, COV, max_weight=0.3)

    def test_validation(self):
        with pytest.raises(ValueError):
            portfolio(MU, [[1, 0], [0, 1]])
        with pytest.raises(ValueError):
            portfolio(MU, [[0.04, 0.1, 0], [0.0, 0.09, 0], [0, 0, 0.01]])
        with pytest.raises(ValueError):
            portfolio(MU, COV, risk_aversion=-1)
        with pytest.raises(KeyError):
            portfolio(MU, COV, max_weight={"crypto": 0.1})
