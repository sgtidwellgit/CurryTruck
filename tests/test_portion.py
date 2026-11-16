import pytest
from currytruck import linprog, allocate, InfeasibleError, UnboundedError


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
