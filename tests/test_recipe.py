import pytest
from currytruck import Recipe, Constraint, solve_csp, all_different, InfeasibleError


class TestSolveCsp:
    def test_simple(self):
        r = solve_csp({"a": [1, 2, 3], "b": [1, 2, 3]}, [(("a", "b"), lambda a, b: a + b == 5)])
        assert r.x["a"] + r.x["b"] == 5
        assert r.converged

    def test_unary_constraint_string_var(self):
        r = solve_csp({"a": range(10)}, [("a", lambda a: a > 7)])
        assert r.x == {"a": 8}

    def test_map_coloring(self):
        regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
        borders = [("WA", "NT"), ("WA", "SA"), ("NT", "SA"), ("NT", "Q"), ("SA", "Q"),
                   ("SA", "NSW"), ("SA", "V"), ("Q", "NSW"), ("NSW", "V")]
        colors = ["red", "green", "blue"]
        r = solve_csp({v: colors for v in regions}, [(b, lambda x, y: x != y) for b in borders])
        for a, b in borders:
            assert r.x[a] != r.x[b]

    def test_n_queens_count(self):
        n = 6
        cols = [f"q{i}" for i in range(n)]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                cons.append(((cols[i], cols[j]), lambda a, b, d=d: a != b and abs(a - b) != d))
        r = solve_csp({c: range(n) for c in cols}, cons, all_solutions=True)
        assert r.extra["n_solutions"] == 4
        assert len(r.extra["solutions"]) == 4

    def test_objective_finds_best(self):
        r = solve_csp(
            {"x": range(10), "y": range(10)},
            [(("x", "y"), lambda x, y: x + y == 9)],
            objective=lambda s: s["x"] * s["y"],
            maximize=True,
        )
        assert r.score == 20
        assert r.extra["n_solutions"] == 10

    def test_limit(self):
        r = solve_csp({"x": range(100)}, [], all_solutions=True, limit=5)
        assert len(r.extra["solutions"]) == 5
        assert not r.converged

    def test_infeasible(self):
        with pytest.raises(InfeasibleError):
            solve_csp({"a": [1], "b": [1]}, all_different(["a", "b"]))

    def test_infeasible_unary(self):
        with pytest.raises(InfeasibleError):
            solve_csp({"a": [1, 2]}, [("a", lambda a: a > 5)])

    def test_unknown_variable(self):
        with pytest.raises(KeyError):
            solve_csp({"a": [1]}, [(("a", "z"), lambda a, z: True)])

    def test_three_variable_constraint(self):
        r = solve_csp(
            {v: range(1, 10) for v in "abc"},
            [(("a", "b", "c"), lambda a, b, c: a * b * c == 30 and a < b < c)],
            all_solutions=True,
        )
        assert sorted(tuple(s.values()) for s in r.extra["solutions"]) == [(1, 5, 6), (2, 3, 5)]


class TestAllDifferent:
    def test_pairs(self):
        cons = all_different(["a", "b", "c"])
        assert len(cons) == 3
        assert all(isinstance(c, Constraint) for c in cons)

    def test_permutations(self):
        r = solve_csp({v: [1, 2, 3] for v in "abc"}, all_different("abc"), all_solutions=True)
        assert r.extra["n_solutions"] == 6


class TestRecipeBuilder:
    def test_shift_scheduling(self):
        # three employees, three shifts; Ann can't work nights, Bob prefers mornings
        shifts = ["morning", "evening", "night"]
        r = (
            Recipe()
            .add_variable("ann", shifts)
            .add_variable("bob", shifts)
            .add_variable("cat", shifts)
            .all_different(["ann", "bob", "cat"])
            .add_constraint("ann", lambda s: s != "night")
        )
        result = r.solve(objective=lambda a: a["bob"] == "morning", maximize=True)
        assert result.x["bob"] == "morning"
        assert result.x["ann"] == "evening"
        assert result.x["cat"] == "night"

    def test_duplicate_variable(self):
        with pytest.raises(ValueError):
            Recipe().add_variable("a", [1]).add_variable("a", [2])

    def test_repr(self):
        assert repr(Recipe().add_variable("a", [1])) == "Recipe(variables=['a'], constraints=0)"
