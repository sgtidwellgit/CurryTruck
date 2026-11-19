import pytest
from currytruck import (
    Recipe,
    Constraint,
    solve_csp,
    all_different,
    count_at_most,
    count_at_least,
    schedule_shifts,
    InfeasibleError,
    SearchBudgetExceeded,
)


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


class TestCountConstraints:
    def test_at_most(self):
        r = solve_csp({v: [0, 1] for v in "abcd"}, [count_at_most("abcd", 1, 2)], all_solutions=True)
        assert r.extra["n_solutions"] == 1 + 4 + 6
        assert all(sum(s.values()) <= 2 for s in r.extra["solutions"])

    def test_at_least(self):
        r = solve_csp({v: [0, 1] for v in "abcd"}, [count_at_least("abcd", 1, 3)], all_solutions=True)
        assert r.extra["n_solutions"] == 4 + 1

    def test_partial_pruning_saves_nodes(self):
        # at most 1 of 12 binary variables is 1: with partial checks, dead
        # branches are cut immediately instead of after all 12 are assigned
        vs = [f"v{i}" for i in range(12)]
        pruned = solve_csp({v: [1, 0] for v in vs}, [count_at_most(vs, 1, 1)], all_solutions=True)
        plain = Constraint(tuple(vs), lambda *vals: sum(vals) <= 1)
        unpruned = solve_csp({v: [1, 0] for v in vs}, [plain], all_solutions=True)
        assert pruned.extra["n_solutions"] == unpruned.extra["n_solutions"] == 13
        assert pruned.n_evals < unpruned.n_evals / 10

    def test_builder_methods(self):
        r = (Recipe().add_variable("a", [1, 2]).add_variable("b", [1, 2])
             .at_most(["a", "b"], 1, 1).at_least(["a", "b"], 2, 1))
        assert len(r.solve(all_solutions=True).extra["solutions"]) == 3


class TestAc3AndBudget:
    def test_ac3_same_answers(self):
        cons = [(("x", "y"), lambda x, y: x < y), (("y", "z"), lambda y, z: y < z)]
        doms = {v: range(6) for v in "xyz"}
        a = solve_csp(doms, cons, all_solutions=True)
        b = solve_csp(doms, cons, all_solutions=True, ac3=False)
        assert a.extra["solutions"] == b.extra["solutions"]
        assert a.extra["n_solutions"] == 20

    def test_ac3_detects_infeasibility_without_search(self):
        r_cons = [(("x", "y"), lambda x, y: x > y + 10)]
        with pytest.raises(InfeasibleError):
            solve_csp({"x": range(5), "y": range(5)}, r_cons)

    def test_budget_returns_best_so_far(self):
        r = solve_csp({v: range(10) for v in "abcd"}, [], objective=lambda s: sum(s.values()),
                      maximize=True, max_nodes=50)
        assert not r.converged and "max_nodes" in r.message
        assert r.n_evals == 50

    def test_budget_without_solution(self):
        # pigeonhole: 7 pigeons, 6 holes — infeasible, but only provable by deep search
        vs = [f"p{i}" for i in range(7)]
        with pytest.raises(SearchBudgetExceeded):
            solve_csp({v: range(6) for v in vs}, all_different(vs), max_nodes=100)

    def test_partial_check_proves_infeasibility_fast(self):
        vs = [f"q{i}" for i in range(8)]
        r_cons = [count_at_least(vs, 0, 9)]  # 9 zeros among 8 variables
        with pytest.raises(InfeasibleError):
            solve_csp({v: [1, 0] for v in vs}, r_cons, max_nodes=5)

    def test_bad_budget(self):
        with pytest.raises(ValueError):
            solve_csp({"a": [1]}, max_nodes=0)


class TestScheduleShifts:
    def test_docstring_example(self):
        r = schedule_shifts(["ann", "bob", "cat"], ["mon", "tue", "wed"],
                            availability={"ann": ["mon"]}, max_shifts=1)
        assert r.x == {"mon": ["ann"], "tue": ["bob"], "wed": ["cat"]}

    def test_coverage_and_distinct_staff(self):
        r = schedule_shifts(list("abcd"), ["am", "pm"], coverage={"am": 2, "pm": 3})
        assert len(r.x["am"]) == 2 and len(set(r.x["am"])) == 2
        assert len(r.x["pm"]) == 3 and len(set(r.x["pm"])) == 3

    def test_conflicts(self):
        r = schedule_shifts(["a", "b"], ["close", "open"], conflicts=[("close", "open")])
        assert r.x["close"] != r.x["open"]

    def test_max_and_min_shifts(self):
        shifts = [f"d{i}" for i in range(6)]
        r = schedule_shifts(["a", "b", "c"], shifts, max_shifts=2, min_shifts=2)
        assert r.extra["load"] == {"a": 2, "b": 2, "c": 2}

    def test_preferences_maximized(self):
        prefs = {"a": {"mon": 1, "tue": 5}, "b": {"mon": 5, "tue": 1}}
        r = schedule_shifts(["a", "b"], ["mon", "tue"], max_shifts=1, preferences=prefs)
        assert r.x == {"mon": ["b"], "tue": ["a"]}
        assert r.score == 10 and r.converged

    def test_week_schedule(self):
        days = ["mon", "tue", "wed", "thu", "fri"]
        shifts = [f"{d}-{p}" for d in days for p in ("am", "pm")]
        staff = ["ann", "bob", "cat", "dan", "eve"]
        conflicts = [(f"{d}-am", f"{d}-pm") for d in days]
        availability = {"ann": [s for s in shifts if s.endswith("am")]}
        r = schedule_shifts(staff, shifts, coverage=2, max_shifts=4, conflicts=conflicts,
                            availability=availability)
        for d in days:
            assert not set(r.x[f"{d}-am"]) & set(r.x[f"{d}-pm"])
        assert all(s.endswith("am") for s in shifts if "ann" in r.x[s])
        assert all(n <= 4 for n in r.extra["load"].values())

    def test_infeasible(self):
        with pytest.raises(InfeasibleError):
            schedule_shifts(["a"], ["mon", "tue"], max_shifts=1)

    def test_unknown_names(self):
        with pytest.raises(KeyError):
            schedule_shifts(["a"], ["mon"], availability={"zed": ["mon"]})
        with pytest.raises(KeyError):
            schedule_shifts(["a"], ["mon"], availability={"a": ["sun"]})
        with pytest.raises(KeyError):
            schedule_shifts(["a"], ["mon"], conflicts=[("mon", "sun")])
        with pytest.raises(ValueError):
            schedule_shifts(["a", "a"], ["mon"])
