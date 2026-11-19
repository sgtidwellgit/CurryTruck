import math

import pytest
from currytruck import dominates, pareto_front, pareto_rank, crowding_distance, pareto_search, nsga2
from currytruck import EmptySpaceError


class TestDominates:
    def test_equal_points_do_not_dominate(self):
        assert not dominates([1, 1], [1, 1])

    def test_maximize(self):
        assert dominates([3, 3], [2, 3], maximize=True)

    def test_mixed_senses(self):
        # minimize cost, maximize quality
        assert dominates([5, 9], [6, 8], maximize=[False, True])
        assert not dominates([5, 7], [6, 8], maximize=[False, True])

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            dominates([1], [1, 2])

    def test_bad_senses_length(self):
        with pytest.raises(ValueError):
            dominates([1, 2], [1, 2], maximize=[True])


class TestFront:
    def test_maximize_front(self):
        pts = [(1, 5), (2, 2), (3, 3), (5, 1)]
        assert pareto_front(pts, maximize=True) == [0, 2, 3]

    def test_duplicates_both_kept(self):
        assert pareto_front([(1, 1), (1, 1), (2, 2)]) == [0, 1]

    def test_rank_layers(self):
        pts = [(1, 1), (2, 2), (3, 3)]
        assert pareto_rank(pts) == [0, 1, 2]

    def test_empty(self):
        assert pareto_front([]) == []
        assert pareto_rank([]) == []


class TestCrowding:
    def test_boundaries_infinite(self):
        d = crowding_distance([(0, 4), (1, 2), (4, 0)])
        assert d[0] == d[2] == float("inf")
        assert d[1] == pytest.approx(2.0)

    def test_empty(self):
        assert crowding_distance([]) == []


class TestParetoSearch:
    def test_portfolio(self):
        # minimize risk, maximize return
        portfolios = [
            {"name": "bonds", "risk": 1, "ret": 2},
            {"name": "mixed", "risk": 3, "ret": 5},
            {"name": "bad", "risk": 4, "ret": 4},
            {"name": "stocks", "risk": 6, "ret": 9},
        ]
        front = pareto_search([lambda p: p["risk"], lambda p: p["ret"]], portfolios,
                              maximize=[False, True])
        assert [p["name"] for p, _ in front] == ["bonds", "mixed", "stocks"]
        assert front[0][1] == (1.0, 2.0)

    def test_top_keeps_spread(self):
        cands = [(0, 10), (1, 9), (2, 8), (5, 5), (10, 0)]
        front = pareto_search([lambda c: c[0], lambda c: c[1]], cands, top=3)
        assert [c for c, _ in front][0] == (0, 10) and [c for c, _ in front][-1] == (10, 0)
        assert len(front) == 3

    def test_empty(self):
        with pytest.raises(EmptySpaceError):
            pareto_search([lambda c: c], [])


def zdt1(v):
    g = 1 + 9 * sum(v[1:]) / (len(v) - 1)
    f1 = v[0]
    return f1, g * (1 - math.sqrt(f1 / g))


class TestNsga2:
    def test_zdt1_front(self):
        front = nsga2([lambda v: zdt1(v)[0], lambda v: zdt1(v)[1]], [(0, 1)] * 5,
                      population=60, generations=150, seed=1)
        # true front: f2 = 1 - sqrt(f1)
        errors = [abs(s[1] - (1 - math.sqrt(s[0]))) for _, s in front]
        assert sum(errors) / len(errors) < 0.05
        f1s = [s[0] for _, s in front]
        assert min(f1s) < 0.1 and max(f1s) > 0.9  # spread across the front

    def test_front_is_non_dominated_and_sorted(self):
        front = nsga2([lambda v: v[0], lambda v: 1 - v[0] + v[1] ** 2], [(0, 1), (-1, 1)], seed=0,
                      generations=30)
        scores = [s for _, s in front]
        assert pareto_front(scores) == list(range(len(scores)))
        assert [s[0] for s in scores] == sorted(s[0] for s in scores)

    def test_mixed_senses(self):
        # minimize cost x, maximize quality sqrt(x): the whole range is a trade-off
        front = nsga2([lambda v: v[0], lambda v: math.sqrt(v[0])], [(0, 4)],
                      maximize=[False, True], seed=0, generations=40)
        xs = [x[0] for x, _ in front]
        assert min(xs) < 0.2 and max(xs) > 3.8

    def test_reproducible(self):
        objs = [lambda v: v[0] ** 2, lambda v: (v[0] - 1) ** 2]
        assert nsga2(objs, [(-2, 2)], seed=5, generations=10) == nsga2(objs, [(-2, 2)], seed=5,
                                                                       generations=10)

    def test_callback(self):
        gens = []
        nsga2([lambda v: v[0], lambda v: -v[0]], [(0, 1)], seed=0,
              callback=lambda g, front: gens.append(g) or g == 3)
        assert gens == [1, 2, 3]

    def test_odd_population_rounded_up(self):
        front = nsga2([lambda v: v[0], lambda v: 1 - v[0]], [(0, 1)], population=7, generations=5, seed=0)
        assert len(front) <= 8

    def test_errors(self):
        with pytest.raises(ValueError):
            nsga2([], [(0, 1)])
        with pytest.raises(ValueError):
            nsga2([lambda v: v[0]], [])
        with pytest.raises(ValueError):
            nsga2([lambda v: v[0]], [(1, 0)])
        with pytest.raises(ValueError):
            nsga2([lambda v: v[0]], [(0, 1)], population=2)
