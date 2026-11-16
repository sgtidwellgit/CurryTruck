import pytest
from currytruck import dominates, pareto_front, pareto_rank, crowding_distance, pareto_search
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
