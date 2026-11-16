import pytest
from currytruck import Result, rank, best, convergence, report, leaderboard, EmptySpaceError


def make(score, maximize=False, method="m", history=None):
    return Result(x=None, score=score, maximize=maximize, method=method, history=history or [])


class TestRank:
    def test_minimize(self):
        rs = [make(3), make(1), make(2)]
        assert [r.score for r in rank(rs)] == [1, 2, 3]

    def test_maximize_top(self):
        rs = [make(3, True), make(1, True), make(2, True)]
        assert [r.score for r in rank(rs, top=2)] == [3, 2]

    def test_mixed_senses_refused(self):
        with pytest.raises(ValueError):
            rank([make(1), make(2, True)])

    def test_empty(self):
        assert rank([]) == []

    def test_best(self):
        assert best([make(3), make(1)]).score == 1
        with pytest.raises(EmptySpaceError):
            best([])


class TestConvergence:
    def test_maximize_stats(self):
        r = make(10, True, history=[2, 6, 10, 10])
        s = convergence(r)
        assert s["improvement"] == 8
        assert s["improvement_pct"] == 400.0
        assert s["iteration_of_best"] == 2
        assert s["plateau"] == 1

    def test_zero_initial(self):
        s = convergence(make(-1, history=[0, -1]))
        assert s["improvement_pct"] is None

    def test_empty_history(self):
        s = convergence(make(1))
        assert s["initial"] is None and s["plateau"] == 0

    def test_tolerance(self):
        s = convergence(make(1, history=[5, 1, 1 - 1e-12]))
        assert s["plateau"] == 1


class TestText:
    def test_report_message(self):
        r = Result(x=[1], score=2.0, method="gd", message="done", history=[3.0, 2.0])
        text = report(r)
        assert text.startswith("gd (minimize)")
        assert "message    : done" in text

    def test_report_truncates_long_solution(self):
        r = Result(x=list(range(1000)), score=0.0)
        assert "..." in report(r)

    def test_leaderboard_top(self):
        text = leaderboard([make(1, method="a"), make(2, method="b"), make(3, method="c")], top=2)
        lines = text.splitlines()
        assert len(lines) == 3
        assert "a" in lines[1] and "b" in lines[2]
