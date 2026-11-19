import pytest
from currytruck import Result, rank, best, convergence, report, leaderboard, EmptySpaceError
from currytruck import plot_convergence, plot_pareto


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


class TestPlotting:
    @pytest.fixture(autouse=True)
    def _agg(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_convergence_single(self):
        ax = plot_convergence(make(1, history=[5, 3, 1]))
        assert len(ax.lines) == 1
        assert list(ax.lines[0].get_ydata()) == [5, 3, 1]
        assert ax.get_legend() is None

    def test_convergence_many_with_labels_and_log(self):
        runs = [make(1, method="a", history=[4, 2, 1]), make(2, method="b", history=[8, 2])]
        ax = plot_convergence(runs, log=True)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
        assert ax.get_yscale() == "log"

    def test_convergence_onto_existing_axes(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        assert plot_convergence(make(1, history=[1]), ax=ax) is ax

    def test_convergence_label_mismatch(self):
        with pytest.raises(ValueError):
            plot_convergence([make(1)], labels=["a", "b"])

    def test_pareto(self):
        pts = [(1, 5), (2, 2), (3, 3), (5, 1)]
        ax = plot_pareto(pts, front=[0, 1, 3], labels=("cost", "time"))
        assert ax.get_xlabel() == "cost"
        assert list(ax.lines[0].get_xdata()) == [1, 2, 5]

    def test_pareto_needs_two_objectives(self):
        with pytest.raises(ValueError):
            plot_pareto([(1, 2, 3)])
