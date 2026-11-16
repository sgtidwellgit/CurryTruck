import math

import pytest
from currytruck import weighted_sum, minmax_normalize, penalized, counted, score_all
from currytruck.taste import ackley, rastrigin, rosenbrock, sphere


class TestWeightedSum:
    def test_basic(self):
        assert weighted_sum({"a": 2, "b": 3}, {"a": 1, "b": 2}) == 8.0

    def test_extra_scores_ignored(self):
        assert weighted_sum({"a": 2, "z": 100}, {"a": 1}) == 2.0

    def test_missing_score(self):
        with pytest.raises(KeyError):
            weighted_sum({"a": 1}, {"b": 1})


class TestMinmaxNormalize:
    def test_invert(self):
        assert minmax_normalize([10, 20, 30], invert=True) == [1.0, 0.5, 0.0]

    def test_constant(self):
        assert minmax_normalize([5, 5]) == [0.0, 0.0]

    def test_empty(self):
        assert minmax_normalize([]) == []


class TestPenalized:
    def test_squared(self):
        f = penalized(lambda x: 0.0, [lambda x: x - 1], weight=1)  # x <= 1
        assert f(3) == 4.0
        assert f(0) == 0.0

    def test_linear(self):
        f = penalized(lambda x: 0.0, [lambda x: x - 1], weight=1, squared=False)
        assert f(3) == 2.0

    def test_maximize_subtracts(self):
        f = penalized(lambda x: 10.0, [lambda x: x], weight=1, maximize=True)
        assert f(2) == 6.0

    def test_kwargs_passthrough(self):
        f = penalized(lambda x, y: x + y, [lambda x, y: x - y], weight=1)
        assert f(x=1, y=2) == 3.0

    def test_preserves_name(self):
        def my_cost(x):
            return x

        assert penalized(my_cost, []).__name__ == "my_cost"


class TestCounted:
    def test_reset(self):
        g = counted(lambda: None)
        g()
        g()
        assert g.calls == 2
        g.reset()
        assert g.calls == 0


class TestScoreAll:
    def test_maximize(self):
        assert score_all(lambda x: x, [3, 1, 2], maximize=True) == [(3, 3.0), (2, 2.0), (1, 1.0)]


class TestBenchmarks:
    @pytest.mark.parametrize("f,x", [(sphere, [0, 0]), (rosenbrock, [1, 1, 1]),
                                     (rastrigin, [0, 0]), (ackley, [0, 0])])
    def test_global_minimum_is_zero(self, f, x):
        assert f(x) == pytest.approx(0.0, abs=1e-12)

    def test_values_positive_elsewhere(self):
        for f in (sphere, rosenbrock, rastrigin, ackley):
            assert f([0.5, -0.7]) > 0

    def test_rastrigin_known_value(self):
        assert rastrigin([1.0]) == pytest.approx(1.0)

    def test_ackley_far_point(self):
        # cos(2*pi*30) == 1, so the exp(cos) and e terms cancel
        assert ackley([30.0, 30.0]) == pytest.approx(20 - 20 * math.exp(-0.2 * 30), abs=1e-9)
