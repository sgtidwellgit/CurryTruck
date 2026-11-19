import pytest
from currytruck import gradient_descent, adam, rmsprop, numerical_gradient
from currytruck.taste import rosenbrock, sphere


class TestNumericalGradient:
    def test_quadratic(self):
        g = numerical_gradient(lambda v: v[0] ** 2 + 2 * v[1] ** 2, [1.0, -2.0])
        assert g == pytest.approx([2.0, -8.0], rel=1e-5)

    def test_does_not_mutate(self):
        x = [1.0, 2.0]
        numerical_gradient(sphere, x)
        assert x == [1.0, 2.0]


class TestGradientDescent:
    def test_sphere(self):
        r = gradient_descent(sphere, [3.0, -4.0], lr=0.1)
        assert r.x == pytest.approx([0.0, 0.0], abs=1e-4)
        assert r.converged

    def test_analytic_gradient(self):
        r = gradient_descent(sphere, [3.0, -4.0], grad=lambda v: [2 * x for x in v], lr=0.1)
        assert r.x == pytest.approx([0.0, 0.0], abs=1e-4)

    def test_maximize(self):
        r = gradient_descent(lambda v: -((v[0] - 2) ** 2) + 5, [0.0], lr=0.1, maximize=True)
        assert r.x[0] == pytest.approx(2.0, abs=1e-4)
        assert r.score == pytest.approx(5.0)
        assert r.maximize

    def test_momentum_rosenbrock(self):
        r = gradient_descent(rosenbrock, [-1.2, 1.0], lr=1e-3, momentum=0.9, max_iter=50_000)
        assert r.x == pytest.approx([1.0, 1.0], abs=1e-2)

    def test_bounds_projection(self):
        r = gradient_descent(lambda v: (v[0] - 5) ** 2, [0.0], lr=0.1, bounds=[(None, 2.0)])
        assert r.x[0] == pytest.approx(2.0)

    def test_max_iter_not_converged(self):
        r = gradient_descent(sphere, [100.0], lr=1e-6, max_iter=5)
        assert not r.converged and r.n_iter == 5

    def test_divergence_reported(self):
        r = gradient_descent(lambda v: v[0] ** 4, [10.0], lr=10.0, max_iter=200)
        assert not r.converged

    def test_history_starts_at_x0(self):
        r = gradient_descent(sphere, [2.0], lr=0.1)
        assert r.history[0] == 4.0

    @pytest.mark.parametrize("kw", [{"lr": 0}, {"momentum": 1.0}, {"bounds": [(0, 1), (0, 1)]}])
    def test_bad_args(self, kw):
        with pytest.raises(ValueError):
            gradient_descent(sphere, [1.0], **kw)


def ill_conditioned(v):
    return (v[0] - 3) ** 2 + 1000 * (v[1] + 1) ** 2


class TestAdaptiveMethods:
    def test_adam_ill_conditioned(self):
        r = adam(ill_conditioned, [0.0, 0.0], lr=0.05, max_iter=5000)
        assert r.x == pytest.approx([3.0, -1.0], abs=1e-2)
        assert r.method == "adam"

    def test_rmsprop(self):
        r = rmsprop(ill_conditioned, [0.0, 0.0], lr=0.01, max_iter=5000)
        assert r.x == pytest.approx([3.0, -1.0], abs=2e-2)

    def test_adam_maximize_and_bounds(self):
        r = adam(lambda v: -((v[0] - 5) ** 2), [0.0], lr=0.1, bounds=[(None, 2.0)],
                 maximize=True, max_iter=2000)
        assert r.x[0] == pytest.approx(2.0)

    @pytest.mark.parametrize("kw", [{"lr": -1}, {"beta1": 1.0}, {"beta2": 1.5}])
    def test_adam_bad_args(self, kw):
        with pytest.raises(ValueError):
            adam(sphere, [1.0], **kw)

    def test_rmsprop_bad_rho(self):
        with pytest.raises(ValueError):
            rmsprop(sphere, [1.0], rho=1.0)


class TestLineSearch:
    def test_rescues_too_large_lr(self):
        plain = gradient_descent(ill_conditioned, [0.0, 0.0], lr=0.5, max_iter=200)
        searched = gradient_descent(ill_conditioned, [0.0, 0.0], lr=0.5, line_search=True,
                                    max_iter=20_000)
        assert not plain.converged
        assert searched.x == pytest.approx([3.0, -1.0], abs=1e-4)

    def test_monotone_decrease(self):
        r = gradient_descent(rosenbrock, [-1.2, 1.0], lr=1.0, line_search=True, max_iter=500)
        assert all(b <= a + 1e-12 for a, b in zip(r.history, r.history[1:]))

    def test_not_with_momentum(self):
        with pytest.raises(ValueError):
            gradient_descent(sphere, [1.0], line_search=True, momentum=0.5)


class TestCallback:
    def test_stops(self):
        r = gradient_descent(sphere, [10.0], lr=0.01, callback=lambda i, x, s: i == 3)
        assert r.n_iter == 3 and r.message == "Stopped by callback" and not r.converged

    def test_receives_user_sense_score(self):
        scores = []
        gradient_descent(lambda v: -(v[0] ** 2), [1.0], lr=0.1, maximize=True, max_iter=5,
                         callback=lambda i, x, s: scores.append(s))
        assert scores and all(s <= 0 for s in scores)
