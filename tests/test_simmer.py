import pytest
from currytruck import gradient_descent, numerical_gradient
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
