import pytest
from currytruck import grid_search, random_search, EmptySpaceError


class TestGridSearch:
    def test_finds_minimum(self):
        r = grid_search(
            lambda x, y: (x - 2) ** 2 + (y + 1) ** 2,
            {"x": range(-5, 6), "y": range(-5, 6)},
        )
        assert r.x == {"x": 2, "y": -1}
        assert r.score == 0.0

    def test_maximize(self):
        r = grid_search(lambda x: -((x - 3) ** 2), {"x": range(10)}, maximize=True)
        assert r.x == {"x": 3}
        assert r.maximize is True

    def test_evaluates_full_product(self):
        r = grid_search(lambda a, b, c: a + b + c, {"a": [1, 2], "b": [1, 2, 3], "c": [0, 1]})
        assert r.n_evals == 12
        assert len(r.extra["trials"]) == 12

    def test_keep_trials_false(self):
        r = grid_search(lambda x: x, {"x": [1, 2]}, keep_trials=False)
        assert r.extra == {}

    def test_history_is_best_so_far(self):
        r = grid_search(lambda x: x, {"x": [5, 3, 4, 1, 2]})
        assert r.history == [5, 3, 3, 1, 1]

    def test_non_numeric_values(self):
        scores = {"rbf": 0.1, "linear": 0.5}
        r = grid_search(lambda kernel: scores[kernel], {"kernel": ["linear", "rbf"]})
        assert r.x == {"kernel": "rbf"}

    def test_empty_space_raises(self):
        with pytest.raises(EmptySpaceError):
            grid_search(lambda: 0, {})

    def test_empty_values_raises(self):
        with pytest.raises(EmptySpaceError):
            grid_search(lambda x: x, {"x": []})


class TestRandomSearch:
    def test_reproducible_with_seed(self):
        def f(x, y):
            return x * x + y * y

        space = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
        a = random_search(f, space, n_iter=50, seed=1)
        b = random_search(f, space, n_iter=50, seed=1)
        assert a.x == b.x and a.score == b.score

    def test_gets_close_to_minimum(self):
        r = random_search(lambda x: (x - 1.5) ** 2, {"x": (-10.0, 10.0)}, n_iter=2000, seed=0)
        assert abs(r.x["x"] - 1.5) < 0.05

    def test_int_range_samples_ints(self):
        r = random_search(lambda n: n, {"n": (1, 10)}, n_iter=100, seed=0)
        assert all(isinstance(p["n"], int) and 1 <= p["n"] <= 10 for p, _ in r.extra["trials"])

    def test_choice_list(self):
        r = random_search(lambda c: len(c), {"c": ["aaa", "b", "cc"]}, n_iter=30, seed=0)
        assert r.x == {"c": "b"}

    def test_callable_sampler(self):
        r = random_search(lambda v: v, {"v": lambda rng: 7}, n_iter=3)
        assert r.x == {"v": 7}

    def test_n_iter_count(self):
        r = random_search(lambda x: x, {"x": (0.0, 1.0)}, n_iter=25)
        assert r.n_evals == 25 and len(r.history) == 25

    def test_bad_n_iter(self):
        with pytest.raises(ValueError):
            random_search(lambda x: x, {"x": (0, 1)}, n_iter=0)

    def test_reversed_bounds(self):
        with pytest.raises(ValueError):
            random_search(lambda x: x, {"x": (5, 1)})


def slow_square(x):
    return (x - 3) ** 2


class TestParallelAndCallbacks:
    def test_thread_workers_same_result(self):
        space = {"x": (-10.0, 10.0), "y": [1, 2, 3]}

        def f(x, y):
            return (x - y) ** 2

        serial = random_search(f, space, n_iter=60, seed=4)
        parallel = random_search(f, space, n_iter=60, seed=4, workers=4)
        assert serial.x == parallel.x and serial.history == parallel.history

    def test_process_backend(self):
        r = grid_search(slow_square, {"x": range(7)}, workers=2, backend="process")
        assert r.x == {"x": 3}
        assert r.n_evals == 7

    def test_callback_stops(self):
        seen = []

        def cb(i, params, score):
            seen.append(i)
            return i == 5

        r = grid_search(lambda x: x, {"x": range(100)}, callback=cb)
        assert r.n_evals == 5 and seen[-1] == 5
        assert not r.converged and r.message == "Stopped by callback"

    def test_callback_stops_parallel_run_early(self):
        r = random_search(lambda x: x, {"x": (0.0, 1.0)}, n_iter=1000, workers=2,
                          callback=lambda i, p, s: i >= 10)
        assert r.n_evals == 10

    def test_bad_backend(self):
        with pytest.raises(ValueError):
            grid_search(lambda x: x, {"x": [1]}, backend="gpu")

    def test_bad_workers(self):
        with pytest.raises(ValueError):
            grid_search(lambda x: x, {"x": [1]}, workers=0)
