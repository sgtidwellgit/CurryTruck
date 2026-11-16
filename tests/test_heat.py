import math
import random

import pytest
from currytruck import (
    simulated_annealing,
    genetic_algorithm,
    swap_neighbor,
    reverse_neighbor,
    InvalidHeatLevel,
)
from currytruck.taste import rastrigin, sphere

# 8 delivery stops on a circle — the optimal route visits them in angular order
STOPS = [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)]
OPTIMAL_TOUR = 8 * 2 * math.sin(math.pi / 8)


def tour_length(order):
    return sum(math.dist(STOPS[order[i]], STOPS[order[(i + 1) % len(order)]]) for i in range(len(order)))


class TestNeighbors:
    def test_swap_is_permutation(self):
        rng = random.Random(0)
        x = list(range(10))
        y = swap_neighbor(x, rng)
        assert sorted(y) == x and y != x
        assert x == list(range(10))

    def test_reverse_is_permutation(self):
        rng = random.Random(0)
        x = list(range(10))
        y = reverse_neighbor(x, rng)
        assert sorted(y) == x

    def test_short_inputs(self):
        rng = random.Random(0)
        assert swap_neighbor([1], rng) == [1]
        assert reverse_neighbor([], rng) == []


class TestSimulatedAnnealing:
    def test_continuous_sphere(self):
        r = simulated_annealing(sphere, [4.0, -3.0], bounds=[(-5, 5), (-5, 5)], seed=1, max_iter=20_000)
        assert r.score < 0.01

    def test_escapes_local_minima_rastrigin(self):
        r = simulated_annealing(rastrigin, [4.5, 4.5], bounds=[(-5.12, 5.12)] * 2, heat=4,
                                seed=3, max_iter=30_000)
        assert r.score < 1.5

    def test_tsp_with_reverse_neighbor(self):
        scrambled = [0, 4, 1, 5, 2, 6, 3, 7]
        r = simulated_annealing(tour_length, scrambled, neighbor=reverse_neighbor, seed=0, max_iter=5000)
        assert r.score == pytest.approx(OPTIMAL_TOUR)
        assert sorted(r.x) == list(range(8))

    def test_maximize(self):
        r = simulated_annealing(lambda v: -((v[0] - 1) ** 2), [0.0], bounds=[(-3, 3)],
                                maximize=True, seed=0, max_iter=5000)
        assert r.x[0] == pytest.approx(1.0, abs=0.05)
        assert r.score <= 0

    def test_reproducible(self):
        a = simulated_annealing(sphere, [2.0], seed=7, max_iter=500)
        b = simulated_annealing(sphere, [2.0], seed=7, max_iter=500)
        assert a.x == b.x

    def test_history_monotone(self):
        r = simulated_annealing(sphere, [3.0], seed=0, max_iter=1000)
        assert all(b <= a for a, b in zip(r.history, r.history[1:]))

    def test_heat_affects_start_temperature(self):
        cold = simulated_annealing(sphere, [3.0], heat=1, seed=0, max_iter=10)
        hot = simulated_annealing(sphere, [3.0], heat=5, seed=0, max_iter=10)
        assert hot.extra["start_temperature"] > cold.extra["start_temperature"]

    @pytest.mark.parametrize("heat", [0, 6, "3"])
    def test_bad_heat(self, heat):
        with pytest.raises(InvalidHeatLevel):
            simulated_annealing(sphere, [1.0], heat=heat)

    def test_bounds_length_mismatch(self):
        with pytest.raises(ValueError):
            simulated_annealing(sphere, [1.0, 2.0], bounds=[(0, 1)])


class TestGeneticAlgorithm:
    def test_vector_sphere(self):
        r = genetic_algorithm(sphere, [(-5, 5)] * 3, seed=0, generations=300)
        assert r.score < 0.05

    def test_integer_mode(self):
        r = genetic_algorithm(lambda v: (v[0] - 3) ** 2 + (v[1] + 2) ** 2, [(-10, 10)] * 2,
                              integer=True, seed=0, generations=100)
        assert r.x == [3, -2]
        assert all(isinstance(g, int) for g in r.x)

    def test_permutation_tsp(self):
        r = genetic_algorithm(tour_length, permutation=range(8), seed=0, generations=300, population=60)
        assert r.score == pytest.approx(OPTIMAL_TOUR)
        assert sorted(r.x) == list(range(8))

    def test_maximize(self):
        r = genetic_algorithm(lambda v: -abs(v[0] - 2), [(0, 5)], maximize=True, seed=0)
        assert r.x[0] == pytest.approx(2.0, abs=0.05)

    def test_patience_stops_early(self):
        r = genetic_algorithm(lambda v: 0.0, [(0, 1)], patience=5, generations=1000, seed=0)
        assert r.converged and r.n_iter == 5

    def test_respects_bounds(self):
        seen = []

        def f(v):
            seen.append(v[0])
            return -v[0]

        genetic_algorithm(f, [(1, 2)], heat=5, seed=0, generations=30)
        assert all(1 <= g <= 2 for g in seen)

    def test_history_monotone(self):
        r = genetic_algorithm(sphere, [(-5, 5)] * 2, seed=1, generations=50)
        assert all(b <= a for a, b in zip(r.history, r.history[1:]))

    def test_mutation_rate_from_heat(self):
        r = genetic_algorithm(sphere, [(-1, 1)], heat=5, generations=1, seed=0)
        assert r.extra["mutation_rate"] == 0.4

    def test_requires_one_mode(self):
        with pytest.raises(ValueError):
            genetic_algorithm(sphere)
        with pytest.raises(ValueError):
            genetic_algorithm(sphere, [(0, 1)], permutation=[1, 2])

    def test_bad_heat(self):
        with pytest.raises(InvalidHeatLevel):
            genetic_algorithm(sphere, [(0, 1)], heat=9)

    def test_bad_population(self):
        with pytest.raises(ValueError):
            genetic_algorithm(sphere, [(0, 1)], population=1)
