import math
import random

import pytest
from currytruck import (
    simulated_annealing,
    genetic_algorithm,
    particle_swarm,
    tabu_search,
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


def shifted_sphere(v):
    return sum((x - 1) ** 2 for x in v)


class TestTabuSearch:
    def test_sorts_permutation(self):
        def disorder(p):
            return sum(abs(v - i) for i, v in enumerate(p))

        r = tabu_search(disorder, [5, 3, 1, 0, 4, 2], neighbor=swap_neighbor, seed=1, max_iter=200)
        assert r.x == [0, 1, 2, 3, 4, 5] and r.score == 0

    def test_tsp(self):
        r = tabu_search(tour_length, [0, 4, 1, 5, 2, 6, 3, 7], neighbor=reverse_neighbor,
                        seed=0, max_iter=300)
        assert r.score == pytest.approx(OPTIMAL_TOUR)

    def test_default_numeric_neighbor(self):
        r = tabu_search(sphere, [3.0, -2.0], bounds=[(-5, 5)] * 2, seed=0, max_iter=500)
        assert r.score < 0.05

    def test_patience(self):
        r = tabu_search(lambda x: 0.0, [1, 2, 3], neighbor=swap_neighbor, patience=4, seed=0)
        assert r.converged and r.n_iter == 4

    def test_maximize(self):
        r = tabu_search(lambda p: p[0] * 10 + p[1], [0, 1, 2], neighbor=swap_neighbor,
                        maximize=True, seed=0, max_iter=50)
        assert r.x[:2] == [2, 1]

    def test_tenure_zero_allowed(self):
        r = tabu_search(sphere, [1.0], tenure=0, seed=0, max_iter=20)
        assert r.n_iter == 20

    def test_bad_args(self):
        with pytest.raises(ValueError):
            tabu_search(sphere, [1.0], n_candidates=0)
        with pytest.raises(ValueError):
            tabu_search(sphere, [1.0], tenure=-1)


class TestParticleSwarm:
    def test_sphere(self):
        r = particle_swarm(shifted_sphere, [(-5, 5)] * 4, seed=0, max_iter=300)
        assert r.score < 1e-6

    def test_rastrigin_global(self):
        r = particle_swarm(rastrigin, [(-5.12, 5.12)] * 2, swarm=40, seed=2, max_iter=300)
        assert r.score < 1e-3

    def test_stays_in_bounds(self):
        seen = []

        def f(v):
            seen.append(v[0])
            return -v[0]

        particle_swarm(f, [(1, 2)], seed=0, max_iter=30, heat=5)
        assert all(1 <= x <= 2 for x in seen)

    def test_maximize(self):
        r = particle_swarm(lambda v: -abs(v[0] - 2), [(0, 5)], maximize=True, seed=0)
        assert r.x[0] == pytest.approx(2.0, abs=1e-3)

    def test_patience(self):
        r = particle_swarm(lambda v: 0.0, [(0, 1)], patience=3, seed=0)
        assert r.converged and r.n_iter == 3

    def test_heat_sets_inertia(self):
        assert particle_swarm(sphere, [(0, 1)], heat=1, max_iter=1).extra["inertia"] == 0.4

    def test_bad_args(self):
        with pytest.raises(ValueError):
            particle_swarm(sphere, [])
        with pytest.raises(ValueError):
            particle_swarm(sphere, [(2, 1)])
        with pytest.raises(InvalidHeatLevel):
            particle_swarm(sphere, [(0, 1)], heat=0)


class TestParallelEvaluation:
    def test_ga_workers_identical(self):
        a = genetic_algorithm(shifted_sphere, [(-5, 5)] * 3, seed=3, generations=40)
        b = genetic_algorithm(shifted_sphere, [(-5, 5)] * 3, seed=3, generations=40, workers=4)
        assert a.x == b.x and a.history == b.history and a.n_evals == b.n_evals

    def test_pso_workers_identical(self):
        a = particle_swarm(shifted_sphere, [(-5, 5)] * 2, seed=3, max_iter=40)
        b = particle_swarm(shifted_sphere, [(-5, 5)] * 2, seed=3, max_iter=40, workers=3)
        assert a.x == b.x and a.history == b.history

    def test_ga_process_backend(self):
        r = genetic_algorithm(shifted_sphere, [(-5, 5)] * 2, seed=0, generations=5,
                              population=10, workers=2, backend="process")
        assert r.n_iter == 5

    def test_bad_backend(self):
        with pytest.raises(ValueError):
            genetic_algorithm(sphere, [(0, 1)], backend="cluster")


class TestCallbacks:
    def test_sa(self):
        r = simulated_annealing(sphere, [3.0], seed=0, callback=lambda i, x, s: i == 7)
        assert r.n_iter == 7 and r.message == "Stopped by callback"

    def test_ga(self):
        r = genetic_algorithm(sphere, [(-1, 1)], seed=0, callback=lambda g, x, s: g == 2)
        assert r.n_iter == 2

    def test_pso_gets_best_so_far(self):
        scores = []
        particle_swarm(sphere, [(-1, 1)], seed=0, max_iter=20,
                       callback=lambda i, x, s: scores.append(s))
        assert all(b <= a for a, b in zip(scores, scores[1:]))
