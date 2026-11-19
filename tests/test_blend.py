import itertools
import math
import random

import pytest
from currytruck import (
    knapsack,
    best_subset,
    shortest_route,
    vehicle_routes,
    InfeasibleError,
    EmptySpaceError,
)


def brute_knapsack(values, weights, cap):
    best = 0
    for k in range(len(values) + 1):
        for combo in itertools.combinations(range(len(values)), k):
            if sum(weights[i] for i in combo) <= cap:
                best = max(best, sum(values[i] for i in combo))
    return best


class TestKnapsack:
    def test_classic(self):
        r = knapsack([60, 100, 120], [10, 20, 30], 50)
        assert r.x == [1, 2]
        assert r.score == 220
        assert r.extra["weight"] == 50
        assert r.method == "knapsack_dp"

    def test_float_weights_use_bnb(self):
        r = knapsack([60, 100, 120], [1.0, 2.0, 3.0], 5.0)
        assert r.x == [1, 2]
        assert r.method == "knapsack_bnb"

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_brute_force_int(self, seed):
        rng = random.Random(seed)
        values = [rng.randint(1, 50) for _ in range(12)]
        weights = [rng.randint(1, 20) for _ in range(12)]
        r = knapsack(values, weights, 40)
        assert r.score == brute_knapsack(values, weights, 40)
        assert sum(weights[i] for i in r.x) <= 40

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_brute_force_float(self, seed):
        rng = random.Random(seed)
        values = [rng.uniform(1, 50) for _ in range(12)]
        weights = [rng.uniform(0.5, 20) for _ in range(12)]
        r = knapsack(values, weights, 40.0)
        assert r.score == pytest.approx(brute_knapsack(values, weights, 40.0))
        assert sum(weights[i] for i in r.x) <= 40.0

    def test_nothing_fits(self):
        r = knapsack([10, 20], [5, 6], 4)
        assert r.x == [] and r.score == 0

    def test_zero_weight_item_taken(self):
        r = knapsack([5, 10], [0, 100], 1)
        assert r.x == [0]

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            knapsack([1, 2], [1], 5)

    def test_negative_weight(self):
        with pytest.raises(ValueError):
            knapsack([1], [-1], 5)

    def test_maximize_flag(self):
        assert knapsack([1], [1], 1).maximize is True


class TestBestSubset:
    def test_exhaustive_exact_size(self):
        r = best_subset([4, 9, 1, 7], sum, size=2)
        assert r.x == [1, 3]
        assert r.extra["items"] == [9, 7]
        assert r.converged

    def test_minimize(self):
        r = best_subset([4, 9, 1, 7], sum, size=2, maximize=False)
        assert sorted(r.extra["items"]) == [1, 4]

    def test_constraint(self):
        r = best_subset([4, 9, 1, 7], sum, constraint=lambda s: sum(s) <= 12)
        assert sum(r.extra["items"]) == 12

    def test_greedy_fallback(self):
        r = best_subset(list(range(30)), sum, size=3, max_evals=10)
        assert r.method == "best_subset_greedy"
        assert sorted(r.extra["items"]) == [27, 28, 29]

    def test_greedy_stops_when_no_gain(self):
        r = best_subset([5, -1, 3, -4], sum, max_evals=1)
        assert sorted(r.extra["items"]) == [3, 5]

    def test_infeasible(self):
        with pytest.raises(InfeasibleError):
            best_subset([1, 2], sum, constraint=lambda s: False)

    def test_empty(self):
        with pytest.raises(EmptySpaceError):
            best_subset([], sum)

    def test_bad_range(self):
        with pytest.raises(ValueError):
            best_subset([1, 2], sum, size=5)


def brute_route(d, start, closed):
    n = len(d)
    best = None
    for perm in itertools.permutations([i for i in range(n) if i != start]):
        tour = [start, *perm]
        length = sum(d[tour[i]][tour[i + 1]] for i in range(n - 1))
        if closed:
            length += d[tour[-1]][start]
        best = length if best is None else min(best, length)
    return best


def random_matrix(rng, n, symmetric=True):
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = d[j][i] if symmetric and j < i else rng.uniform(1, 100)
    return d


class TestShortestRoute:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("closed", [True, False])
    def test_exact_matches_brute_force(self, seed, closed):
        rng = random.Random(seed)
        d = random_matrix(rng, 7, symmetric=seed % 2 == 0)
        r = shortest_route(d, start=seed % 7, closed=closed)
        assert r.extra["exact"]
        assert r.score == pytest.approx(brute_route(d, seed % 7, closed))
        assert r.x[0] == seed % 7 and sorted(r.x) == list(range(7))

    def test_heuristic_on_circle_is_optimal(self):
        n = 40
        pts = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        rng = random.Random(0)
        order = list(range(n))
        rng.shuffle(order)
        r = shortest_route(points=[pts[i] for i in order])
        assert not r.extra["exact"]
        assert r.score == pytest.approx(2 * n * math.sin(math.pi / n))

    def test_heuristic_close_to_exact(self):
        rng = random.Random(5)
        pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(10)]
        exact = shortest_route(points=pts)
        heuristic = shortest_route(points=pts, exact_limit=0)
        assert heuristic.score <= exact.score * 1.1

    def test_asymmetric_heuristic_valid(self):
        d = random_matrix(random.Random(2), 15, symmetric=False)
        r = shortest_route(d)
        assert sorted(r.x) == list(range(15))

    def test_tiny(self):
        assert shortest_route([[0]]).x == [0]
        assert shortest_route([[0, 3], [4, 0]]).score == 7

    def test_errors(self):
        with pytest.raises(ValueError):
            shortest_route()
        with pytest.raises(ValueError):
            shortest_route([[0, 1]])
        with pytest.raises(IndexError):
            shortest_route([[0, 1], [1, 0]], start=5)
        with pytest.raises(EmptySpaceError):
            shortest_route([])


def assert_valid_plan(r, n, depot, demands, capacity):
    served = [c for route in r.x for c in route]
    assert sorted(served) == [i for i in range(n) if i != depot]
    for route in r.x:
        assert sum(demands[c] for c in route) <= capacity + 1e-9


class TestVehicleRoutes:
    def test_capacity(self):
        rng = random.Random(0)
        pts = [(5, 5)] + [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(25)]
        demands = [0] + [rng.randint(1, 5) for _ in range(25)]
        r = vehicle_routes(points=pts, demands=demands, capacity=15)
        assert_valid_plan(r, 26, 0, demands, 15)
        assert r.score == pytest.approx(sum(r.extra["distances"]))
        assert len(r.x) >= math.ceil(sum(demands) / 15)

    def test_beats_one_truck_per_stop(self):
        rng = random.Random(1)
        pts = [(0, 0)] + [(rng.uniform(5, 10), rng.uniform(5, 10)) for _ in range(10)]
        r = vehicle_routes(points=pts, demands=[0] + [1] * 10, capacity=5)
        naive = sum(2 * math.dist(pts[0], p) for p in pts[1:])
        assert r.score < naive / 2

    def test_time_windows_respected(self):
        pts = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        windows = [(0, 100), (0, 2), (10, 12), (0, 100), (0, 100)]
        r = vehicle_routes(points=pts, demands=[0, 1, 1, 1, 1], capacity=10, time_windows=windows,
                           service_times=1)
        for route, arrivals in zip(r.x, r.extra["arrivals"]):
            for c, t in zip(route, arrivals):
                assert windows[c][0] <= t <= windows[c][1]

    def test_conflicting_windows_split_routes(self):
        # two stops far apart, both must be reached by t=5
        pts = [(0, 0), (4, 0), (-4, 0)]
        r = vehicle_routes(points=pts, demands=[0, 1, 1], capacity=10,
                           time_windows=[None, (0, 5), (0, 5)])
        assert len(r.x) == 2

    def test_fleet_limit(self):
        pts = [(0, 0), (1, 0), (2, 0), (3, 0)]
        with pytest.raises(InfeasibleError):
            vehicle_routes(points=pts, demands=[0, 2, 2, 2], capacity=2, n_vehicles=2)
        r = vehicle_routes(points=pts, demands=[0, 2, 2, 2], capacity=2, n_vehicles=3)
        assert len(r.x) == 3

    def test_demand_over_capacity(self):
        with pytest.raises(InfeasibleError):
            vehicle_routes(points=[(0, 0), (1, 1)], demands=[0, 5], capacity=4)

    def test_unreachable_window(self):
        with pytest.raises(InfeasibleError):
            vehicle_routes(points=[(0, 0), (10, 0)], demands=[0, 1], capacity=4,
                           time_windows=[None, (0, 5)])

    def test_depot_day_length(self):
        with pytest.raises(InfeasibleError):
            vehicle_routes(points=[(0, 0), (10, 0)], demands=[0, 1], capacity=4,
                           time_windows=[(0, 15), None])

    def test_non_zero_depot(self):
        pts = [(1, 0), (0, 0), (2, 0)]
        r = vehicle_routes(points=pts, demands=[1, 0, 1], capacity=5, depot=1)
        assert_valid_plan(r, 3, 1, [1, 0, 1], 5)
        assert r.score == pytest.approx(4.0)

    def test_errors(self):
        with pytest.raises(ValueError):
            vehicle_routes(points=[(0, 0), (1, 1)], demands=[0], capacity=1)
        with pytest.raises(ValueError):
            vehicle_routes(points=[(0, 0), (1, 1)], demands=[0, 1], capacity=1, service_times=[1])
        with pytest.raises(IndexError):
            vehicle_routes(points=[(0, 0)], demands=[0], capacity=1, depot=3)
