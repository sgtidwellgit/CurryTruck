import itertools
import random

import pytest
from currytruck import knapsack, best_subset, InfeasibleError, EmptySpaceError


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
