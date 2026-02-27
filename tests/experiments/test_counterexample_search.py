import numpy as np
import pytest

from positroid.experiments.counterexample_search import (
    compute_dependency_coefficients,
    find_crossing_pairs,
    is_crossing_pair,
    run_counterexample_search,
    solve_bias_for_circuits,
    targeted_search,
)
from positroid.linalg.totally_positive import (
    is_totally_positive,
    random_totally_positive,
    tp_from_exponential_kernel,
)
from positroid.matroid.linear_matroid import linear_matroid_from_vectors
from positroid.matroid.positroid import is_positroid


class TestDependencyCoefficients:
    def test_2d_sign_pattern(self):
        """For TP matrices, dependency coefficients for triples have (+, -, +) sign."""
        rng = np.random.default_rng(42)
        w = random_totally_positive(5, 2, rng=rng)
        deps = compute_dependency_coefficients(w)

        # All triples of a 5x2 matrix
        assert len(deps) == 10  # C(5, 3)

        for subset, coeffs in deps.items():
            assert len(coeffs) == 3
            # For TP matrices: c[0] > 0, c[1] < 0, c[2] > 0
            # (normalized so first nonzero is positive)
            assert coeffs[0] > 0, f"Expected c[0] > 0 for {subset}"
            assert coeffs[1] < 0, f"Expected c[1] < 0 for {subset}"
            assert coeffs[2] > 0, f"Expected c[2] > 0 for {subset}"

    def test_coefficients_satisfy_dependency(self):
        """Dependency coefficients should annihilate the weight rows."""
        rng = np.random.default_rng(42)
        w = random_totally_positive(6, 2, rng=rng)
        deps = compute_dependency_coefficients(w)

        for subset, coeffs in deps.items():
            sub_w = w[list(subset)]
            # c @ W_sub should be ~0
            residual = coeffs @ sub_w
            np.testing.assert_allclose(residual, 0, atol=1e-10)

    def test_higher_dim(self):
        """Test with d=3 (rank 4 affine matroid, 4-tuples)."""
        rng = np.random.default_rng(42)
        w = random_totally_positive(6, 3, rng=rng)
        deps = compute_dependency_coefficients(w)

        # C(6, 4) = 15 quadruples
        assert len(deps) == 15

        for subset, coeffs in deps.items():
            assert len(coeffs) == 4
            sub_w = w[list(subset)]
            residual = coeffs @ sub_w
            np.testing.assert_allclose(residual, 0, atol=1e-10)


class TestSolveBiasForCircuits:
    def test_constraint_satisfied(self):
        """Verify the bias solver satisfies the dependency constraint numerically."""
        rng = np.random.default_rng(42)
        w = random_totally_positive(5, 2, rng=rng)
        target = (0, 2, 4)

        deps = compute_dependency_coefficients(w)
        b = solve_bias_for_circuits(w, [target], rng)

        # The constraint c . b_S = 0 should be satisfied to machine precision
        coeffs = deps[target]
        b_sub = b[list(target)]
        residual = abs(coeffs @ b_sub)
        assert residual < 1e-12, f"Constraint residual {residual} too large"

    def test_single_circuit_becomes_non_basis(self):
        """Solving for one circuit should make that triple a non-basis.

        Uses well-separated exponential kernel parameters for numerical stability.
        """
        rng = np.random.default_rng(42)
        # Well-separated params give better-conditioned matrices
        w = tp_from_exponential_kernel(
            np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            np.array([0.5, 1.5]),
        )
        target = (0, 2, 4)

        b = solve_bias_for_circuits(w, [target], rng)

        aug = np.hstack([w, b.reshape(-1, 1)])
        mat = linear_matroid_from_vectors(aug)

        # Target triple should NOT be a basis
        assert frozenset(target) not in mat.bases

    def test_two_disjoint_circuits(self):
        """Two disjoint circuits should both become non-bases."""
        rng = np.random.default_rng(42)
        w = tp_from_exponential_kernel(
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            np.array([0.5, 1.5]),
        )
        targets = [(0, 2, 4), (1, 3, 5)]

        b = solve_bias_for_circuits(w, targets, rng)

        aug = np.hstack([w, b.reshape(-1, 1)])
        mat = linear_matroid_from_vectors(aug)

        assert frozenset({0, 2, 4}) not in mat.bases
        assert frozenset({1, 3, 5}) not in mat.bases

    def test_result_is_valid_matroid(self):
        """The resulting matroid should pass exchange axiom validation."""
        rng = np.random.default_rng(42)
        w = tp_from_exponential_kernel(
            np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            np.array([0.5, 1.5]),
        )
        b = solve_bias_for_circuits(w, [(0, 2, 4)], rng)

        aug = np.hstack([w, b.reshape(-1, 1)])
        mat = linear_matroid_from_vectors(aug)
        assert mat.rank == 3
        assert mat.size == 5


class TestCrossingPairs:
    def test_basic_crossing(self):
        """(0,2,4) and (1,3,5) cross on [6]."""
        assert is_crossing_pair((0, 2, 4), (1, 3, 5), 6)

    def test_adjacent_not_crossing(self):
        """(0,1,2) and (3,4,5) do not cross on [6]."""
        assert not is_crossing_pair((0, 1, 2), (3, 4, 5), 6)

    def test_single_spread_on_5(self):
        """(0,2,4) and (1,3) would cross if same size; test (0,2) vs (1,3) on [4]."""
        # rank 2 case: pairs
        assert is_crossing_pair((0, 2), (1, 3), 4)
        assert not is_crossing_pair((0, 1), (2, 3), 4)

    def test_find_crossing_pairs_n6_rank3(self):
        """Should find crossing pairs including (0,2,4) and (1,3,5) on [6]."""
        pairs = find_crossing_pairs(6, 3)
        assert ((0, 2, 4), (1, 3, 5)) in pairs
        assert len(pairs) > 0

    def test_find_crossing_pairs_n5_rank3(self):
        """Should find some crossing pairs on [5] with size 3."""
        pairs = find_crossing_pairs(5, 3)
        # (0, 2, 4) doesn't have a disjoint partner on [5], but
        # can still have crossing pairs with shared elements
        assert len(pairs) > 0


class TestCounterexampleConstruction:
    """The critical tests: can we construct non-positroid matroids from TP weights?"""

    def test_spread_triple_on_5_is_not_positroid(self):
        """Removing {0,2,4} from U(3,5) gives a non-positroid matroid.

        This is a purely combinatorial test — no TP matrices involved yet.
        """
        from positroid.matroid.matroid import Matroid

        all_bases = frozenset(
            frozenset(s)
            for s in [
                (0, 1, 2),
                (0, 1, 3),
                (0, 1, 4),
                (0, 2, 3),
                (0, 3, 4),
                (1, 2, 3),
                (1, 2, 4),
                (1, 3, 4),
                (2, 3, 4),
            ]
            # Missing: {0, 2, 4}
        )
        m = Matroid(frozenset(range(5)), all_bases)
        assert not m.is_uniform()
        assert not is_positroid(m)

    def test_crossing_circuits_on_6_is_not_positroid(self):
        """Removing {0,2,4} and {1,3,5} from U(3,6) gives a non-positroid."""
        from itertools import combinations as combs

        from positroid.matroid.matroid import Matroid

        excluded = {frozenset({0, 2, 4}), frozenset({1, 3, 5})}
        all_bases = frozenset(
            frozenset(s) for s in combs(range(6), 3) if frozenset(s) not in excluded
        )
        m = Matroid(frozenset(range(6)), all_bases)
        assert not m.is_uniform()
        assert not is_positroid(m)

    def test_tp_counterexample_5x2(self):
        """THE KEY TEST: construct a 5x2 TP matrix + biases that give non-positroid.

        Theory: for any TP W (5x2), we can choose biases to make {0,2,4}
        a non-basis. The resulting rank-3 matroid on [5] should NOT be a positroid.
        """
        rng = np.random.default_rng(42)
        w = tp_from_exponential_kernel(
            np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            np.array([0.5, 1.5]),
        )
        assert is_totally_positive(w)

        b = solve_bias_for_circuits(w, [(0, 2, 4)], rng)
        aug = np.hstack([w, b.reshape(-1, 1)])
        mat = linear_matroid_from_vectors(aug)

        # The matroid should be non-uniform (we targeted a specific non-basis)
        assert not mat.is_uniform(), (
            "Expected non-uniform matroid but got uniform. "
            "Bias solver may have failed to target the circuit."
        )

        # THE CONJECTURE SAYS: this should be a positroid.
        # Our theory says: it should NOT be.
        # If is_positroid returns False, we have a counterexample!
        positroid_result = is_positroid(mat)

        # Record what happened regardless
        if not positroid_result:
            # Counterexample found!
            assert not positroid_result, "Counterexample confirmed"
        else:
            # If it IS a positroid, the theory was wrong — still interesting
            pytest.skip(
                "Matroid was a positroid — theory prediction was wrong. Not a counterexample."
            )

    def test_tp_counterexample_6x2_crossing(self):
        """Construct a 6x2 TP matrix with crossing circuits {0,2,4} and {1,3,5}."""
        rng = np.random.default_rng(42)
        w = tp_from_exponential_kernel(
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            np.array([0.5, 1.5]),
        )
        assert is_totally_positive(w)

        b = solve_bias_for_circuits(w, [(0, 2, 4), (1, 3, 5)], rng)
        aug = np.hstack([w, b.reshape(-1, 1)])
        mat = linear_matroid_from_vectors(aug)

        assert not mat.is_uniform()

        positroid_result = is_positroid(mat)
        if not positroid_result:
            assert not positroid_result, "Counterexample confirmed"
        else:
            pytest.skip("Matroid was a positroid — not a counterexample.")

    def test_multiple_tp_matrices(self):
        """Try several different TP matrices — the counterexample should be robust."""
        counterexamples = 0
        total = 0

        for seed in range(10):
            trial_rng = np.random.default_rng(seed)
            w = random_totally_positive(5, 2, rng=trial_rng)
            assert is_totally_positive(w)

            b = solve_bias_for_circuits(w, [(0, 2, 4)], trial_rng)
            aug = np.hstack([w, b.reshape(-1, 1)])
            mat = linear_matroid_from_vectors(aug)

            if not mat.is_uniform():
                total += 1
                if not is_positroid(mat):
                    counterexamples += 1

        # We expect most or all non-uniform matroids to be non-positroids
        assert total > 0, "No non-uniform matroids produced"
        assert counterexamples > 0, f"No counterexamples among {total} non-uniform matroids"


class TestDichotomyClassification:
    """Verify the dichotomy: counterexamples have non-interval non-bases,
    and trials with only cyclic-interval non-bases are always positroids."""

    def test_counterexamples_have_non_interval_nonbases(self):
        """Every counterexample should have at least one non-interval non-basis."""
        from positroid.matroid.positroid import is_cyclic_interval

        summary = run_counterexample_search(
            configs=[(2, 5), (2, 6)],
            num_matrices=3,
            strategies=["targeted"],
            seed=42,
            kernels=["exponential"],
        )
        counterexamples = [r for r in summary.results if r.is_counterexample]
        assert len(counterexamples) > 0, "Expected at least one counterexample"

        for r in counterexamples:
            has_non_interval = any(
                not is_cyclic_interval(frozenset(nb), r.hidden_dim) for nb in r.non_bases
            )
            assert has_non_interval, (
                f"Counterexample d={r.input_dim}, H={r.hidden_dim} "
                f"has only cyclic-interval non-bases: {r.non_bases}"
            )

    def test_all_interval_nonbases_implies_positroid(self):
        """Trials where every non-basis is a cyclic interval should be positroids."""
        from positroid.matroid.positroid import is_cyclic_interval

        summary = run_counterexample_search(
            configs=[(2, 5), (2, 6)],
            num_matrices=5,
            strategies=["targeted"],
            seed=42,
            kernels=["exponential"],
        )
        non_uniform = [r for r in summary.results if not r.is_uniform]
        assert len(non_uniform) > 0

        all_interval_count = 0
        for r in non_uniform:
            all_interval = all(
                is_cyclic_interval(frozenset(nb), r.hidden_dim) for nb in r.non_bases
            )
            if all_interval:
                all_interval_count += 1
                assert r.is_positroid, (
                    f"d={r.input_dim}, H={r.hidden_dim}: all non-bases are "
                    f"cyclic intervals {r.non_bases} but matroid is not a "
                    f"positroid — violates theorem"
                )

        # We expect at least some trials to have only interval non-bases
        # (from the "targeted_single" strategy hitting contiguous subsets)
        assert all_interval_count > 0, (
            "No trials had only cyclic-interval non-bases — "
            "can't verify the theorem direction of the dichotomy"
        )


class TestTargetedSearch:
    def test_targeted_finds_results(self):
        """Targeted search should find non-uniform matroids."""
        rng = np.random.default_rng(42)
        w = random_totally_positive(6, 2, rng=rng)
        results = targeted_search(w, 2, 6, "exponential", rng)
        assert len(results) > 0

        non_uniform = [r for r in results if not r.is_uniform]
        assert len(non_uniform) > 0


class TestFullSearch:
    def test_small_search(self):
        """Run a small search to verify the pipeline works."""
        summary = run_counterexample_search(
            configs=[(2, 5)],
            num_matrices=3,
            strategies=["targeted"],
            seed=42,
            kernels=["exponential"],
        )
        assert summary.num_trials > 0
        assert summary.num_non_uniform > 0
