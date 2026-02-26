"""Tests for the Contiguous-Implies-Positroid theorem.

Theorem: If every non-basis of a matroid M on [n] is a cyclic interval
({j, j+1, ..., j+k-1} mod n), then M is a positroid.

These tests verify the helper functions and exhaustively confirm the theorem
for small parameters.
"""

from itertools import combinations

import pytest

from positroid.matroid.matroid import Matroid, uniform_matroid
from positroid.matroid.positroid import (
    has_only_cyclic_interval_nonbases,
    is_cyclic_interval,
    is_positroid,
)


class TestIsCyclicInterval:
    """Tests for is_cyclic_interval."""

    def test_contiguous_range(self):
        """Standard contiguous subsets are cyclic intervals."""
        assert is_cyclic_interval(frozenset({0, 1, 2}), 6)
        assert is_cyclic_interval(frozenset({1, 2, 3}), 6)
        assert is_cyclic_interval(frozenset({3, 4, 5}), 6)

    def test_wrap_around(self):
        """Subsets wrapping around 0 are cyclic intervals."""
        assert is_cyclic_interval(frozenset({4, 5, 0}), 6)
        assert is_cyclic_interval(frozenset({5, 0, 1}), 6)
        assert is_cyclic_interval(frozenset({4, 5, 0, 1}), 6)

    def test_spread_triples_not_intervals(self):
        """Spread subsets with gaps are NOT cyclic intervals."""
        assert not is_cyclic_interval(frozenset({0, 2, 4}), 6)
        assert not is_cyclic_interval(frozenset({1, 3, 5}), 6)
        assert not is_cyclic_interval(frozenset({0, 2, 4}), 5)

    def test_pairs(self):
        """Adjacent pairs are intervals; non-adjacent are not."""
        assert is_cyclic_interval(frozenset({0, 1}), 4)
        assert is_cyclic_interval(frozenset({3, 0}), 4)  # wrap-around
        assert not is_cyclic_interval(frozenset({0, 2}), 4)
        assert not is_cyclic_interval(frozenset({1, 3}), 4)

    def test_singletons_and_empty(self):
        """Singletons and empty sets are trivially cyclic intervals."""
        assert is_cyclic_interval(frozenset({3}), 5)
        assert is_cyclic_interval(frozenset(), 5)

    def test_full_set(self):
        """The full ground set is a cyclic interval."""
        assert is_cyclic_interval(frozenset({0, 1, 2, 3}), 4)
        assert is_cyclic_interval(frozenset(range(6)), 6)

    def test_tuple_input(self):
        """Works with tuples too."""
        assert is_cyclic_interval((0, 1, 2), 5)
        assert not is_cyclic_interval((0, 2, 4), 6)

    def test_exhaustive_intervals_on_4(self):
        """All cyclic intervals of size 2 on [4]: exactly {0,1},{1,2},{2,3},{3,0}."""
        intervals = [
            s for s in combinations(range(4), 2) if is_cyclic_interval(s, 4)
        ]
        # (3, 0) wraps — but combinations gives (0, 3)
        expected_fs = {frozenset(s) for s in [(0, 1), (1, 2), (2, 3), (0, 3)]}
        actual_fs = {frozenset(s) for s in intervals}
        assert actual_fs == expected_fs

    def test_exhaustive_intervals_on_6_size_3(self):
        """All cyclic intervals of size 3 on [6]: exactly 6 of them."""
        intervals = [
            frozenset(s)
            for s in combinations(range(6), 3)
            if is_cyclic_interval(s, 6)
        ]
        expected = {
            frozenset({0, 1, 2}),
            frozenset({1, 2, 3}),
            frozenset({2, 3, 4}),
            frozenset({3, 4, 5}),
            frozenset({4, 5, 0}),
            frozenset({5, 0, 1}),
        }
        assert set(intervals) == expected

    def test_n_equals_k(self):
        """Size-n subset on [n] is always an interval."""
        assert is_cyclic_interval(frozenset({0, 1, 2}), 3)

    def test_size_n_minus_1(self):
        """Size (n-1) subsets are always cyclic intervals (one element missing)."""
        for n in range(3, 7):
            for s in combinations(range(n), n - 1):
                assert is_cyclic_interval(frozenset(s), n)


class TestHasOnlyCyclicIntervalNonbases:
    """Tests for has_only_cyclic_interval_nonbases."""

    def test_uniform_matroid_vacuously_true(self):
        """Uniform matroid has no non-bases, so condition holds vacuously."""
        assert has_only_cyclic_interval_nonbases(uniform_matroid(2, 4))
        assert has_only_cyclic_interval_nonbases(uniform_matroid(3, 5))

    def test_adjacent_parallel_true(self):
        """Remove {0,1} from U(2,4) — contiguous non-basis."""
        bases = frozenset(
            frozenset(s) for s in combinations(range(4), 2)
        ) - {frozenset({0, 1})}
        m = Matroid(frozenset(range(4)), bases)
        assert has_only_cyclic_interval_nonbases(m)

    def test_interleaved_parallel_false(self):
        """Remove {0,2} from U(2,4) — non-contiguous non-basis."""
        bases = frozenset(
            frozenset(s) for s in combinations(range(4), 2)
        ) - {frozenset({0, 2})}
        m = Matroid(frozenset(range(4)), bases)
        assert not has_only_cyclic_interval_nonbases(m)

    def test_spread_triple_false(self):
        """Remove {0,2,4} from U(3,6) — spread non-basis."""
        bases = frozenset(
            frozenset(s) for s in combinations(range(6), 3)
        ) - {frozenset({0, 2, 4})}
        m = Matroid(frozenset(range(6)), bases)
        assert not has_only_cyclic_interval_nonbases(m)

    def test_two_contiguous_nonbases(self):
        """Remove {0,1} and {2,3} from U(2,4) — both contiguous."""
        bases = frozenset(
            frozenset(s) for s in combinations(range(4), 2)
        ) - {frozenset({0, 1}), frozenset({2, 3})}
        m = Matroid(frozenset(range(4)), bases)
        assert has_only_cyclic_interval_nonbases(m)


class TestContiguousImpliesPositroid:
    """Exhaustive verification of the theorem for small (n, k).

    For each (n, k), enumerate all subsets of cyclic intervals that can
    be removed from U(k, n) to still yield a valid matroid (exchange axiom).
    Verify that every such matroid is a positroid.
    """

    @pytest.mark.parametrize(
        "n,k",
        [(4, 2), (5, 2), (5, 3), (6, 2), (6, 3), (7, 3), (8, 3)],
    )
    def test_all_cyclic_interval_removals(self, n: int, k: int):
        """Every valid matroid formed by removing cyclic-interval non-bases
        from U(k, n) is a positroid."""
        all_bases = frozenset(frozenset(s) for s in combinations(range(n), k))

        # Find all cyclic intervals of size k on [n]
        cyclic_intervals = [
            frozenset(s)
            for s in combinations(range(n), k)
            if is_cyclic_interval(s, n)
        ]

        num_tested = 0
        # Enumerate all 2^|intervals| subsets of intervals to remove
        for mask in range(1, 1 << len(cyclic_intervals)):
            to_remove = frozenset(
                cyclic_intervals[i]
                for i in range(len(cyclic_intervals))
                if mask & (1 << i)
            )
            remaining = all_bases - to_remove
            if not remaining:
                continue

            # Check if remaining forms a valid matroid (exchange axiom)
            try:
                m = Matroid(frozenset(range(n)), remaining)
            except ValueError:
                # Exchange axiom violated — not a matroid, skip
                continue

            num_tested += 1
            assert is_positroid(m), (
                f"Contiguous-implies-positroid FAILED for n={n}, k={k}: "
                f"removed {to_remove}, got non-positroid matroid with "
                f"{len(remaining)} bases"
            )

        # Sanity: we tested at least a few matroids
        assert num_tested > 0, f"No valid matroids for n={n}, k={k}"


class TestSingleRemovalDichotomy:
    """Removing a single k-subset from U(k, n): cyclic interval → positroid,
    non-cyclic-interval → non-positroid."""

    @pytest.mark.parametrize(
        "n,k",
        [(4, 2), (5, 2), (5, 3), (6, 2), (6, 3), (7, 3)],
    )
    def test_single_removal(self, n: int, k: int):
        """For each k-subset S of [n], removing S from U(k,n):
        - If S is a cyclic interval → result is a positroid.
        - If S is NOT a cyclic interval → result is NOT a positroid.
        """
        all_bases = frozenset(frozenset(s) for s in combinations(range(n), k))

        for subset in combinations(range(n), k):
            s = frozenset(subset)
            remaining = all_bases - {s}

            try:
                m = Matroid(frozenset(range(n)), remaining)
            except ValueError:
                # Can't form a matroid — skip (shouldn't happen for single removal
                # from uniform matroid with n > k+1)
                continue

            if is_cyclic_interval(s, n):
                assert is_positroid(m), (
                    f"Cyclic interval {s} removed from U({k},{n}) "
                    f"should give positroid"
                )
            else:
                assert not is_positroid(m), (
                    f"Non-interval {s} removed from U({k},{n}) "
                    f"should give non-positroid"
                )


class TestConnectionToCounterexamples:
    """Link to the counterexample search: crossing pairs are exactly
    the non-cyclic-interval patterns."""

    def test_spread_triples_not_intervals(self):
        """{0,2,4} and {1,3,5} — the canonical counterexample generators —
        are not cyclic intervals."""
        assert not is_cyclic_interval(frozenset({0, 2, 4}), 6)
        assert not is_cyclic_interval(frozenset({1, 3, 5}), 6)

    def test_disjoint_cyclic_intervals_never_cross(self):
        """Two disjoint cyclic intervals of the same size never cross.

        Contiguous arcs on a circle that don't share elements can't interleave.
        This is why removing only cyclic-interval non-bases preserves positroid
        structure: the "crossing pair" mechanism for breaking positroids
        requires non-interval subsets.
        """
        from positroid.experiments.counterexample_search import is_crossing_pair

        for n in range(4, 8):
            for k in range(2, min(n, 5)):
                intervals = [
                    s
                    for s in combinations(range(n), k)
                    if is_cyclic_interval(s, n)
                ]
                for i, c1 in enumerate(intervals):
                    for c2 in intervals[i + 1 :]:
                        if set(c1).isdisjoint(set(c2)):
                            assert not is_crossing_pair(c1, c2, n), (
                                f"Disjoint cyclic intervals {c1} and {c2} "
                                f"cross on [{n}]"
                            )

    def test_non_interval_pairs_can_cross(self):
        """Non-interval subsets can form crossing pairs."""
        from positroid.experiments.counterexample_search import is_crossing_pair

        assert is_crossing_pair((0, 2, 4), (1, 3, 5), 6)
        assert is_crossing_pair((0, 2), (1, 3), 4)
