from itertools import combinations

from positroid.matroid.matroid import Matroid, uniform_matroid
from positroid.matroid.positroid import (
    nonbase_support,
    support_is_cyclic_interval,
    support_rank_deficiency,
)


def _remove_bases(n: int, k: int, removals: list[frozenset[int]]) -> Matroid:
    """Create U(k,n) with specified bases removed."""
    all_bases = frozenset(frozenset(s) for s in combinations(range(n), k))
    return Matroid(frozenset(range(n)), all_bases - frozenset(removals))


class TestNonbaseSupport:
    def test_uniform_empty(self):
        """Uniform matroid has no non-bases, so support is empty."""
        m = uniform_matroid(2, 5)
        assert nonbase_support(m) == frozenset()

    def test_single_removal(self):
        """Removing {0,1} from U(2,4): support = {0,1}."""
        m = _remove_bases(4, 2, [frozenset({0, 1})])
        assert nonbase_support(m) == frozenset({0, 1})

    def test_multiple_removals(self):
        """Removing {0,1} and {2,3}: support = {0,1,2,3}."""
        m = _remove_bases(4, 2, [frozenset({0, 1}), frozenset({2, 3})])
        assert nonbase_support(m) == frozenset({0, 1, 2, 3})

    def test_spread_nonbasis(self):
        """Removing {0,2} from U(2,4): support = {0,2}."""
        m = _remove_bases(4, 2, [frozenset({0, 2})])
        assert nonbase_support(m) == frozenset({0, 2})


class TestSupportIsCyclicInterval:
    def test_uniform_trivially_true(self):
        m = uniform_matroid(2, 5)
        assert support_is_cyclic_interval(m) is True

    def test_adjacent_true(self):
        """Support {0,1} on [4] is a cyclic interval."""
        m = _remove_bases(4, 2, [frozenset({0, 1})])
        assert support_is_cyclic_interval(m) is True

    def test_spread_false(self):
        """Support {0,2} on [4] is NOT a cyclic interval."""
        m = _remove_bases(4, 2, [frozenset({0, 2})])
        assert support_is_cyclic_interval(m) is False

    def test_wraparound_true(self):
        """Support {0,4} on [5] wraps around — is a cyclic interval."""
        m = _remove_bases(5, 2, [frozenset({0, 4})])
        assert support_is_cyclic_interval(m) is True


class TestSupportRankDeficiency:
    def test_uniform_zero(self):
        m = uniform_matroid(2, 5)
        assert support_rank_deficiency(m) == 0

    def test_single_removal_deficient(self):
        """Remove {0,1} from U(2,4): support={0,1}, rank(support)=1, deficiency=1."""
        m = _remove_bases(4, 2, [frozenset({0, 1})])
        assert support_rank_deficiency(m) == 1

    def test_spread_nonbases_full_rank(self):
        """Remove {0,2} and {1,3} from U(2,5): support={0,1,2,3}, rank=2, deficiency=0."""
        m = _remove_bases(5, 2, [frozenset({0, 2}), frozenset({1, 3})])
        assert support_rank_deficiency(m) == 0

    def test_tail_collapse_deficient(self):
        """Remove all 2-subsets of {0,1,2} from U(2,5): support={0,1,2}, rank=1, deficiency=1."""
        removals = [frozenset(s) for s in combinations(range(3), 2)]
        m = _remove_bases(5, 2, removals)
        assert support_rank_deficiency(m) == 1
