from positroid.matroid.matroid import Matroid, uniform_matroid
from positroid.matroid.positroid import (
    bases_from_grassmann_necklace,
    decorated_permutation,
    grassmann_necklace,
    is_positroid,
)


class TestGrassmannNecklace:
    def test_u24_necklace(self):
        """Grassmann necklace of U(2, 4) = ({0,1}, {1,2}, {2,3}, {0,3})."""
        m = uniform_matroid(2, 4)
        necklace = grassmann_necklace(m)
        assert len(necklace) == 4
        assert necklace[0] == frozenset({0, 1})
        assert necklace[1] == frozenset({1, 2})
        assert necklace[2] == frozenset({2, 3})
        assert necklace[3] == frozenset({0, 3})

    def test_u13_necklace(self):
        """Grassmann necklace of U(1, 3) = ({0}, {1}, {2})."""
        m = uniform_matroid(1, 3)
        necklace = grassmann_necklace(m)
        assert necklace == (frozenset({0}), frozenset({1}), frozenset({2}))

    def test_u23_necklace(self):
        m = uniform_matroid(2, 3)
        necklace = grassmann_necklace(m)
        assert len(necklace) == 3
        assert necklace[0] == frozenset({0, 1})
        assert necklace[1] == frozenset({1, 2})
        assert necklace[2] == frozenset({0, 2})


class TestBasesFromNecklace:
    def test_roundtrip_u24(self):
        """Reconstruct bases of U(2,4) from its Grassmann necklace."""
        m = uniform_matroid(2, 4)
        necklace = grassmann_necklace(m)
        reconstructed = bases_from_grassmann_necklace(necklace, 4, 2)
        assert reconstructed == m.bases

    def test_roundtrip_u13(self):
        m = uniform_matroid(1, 3)
        necklace = grassmann_necklace(m)
        reconstructed = bases_from_grassmann_necklace(necklace, 3, 1)
        assert reconstructed == m.bases


class TestIsPositroid:
    def test_uniform_is_positroid(self):
        """Every uniform matroid is a positroid."""
        assert is_positroid(uniform_matroid(1, 3))
        assert is_positroid(uniform_matroid(2, 4))
        assert is_positroid(uniform_matroid(2, 5))
        assert is_positroid(uniform_matroid(3, 6))

    def test_non_uniform_positroid(self):
        """A non-uniform matroid that IS a positroid.

        Remove basis {0,1} from U(2,4). The resulting matroid
        (where 0 and 1 are parallel) is a positroid — it's the
        positroid of rank 2 on [4] realized by the matrix
        [[1, 0, 1, 1], [0, 0, 1, 1]] (with columns 0,1 parallel).
        """
        bases = frozenset([
            frozenset({0, 2}),
            frozenset({0, 3}),
            frozenset({1, 2}),
            frozenset({1, 3}),
            frozenset({2, 3}),
        ])
        m = Matroid(frozenset({0, 1, 2, 3}), bases)
        assert is_positroid(m)

    def test_adjacent_parallel_is_positroid(self):
        """Adjacent parallel pairs (0||1 and 2||3) form a positroid.

        Missing {0,1} and {2,3}. The parallel pairs are adjacent in
        cyclic order, so this IS realizable in Gr_+(2,4).
        """
        bases = frozenset([
            frozenset({0, 2}),
            frozenset({0, 3}),
            frozenset({1, 2}),
            frozenset({1, 3}),
        ])
        m = Matroid(frozenset({0, 1, 2, 3}), bases)
        assert is_positroid(m)

    def test_interleaved_parallel_is_not_positroid(self):
        """Interleaved parallel pairs (0||2 and 1||3) are NOT a positroid.

        Bases: {0,1}, {0,3}, {1,2}, {2,3}. Missing {0,2} and {1,3}.
        The dependent pairs (0,2) and (1,3) are interleaved in the cyclic
        order 0,1,2,3, so no totally nonneg realization exists.
        """
        bases = frozenset([
            frozenset({0, 1}),
            frozenset({0, 3}),
            frozenset({1, 2}),
            frozenset({2, 3}),
        ])
        m = Matroid(frozenset({0, 1, 2, 3}), bases)
        assert not is_positroid(m)


class TestDecoratedPermutation:
    def test_u24_permutation(self):
        m = uniform_matroid(2, 4)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 4)
        # For U(2,4), every element is replaced by the next in cyclic order
        # I_0 = {0,1}, I_1 = {1,2} -> 0 is replaced by 2, so pi(0) = 2
        # I_1 = {1,2}, I_2 = {2,3} -> 1 is replaced by 3, so pi(1) = 3
        # I_2 = {2,3}, I_3 = {0,3} -> 2 is replaced by 0, so pi(2) = 0
        # I_3 = {0,3}, I_0 = {0,1} -> 3 is replaced by 1, so pi(3) = 1
        assert perm == [2, 3, 0, 1]
