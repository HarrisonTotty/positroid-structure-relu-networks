import pytest

from positroid.matroid.matroid import Matroid, uniform_matroid


class TestMatroid:
    def test_uniform_matroid_u24(self):
        m = uniform_matroid(2, 4)
        assert m.rank == 2
        assert m.size == 4
        assert len(m.bases) == 6  # C(4,2)

    def test_uniform_matroid_is_uniform(self):
        assert uniform_matroid(2, 4).is_uniform()
        assert uniform_matroid(3, 5).is_uniform()

    def test_rank_of_subset(self):
        m = uniform_matroid(2, 4)
        assert m.rank_of(frozenset({0, 1})) == 2
        assert m.rank_of(frozenset({0})) == 1
        assert m.rank_of(frozenset()) == 0
        assert m.rank_of(frozenset({0, 1, 2})) == 2

    def test_is_independent(self):
        m = uniform_matroid(2, 4)
        assert m.is_independent(frozenset({0, 1}))
        assert m.is_independent(frozenset({2}))
        assert m.is_independent(frozenset())
        assert not m.is_independent(frozenset({0, 1, 2}))

    def test_circuits_u24(self):
        m = uniform_matroid(2, 4)
        circuits = m.circuits()
        # Circuits of U(2,4) are all 3-element subsets
        assert len(circuits) == 4  # C(4,3)
        for c in circuits:
            assert len(c) == 3

    def test_dual(self):
        m = uniform_matroid(2, 4)
        d = m.dual()
        assert d.rank == 2  # U(2,4)* = U(2,4)
        assert d.size == 4
        assert len(d.bases) == 6

    def test_invalid_bases_rejected(self):
        with pytest.raises(ValueError, match="same size"):
            Matroid(frozenset({0, 1, 2}), frozenset([frozenset({0}), frozenset({1, 2})]))

    def test_exchange_axiom_violation(self):
        # Try to create a "matroid" with bases that violate exchange
        # {0,1} and {2,3} but NOT {0,2}, {0,3}, {1,2}, {1,3}
        with pytest.raises(ValueError, match="exchange"):
            Matroid(
                frozenset({0, 1, 2, 3}),
                frozenset([frozenset({0, 1}), frozenset({2, 3})]),
            )

    def test_non_uniform_matroid(self):
        """A rank-2 matroid on 4 elements that is NOT uniform.

        Remove {0,1} from the bases of U(2,4). This means 0 and 1 are
        "parallel" (dependent pair).
        """
        bases = frozenset(
            [
                frozenset({0, 2}),
                frozenset({0, 3}),
                frozenset({1, 2}),
                frozenset({1, 3}),
                frozenset({2, 3}),
            ]
        )
        m = Matroid(frozenset({0, 1, 2, 3}), bases)
        assert not m.is_uniform()
        assert m.rank == 2
        assert not m.is_independent(frozenset({0, 1}))

    def test_equality(self):
        m1 = uniform_matroid(2, 4)
        m2 = uniform_matroid(2, 4)
        assert m1 == m2

    def test_repr(self):
        m = uniform_matroid(2, 4)
        assert "n=4" in repr(m)
        assert "r=2" in repr(m)
