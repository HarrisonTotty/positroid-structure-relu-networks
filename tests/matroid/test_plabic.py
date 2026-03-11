"""Tests for plabic graph construction."""

import pytest

from positroid.matroid.matroid import Matroid, uniform_matroid
from positroid.matroid.plabic import (
    PlabicGraph,
    plabic_graph_from_decorated_permutation,
    plabic_graph_from_matroid,
    reduced_word_for_permutation,
)
from positroid.matroid.positroid import (
    decorated_permutation,
    grassmann_necklace,
)


class TestReducedWord:
    def test_identity(self):
        """Identity permutation has empty reduced word."""
        assert reduced_word_for_permutation([0, 1, 2, 3]) == []

    def test_single_transposition(self):
        """[1, 0] has reduced word [0]."""
        word = reduced_word_for_permutation([1, 0])
        assert word == [0]

    def test_single_transposition_n3(self):
        """[0, 2, 1] has reduced word [1]."""
        word = reduced_word_for_permutation([0, 2, 1])
        assert word == [1]

    def test_cyclic_shift_4(self):
        """[2,3,0,1] — cyclic shift of n=4 by 2."""
        perm = [2, 3, 0, 1]
        word = reduced_word_for_permutation(perm)
        # Number of inversions: pairs (0,2),(0,3),(1,2),(1,3) = 4
        assert len(word) == 4
        # Verify it reconstructs the permutation
        arr = list(range(4))
        for j in word:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
        assert arr == perm

    def test_cyclic_shift_6(self):
        """[3,4,5,0,1,2] — cyclic shift of n=6 by 3."""
        perm = [3, 4, 5, 0, 1, 2]
        word = reduced_word_for_permutation(perm)
        # Inversions: 9
        assert len(word) == 9
        arr = list(range(6))
        for j in word:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
        assert arr == perm

    def test_word_length_is_inversion_count(self):
        """Reduced word length equals number of inversions."""
        for perm in [[1, 0, 2], [2, 0, 1], [2, 1, 0], [3, 1, 2, 0]]:
            word = reduced_word_for_permutation(perm)
            n = len(perm)
            inversions = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
            assert len(word) == inversions, f"perm={perm}"

    def test_reconstruction(self):
        """Every reduced word reconstructs its permutation."""
        perms = [[1, 0], [2, 0, 1], [2, 1, 0], [1, 2, 0], [3, 2, 1, 0]]
        for perm in perms:
            word = reduced_word_for_permutation(perm)
            arr = list(range(len(perm)))
            for j in word:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            assert arr == perm, f"perm={perm}"


class TestPlabicConstruction:
    def _build_from_matroid(self, k: int, n: int) -> PlabicGraph:
        m = uniform_matroid(k, n)
        return plabic_graph_from_matroid(m)

    def test_u24_trip_permutation(self):
        """U(2,4) trip permutation should be [2,3,0,1]."""
        g = self._build_from_matroid(2, 4)
        assert g.trip_permutation() == [2, 3, 0, 1]

    def test_u13_trip_permutation(self):
        """U(1,3) trip permutation should be [1,2,0]."""
        g = self._build_from_matroid(1, 3)
        assert g.trip_permutation() == [1, 2, 0]

    def test_u36_trip_permutation(self):
        """U(3,6) trip permutation should be [3,4,5,0,1,2]."""
        g = self._build_from_matroid(3, 6)
        assert g.trip_permutation() == [3, 4, 5, 0, 1, 2]

    def test_boundary_degree(self):
        """Boundary vertices have degree 1 (lollipops) or 2 (wiring diagram)."""
        g = self._build_from_matroid(2, 4)
        for i in range(g.n):
            deg = len(g.neighbors_cw[i])
            assert deg in (1, 2), f"boundary {i} has degree {deg}"

    def test_all_internal_colored(self):
        """Every internal vertex has a color."""
        g = self._build_from_matroid(2, 4)
        for v in g.vertices.values():
            if not v.is_boundary:
                assert v.color in ("black", "white"), f"vertex {v.id} has no color"

    def test_internal_degree_3(self):
        """Non-lollipop internal vertices have degree 3 (2 strand + 1 crossing)."""
        g = self._build_from_matroid(2, 4)
        for vid, v in g.vertices.items():
            if not v.is_boundary:
                deg = len(g.neighbors_cw[vid])
                assert deg >= 1, f"internal vertex {vid} has degree {deg}"


class TestTripPermutation:
    def test_roundtrip_u24(self):
        """perm -> graph -> trip_perm == perm for U(2,4)."""
        m = uniform_matroid(2, 4)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 4)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        g = plabic_graph_from_decorated_permutation(perm_int, necklace, 4, 2)
        assert g.trip_permutation() == perm_int

    def test_roundtrip_u13(self):
        m = uniform_matroid(1, 3)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 3)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        g = plabic_graph_from_decorated_permutation(perm_int, necklace, 3, 1)
        assert g.trip_permutation() == perm_int

    def test_roundtrip_u23(self):
        m = uniform_matroid(2, 3)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 3)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        g = plabic_graph_from_decorated_permutation(perm_int, necklace, 3, 2)
        assert g.trip_permutation() == perm_int

    def test_roundtrip_u25(self):
        m = uniform_matroid(2, 5)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 5)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        g = plabic_graph_from_decorated_permutation(perm_int, necklace, 5, 2)
        assert g.trip_permutation() == perm_int

    def test_roundtrip_u36(self):
        m = uniform_matroid(3, 6)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 6)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        g = plabic_graph_from_decorated_permutation(perm_int, necklace, 6, 3)
        assert g.trip_permutation() == perm_int


class TestFromMatroid:
    def test_uniform_u24(self):
        g = plabic_graph_from_matroid(uniform_matroid(2, 4))
        assert g.n == 4
        assert g.k == 2
        assert g.trip_permutation() == [2, 3, 0, 1]

    def test_uniform_u13(self):
        g = plabic_graph_from_matroid(uniform_matroid(1, 3))
        assert g.n == 3
        assert g.k == 1
        assert g.trip_permutation() == [1, 2, 0]

    def test_non_uniform_positroid(self):
        """Single removal from U(2,4): remove {0,1}."""
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
        g = plabic_graph_from_matroid(m)
        assert g.n == 4
        assert g.k == 2
        # Trip permutation should match the decorated permutation
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 4)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        assert g.trip_permutation() == perm_int

    def test_non_positroid_raises(self):
        """Non-positroid matroid raises ValueError."""
        bases = frozenset(
            [
                frozenset({0, 1}),
                frozenset({0, 3}),
                frozenset({1, 2}),
                frozenset({2, 3}),
            ]
        )
        m = Matroid(frozenset({0, 1, 2, 3}), bases)
        with pytest.raises(ValueError, match="not a positroid"):
            plabic_graph_from_matroid(m)

    def test_adjacent_parallel_positroid(self):
        """Adjacent parallel pairs: remove {0,1} and {2,3} from U(2,4)."""
        bases = frozenset(
            [
                frozenset({0, 2}),
                frozenset({0, 3}),
                frozenset({1, 2}),
                frozenset({1, 3}),
            ]
        )
        m = Matroid(frozenset({0, 1, 2, 3}), bases)
        g = plabic_graph_from_matroid(m)
        necklace = grassmann_necklace(m)
        perm = decorated_permutation(necklace, 4)
        perm_int = [p if p is not None else i for i, p in enumerate(perm)]
        assert g.trip_permutation() == perm_int
