"""Core matroid class.

A matroid M = (E, B) on a finite ground set E with bases B.
Represented by bases for small matroids (ground set up to ~20 elements).
"""

from itertools import combinations


class Matroid:
    """A matroid on a finite ground set, represented by its collection of bases."""

    def __init__(
        self, ground_set: frozenset[int], bases: frozenset[frozenset[int]]
    ) -> None:
        if not bases:
            raise ValueError("Bases collection must be non-empty")

        sizes = {len(b) for b in bases}
        if len(sizes) != 1:
            raise ValueError(f"All bases must have the same size, got sizes: {sizes}")

        for b in bases:
            if not b.issubset(ground_set):
                raise ValueError(f"Basis {b} is not a subset of ground set {ground_set}")

        self._ground_set = ground_set
        self._bases = bases
        self._rank = next(iter(sizes))

        self._validate_exchange_axiom()

    def _validate_exchange_axiom(self) -> None:
        """Verify the basis exchange axiom.

        For any two bases B1, B2 and any element x in B1 \\ B2,
        there exists y in B2 \\ B1 such that (B1 - {x}) | {y} is a basis.
        """
        for b1 in self._bases:
            for b2 in self._bases:
                for x in b1 - b2:
                    found = False
                    for y in b2 - b1:
                        candidate = (b1 - {x}) | {y}
                        if candidate in self._bases:
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"Basis exchange axiom violated: B1={b1}, B2={b2}, "
                            f"x={x}, no valid y found in B2\\B1={b2 - b1}"
                        )

    @property
    def ground_set(self) -> frozenset[int]:
        return self._ground_set

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def bases(self) -> frozenset[frozenset[int]]:
        return self._bases

    @property
    def size(self) -> int:
        """Size of the ground set."""
        return len(self._ground_set)

    def rank_of(self, subset: frozenset[int]) -> int:
        """Rank of a subset: size of the largest independent set contained in it."""
        max_rank = 0
        for size in range(min(len(subset), self._rank), 0, -1):
            for s in combinations(sorted(subset), size):
                fs = frozenset(s)
                if self.is_independent(fs):
                    return size
            max_rank = max(max_rank, size - 1)
        return 0

    def is_independent(self, subset: frozenset[int]) -> bool:
        """Check if subset is independent (contained in some basis)."""
        if len(subset) > self._rank:
            return False
        if len(subset) == self._rank:
            return subset in self._bases
        # A subset is independent if it can be extended to a basis
        return any(subset.issubset(basis) for basis in self._bases)

    def circuits(self) -> frozenset[frozenset[int]]:
        """All circuits (minimal dependent sets)."""
        result: set[frozenset[int]] = set()
        # A circuit is a minimal set that is not independent
        for size in range(2, self._rank + 2):
            for subset in combinations(sorted(self._ground_set), size):
                fs = frozenset(subset)
                if not self.is_independent(fs):
                    # Check minimality: all proper subsets must be independent
                    is_circuit = all(
                        self.is_independent(fs - {x}) for x in fs
                    )
                    if is_circuit:
                        result.add(fs)
        return frozenset(result)

    def dual(self) -> Matroid:
        """The dual matroid M* (bases are complements of bases of M)."""
        dual_bases = frozenset(self._ground_set - b for b in self._bases)
        return Matroid(self._ground_set, dual_bases)

    def is_uniform(self) -> bool:
        """Check if this is the uniform matroid U(r, n)."""
        from math import comb

        return len(self._bases) == comb(self.size, self._rank)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matroid):
            return NotImplemented
        return self._ground_set == other._ground_set and self._bases == other._bases

    def __hash__(self) -> int:
        return hash((self._ground_set, self._bases))

    def __repr__(self) -> str:
        return (
            f"Matroid(n={self.size}, r={self._rank}, "
            f"|bases|={len(self._bases)})"
        )


def uniform_matroid(k: int, n: int) -> Matroid:
    """Create the uniform matroid U(k, n): every k-element subset is a basis."""
    ground = frozenset(range(n))
    bases = frozenset(frozenset(s) for s in combinations(range(n), k))
    return Matroid(ground, bases)
