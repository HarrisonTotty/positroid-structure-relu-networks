"""Positroid verification via Grassmann necklaces.

A matroid is a positroid if it can be realized by a point in the totally
nonnegative Grassmannian Gr_+(k, n). Positroids are characterized by
their Grassmann necklace — a sequence I = (I_0, ..., I_{n-1}) where I_j
is the lexicographically smallest basis in the cyclic order starting at j.
"""

from itertools import combinations

from positroid.matroid.matroid import Matroid


def _cyclic_order(start: int, n: int) -> list[int]:
    """Return elements 0..n-1 in cyclic order starting at `start`."""
    return [(start + i) % n for i in range(n)]


def _lex_compare_cyclic(a: frozenset[int], b: frozenset[int], order: list[int]) -> int:
    """Compare two sets lexicographically in the given cyclic order.

    Returns -1 if a < b, 0 if a == b, 1 if a > b.
    """
    pos = {elem: i for i, elem in enumerate(order)}
    sa = sorted(a, key=lambda x: pos[x])
    sb = sorted(b, key=lambda x: pos[x])
    for x, y in zip(sa, sb, strict=True):
        if pos[x] < pos[y]:
            return -1
        if pos[x] > pos[y]:
            return 1
    return 0


def lex_min_basis_cyclic(matroid: Matroid, start: int) -> frozenset[int]:
    """Find the lex-min basis of the matroid in cyclic order starting at `start`.

    Uses the greedy algorithm: iterate through elements in cyclic order,
    greedily adding elements that maintain independence.
    """
    n = matroid.size
    order = _cyclic_order(start, n)

    # Greedy: go through elements in cyclic order, add if independent
    selected: list[int] = []
    for elem in order:
        candidate = frozenset(selected + [elem])
        if matroid.is_independent(candidate):
            selected.append(elem)
            if len(selected) == matroid.rank:
                break

    return frozenset(selected)


def grassmann_necklace(matroid: Matroid) -> tuple[frozenset[int], ...]:
    """Compute the Grassmann necklace of a matroid.

    The Grassmann necklace is I = (I_0, I_1, ..., I_{n-1}) where I_j is
    the lexicographically smallest basis in the cyclic order starting at j.

    This is well-defined for any matroid; the matroid is a positroid iff
    the necklace determines exactly the same set of bases.
    """
    n = matroid.size
    return tuple(lex_min_basis_cyclic(matroid, j) for j in range(n))


def bases_from_grassmann_necklace(
    necklace: tuple[frozenset[int], ...],
    n: int,
    k: int,
) -> frozenset[frozenset[int]]:
    """Reconstruct the bases of a positroid from its Grassmann necklace.

    A k-element subset B of {0, ..., n-1} is a basis of the positroid
    with necklace I = (I_0, ..., I_{n-1}) iff for all j, B >=_j I_j
    in the Gale order with respect to the cyclic order starting at j.

    The Gale order >=_j means: writing B = {b_1 <_j ... <_j b_k} and
    I_j = {i_1 <_j ... <_j i_k} in the cyclic order starting at j,
    we need b_l >=_j i_l for all l = 1, ..., k.
    """
    bases: set[frozenset[int]] = set()

    for subset in combinations(range(n), k):
        b = frozenset(subset)
        is_basis = True
        for j in range(n):
            order = _cyclic_order(j, n)
            pos = {elem: idx for idx, elem in enumerate(order)}

            b_sorted = sorted(b, key=lambda x: pos[x])
            ij_sorted = sorted(necklace[j], key=lambda x: pos[x])

            for bl, il in zip(b_sorted, ij_sorted, strict=True):
                if pos[bl] < pos[il]:
                    is_basis = False
                    break
            if not is_basis:
                break

        if is_basis:
            bases.add(b)

    return frozenset(bases)


def is_positroid(matroid: Matroid) -> bool:
    """Check if a matroid is a positroid.

    Algorithm:
    1. Compute the Grassmann necklace I = (I_0, ..., I_{n-1}).
    2. Reconstruct the set of bases from the Grassmann necklace.
    3. Check if the reconstructed bases equal the original matroid's bases.
    """
    necklace = grassmann_necklace(matroid)
    reconstructed = bases_from_grassmann_necklace(necklace, matroid.size, matroid.rank)
    return reconstructed == matroid.bases


def is_cyclic_interval(subset: frozenset[int] | tuple[int, ...], n: int) -> bool:
    """Check if a subset of [n] forms a cyclic interval (contiguous arc).

    A cyclic interval on {0, ..., n-1} is a set of the form
    {j, j+1, ..., j+k-1} mod n for some starting index j.

    Equivalently, when elements are placed on a circle, they form a
    contiguous arc with no gaps.
    """
    elements = sorted(subset)
    k = len(elements)
    if k <= 1 or k >= n:
        return True
    # Count consecutive gaps of size 1 in cyclic order.
    # A cyclic interval of size k has exactly k-1 gaps of size 1
    # (and one "big gap" covering the rest of the circle).
    gaps = [(elements[(i + 1) % k] - elements[i]) % n for i in range(k)]
    return gaps.count(1) >= k - 1


def has_only_cyclic_interval_nonbases(matroid: Matroid) -> bool:
    """Check if every non-basis of the matroid is a cyclic interval.

    This is the hypothesis of the Contiguous-Implies-Positroid theorem:
    if all non-bases are cyclic intervals, then the matroid is a positroid.
    """
    n = matroid.size
    k = matroid.rank
    all_k_subsets = frozenset(frozenset(s) for s in combinations(range(n), k))
    non_bases = all_k_subsets - matroid.bases
    return all(is_cyclic_interval(nb, n) for nb in non_bases)


def nonbase_support(matroid: Matroid) -> frozenset[int]:
    """Union of all elements appearing in any non-basis.

    Returns frozenset() for uniform matroids (no non-bases).
    """
    n = matroid.size
    k = matroid.rank
    all_k_subsets = frozenset(frozenset(s) for s in combinations(range(n), k))
    non_bases = all_k_subsets - matroid.bases
    if not non_bases:
        return frozenset()
    return frozenset().union(*non_bases)


def support_is_cyclic_interval(matroid: Matroid) -> bool:
    """Check if the non-basis support forms a cyclic interval.

    Returns True for uniform matroids (empty support is trivially an interval).
    """
    support = nonbase_support(matroid)
    if not support:
        return True
    return is_cyclic_interval(support, matroid.size)


def support_rank_deficiency(matroid: Matroid) -> int:
    """Rank deficiency of the non-basis support: k - rank(support).

    Returns 0 for uniform matroids. Positive means the support is
    rank-deficient, which is the key condition in the Contiguous-Support
    Positroid Theorem.
    """
    support = nonbase_support(matroid)
    if not support:
        return 0
    return matroid.rank - matroid.rank_of(support)


def decorated_permutation(necklace: tuple[frozenset[int], ...], n: int) -> list[int | None]:
    """Compute the decorated permutation from a Grassmann necklace.

    For j = 0, ..., n-1:
    - If j in I_j and j not in I_{(j+1) % n}: pi(j) = the element in
      I_{(j+1) % n} \\ I_j (the element that replaces j).
    - If j not in I_j: pi(j) = j (a fixed point, decorated as a "loop").
    - If j in I_j and j in I_{(j+1) % n}: pi(j) = j (a "coloop").

    Returns a list where pi[j] is the image of j, or None if undefined.
    """
    perm: list[int | None] = [None] * n

    for j in range(n):
        j_next = (j + 1) % n
        if j not in necklace[j]:
            # j is a loop: fixed point
            perm[j] = j
        elif j not in necklace[j_next]:
            # j was in I_j but not in I_{j+1}: it got replaced
            replacement = necklace[j_next] - necklace[j]
            if replacement:
                perm[j] = next(iter(replacement))
            else:
                perm[j] = j
        else:
            # j is in both I_j and I_{j+1}: coloop
            perm[j] = j

    return perm
