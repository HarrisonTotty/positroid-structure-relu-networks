"""Plabic graph construction from positroid decorated permutations.

Builds reduced plabic graphs using the wiring diagram approach:
decompose the decorated permutation into adjacent transpositions,
then construct the bipartite planar graph from crossing data.

References:
  Postnikov, "Total positivity, Grassmannians, and networks" (2006)
"""

from __future__ import annotations

from dataclasses import dataclass

from positroid.matroid.matroid import Matroid
from positroid.matroid.positroid import (
    decorated_permutation,
    grassmann_necklace,
    is_positroid,
)


@dataclass(frozen=True)
class PlabicVertex:
    """A vertex in a plabic graph."""

    id: int
    color: str | None  # 'black', 'white', or None (boundary)
    is_boundary: bool


class PlabicGraph:
    """A plabic graph with planar embedding.

    Boundary vertices are numbered 0..n-1. Internal vertices have ids >= n.
    Each vertex stores its neighbors in clockwise order (planar embedding).
    """

    def __init__(
        self,
        n: int,
        k: int,
        vertices: dict[int, PlabicVertex],
        neighbors_cw: dict[int, list[int]],
    ) -> None:
        self.n = n
        self.k = k
        self.vertices = vertices
        self.neighbors_cw = neighbors_cw

    def trip(self, start: int) -> int:
        """Follow a trip (strand) from boundary vertex `start`.

        At black vertices turn right (next CW neighbor).
        At white vertices turn left (next CCW neighbor).
        Returns the boundary vertex where the trip ends.
        """
        if not self.vertices[start].is_boundary:
            raise ValueError(f"Vertex {start} is not a boundary vertex")

        prev = start
        current = self.neighbors_cw[start][0]  # follow outgoing edge

        while not self.vertices[current].is_boundary:
            nbrs = self.neighbors_cw[current]
            entry_idx = nbrs.index(prev)
            if self.vertices[current].color == "black":
                exit_idx = (entry_idx + 1) % len(nbrs)  # turn right (CW)
            else:
                exit_idx = (entry_idx - 1) % len(nbrs)  # turn left (CCW)
            prev = current
            current = nbrs[exit_idx]

        return current

    def trip_permutation(self) -> list[int]:
        """Compute the full trip permutation pi where pi[i] = trip(i)."""
        return [self.trip(i) for i in range(self.n)]

    @property
    def edges(self) -> list[tuple[int, int]]:
        """All edges as (u, v) pairs with u < v."""
        seen: set[tuple[int, int]] = set()
        for u, nbrs in self.neighbors_cw.items():
            for v in nbrs:
                edge = (min(u, v), max(u, v))
                seen.add(edge)
        return sorted(seen)

    @property
    def num_internal(self) -> int:
        """Number of internal (non-boundary) vertices."""
        return sum(1 for v in self.vertices.values() if not v.is_boundary)

    def __repr__(self) -> str:
        return (
            f"PlabicGraph(n={self.n}, k={self.k}, "
            f"internal={self.num_internal}, edges={len(self.edges)})"
        )


def reduced_word_for_permutation(perm: list[int]) -> list[int]:
    """Decompose a permutation into adjacent transpositions via bubble sort.

    Returns a reduced word [j_1, j_2, ...] where s_{j_i} swaps positions
    j_i and j_i+1. Applying left-to-right reconstructs the permutation.
    The length equals the number of inversions.
    """
    n = len(perm)
    arr = list(perm)
    word: list[int] = []

    # Bubble sort: each swap records a transposition
    # This sorts perm -> identity, so we reverse to get identity -> perm
    changed = True
    while changed:
        changed = False
        for i in range(n - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                word.append(i)
                changed = True

    word.reverse()
    return word


def plabic_graph_from_decorated_permutation(
    perm: list[int],
    necklace: tuple[frozenset[int], ...],
    n: int,
    k: int,
) -> PlabicGraph:
    """Build a plabic graph from a decorated permutation.

    Uses the wiring diagram construction: decompose perm into adjacent
    transpositions, create black/white vertex pairs at each crossing,
    then set the planar embedding algebraically for correct trip permutation.

    At each crossing, the upper wire visits a black vertex and the lower
    wire visits a white vertex. The two are connected by a crossing edge.
    The CW neighbor ordering is:
      - BLACK: [left_strand, right_strand, crossing_partner]
      - WHITE: [left_strand, crossing_partner, right_strand]
    ensuring the trip turn rules follow strands correctly.
    """
    vertices: dict[int, PlabicVertex] = {}
    neighbors_cw: dict[int, list[int]] = {}

    # Boundary vertices 0..n-1
    for i in range(n):
        vertices[i] = PlabicVertex(id=i, color=None, is_boundary=True)
        neighbors_cw[i] = []

    # Classify fixed points
    fixed_loops: set[int] = set()  # loops: i not in I_i
    fixed_coloops: set[int] = set()  # coloops: i in I_i and i in I_{i+1}
    for i in range(n):
        if perm[i] == i:
            if i not in necklace[i]:
                fixed_loops.add(i)
            else:
                fixed_coloops.add(i)

    word = reduced_word_for_permutation(perm)
    next_id = n

    if not word:
        # Pure fixed-point permutation: only lollipops
        for i in range(n):
            if i in fixed_loops:
                vid = next_id
                next_id += 1
                vertices[vid] = PlabicVertex(id=vid, color="black", is_boundary=False)
                neighbors_cw[vid] = [i]
                neighbors_cw[i] = [vid]
            elif i in fixed_coloops:
                vid = next_id
                next_id += 1
                vertices[vid] = PlabicVertex(id=vid, color="white", is_boundary=False)
                neighbors_cw[vid] = [i]
                neighbors_cw[i] = [vid]
        return PlabicGraph(n, k, vertices, neighbors_cw)

    # Build wiring diagram with black/white vertex pairs at each crossing.
    # Each wire visits exactly ONE internal vertex per crossing:
    #   upper wire -> black vertex, lower wire -> white vertex.
    wire_at = list(range(n))

    # wire_path[wire] = [boundary_start, internal_v1, internal_v2, ..., boundary_end]
    wire_path: dict[int, list[int]] = {i: [i] for i in range(n)}

    # Track crossing partners: crossing_partner[b] = w, crossing_partner[w] = b
    crossing_partner: dict[int, int] = {}

    for j in word:
        upper_wire = wire_at[j]
        lower_wire = wire_at[j + 1]

        b_id = next_id
        w_id = next_id + 1
        next_id += 2

        vertices[b_id] = PlabicVertex(id=b_id, color="black", is_boundary=False)
        vertices[w_id] = PlabicVertex(id=w_id, color="white", is_boundary=False)

        crossing_partner[b_id] = w_id
        crossing_partner[w_id] = b_id

        # Upper wire visits black vertex, lower wire visits white vertex
        wire_path[upper_wire].append(b_id)
        wire_path[lower_wire].append(w_id)

        wire_at[j], wire_at[j + 1] = wire_at[j + 1], wire_at[j]

    # Complete wire paths with boundary endpoints
    for wire in range(n):
        wire_path[wire].append(perm[wire])

    # Build CW neighbor orderings algebraically.
    # For each internal vertex, find its position in its wire path.
    vertex_wire: dict[int, int] = {}
    vertex_pos: dict[int, int] = {}
    for wire, path in wire_path.items():
        for idx, vid in enumerate(path):
            if not vertices[vid].is_boundary:
                vertex_wire[vid] = wire
                vertex_pos[vid] = idx

    for vid in list(vertices):
        v = vertices[vid]
        if v.is_boundary:
            continue
        if vid not in vertex_wire:
            # Lollipop vertex (handled later)
            continue

        wire = vertex_wire[vid]
        pos = vertex_pos[vid]
        path = wire_path[wire]

        left_nbr = path[pos - 1]  # previous vertex on strand
        right_nbr = path[pos + 1]  # next vertex on strand
        partner = crossing_partner[vid]

        if v.color == "black":
            # CW: [left_strand, right_strand, crossing_partner]
            neighbors_cw[vid] = [left_nbr, right_nbr, partner]
        else:
            # CW: [left_strand, crossing_partner, right_strand]
            neighbors_cw[vid] = [left_nbr, partner, right_nbr]

    # Handle fixed points (lollipops)
    for i in range(n):
        if i in fixed_loops:
            vid = next_id
            next_id += 1
            vertices[vid] = PlabicVertex(id=vid, color="black", is_boundary=False)
            neighbors_cw[vid] = [i]
            # Add to boundary's outgoing (will be placed first in CW order below)
            wire_path[i] = [i, vid, i]
        elif i in fixed_coloops:
            vid = next_id
            next_id += 1
            vertices[vid] = PlabicVertex(id=vid, color="white", is_boundary=False)
            neighbors_cw[vid] = [i]
            wire_path[i] = [i, vid, i]

    # Set boundary CW orderings.
    # Boundary i is the start of wire i and the end of wire sigma^{-1}(i).
    # CW order: [outgoing_first_vertex, incoming_last_vertex]
    # (outgoing = first internal vertex on wire i,
    #  incoming = last internal vertex on wire ending at i)
    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i

    for i in range(n):
        outgoing_wire = i
        incoming_wire = inv_perm[i]

        out_path = wire_path[outgoing_wire]
        in_path = wire_path[incoming_wire]

        outgoing_nbr = out_path[1] if len(out_path) > 1 else None
        incoming_nbr = in_path[-2] if len(in_path) > 1 else None

        if outgoing_nbr is not None and incoming_nbr is not None:
            if outgoing_nbr == incoming_nbr:
                # Same vertex: degree 1
                neighbors_cw[i] = [outgoing_nbr]
            else:
                neighbors_cw[i] = [outgoing_nbr, incoming_nbr]
        elif outgoing_nbr is not None:
            neighbors_cw[i] = [outgoing_nbr]
        elif incoming_nbr is not None:
            neighbors_cw[i] = [incoming_nbr]

    return PlabicGraph(n, k, vertices, neighbors_cw)


def plabic_graph_from_matroid(matroid: Matroid) -> PlabicGraph:
    """Build a plabic graph from a positroid matroid.

    Raises ValueError if the matroid is not a positroid.
    """
    if not is_positroid(matroid):
        raise ValueError("Matroid is not a positroid")

    necklace = grassmann_necklace(matroid)
    perm = decorated_permutation(necklace, matroid.size)

    # decorated_permutation returns list[int | None]; for positroids all defined
    perm_int = [p if p is not None else i for i, p in enumerate(perm)]

    return plabic_graph_from_decorated_permutation(perm_int, necklace, matroid.size, matroid.rank)
