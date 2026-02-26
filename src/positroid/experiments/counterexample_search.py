"""Deliberate Counterexample Search for the Activation Positroid Conjecture.

Instead of training networks and hoping for non-uniform matroids, this experiment
constructs TP weight matrices and deliberately chooses biases to create specific
non-uniform matroids — then checks whether they're positroids.

Key insight: for a TP matrix W (H x d), the affine matroid from [W | b] has
rank d+1. We can choose biases to make specific (d+1)-tuples dependent (circuits).
If these circuits "cross" in cyclic order, the resulting matroid is NOT a positroid
by Grassmann necklace theory. The question is whether TP structure prevents
crossing circuits — this experiment tests that computationally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from positroid.arrangement.hyperplane import Hyperplane, HyperplaneArrangement
from positroid.linalg.totally_positive import (
    is_totally_positive,
    random_totally_positive,
    tp_from_cauchy_kernel,
    tp_from_exponential_kernel,
)
from positroid.matroid.positroid import (
    decorated_permutation,
    grassmann_necklace,
    is_positroid,
)


def compute_dependency_coefficients(
    w: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Compute dependency coefficients for each (d+1)-tuple of rows.

    For a H x d weight matrix W, each (d+1)-element subset S of rows is linearly
    dependent (the rows span at most d dimensions). The dependency coefficients c
    satisfy: sum_i c[i] * w[S[i]] = 0.

    The affine matroid has the augmented vectors [w_i | b_i]. The subset S is
    a non-basis (dependent in the augmented matrix) iff sum_i c[i] * b[S[i]] = 0.

    Returns:
        Dict mapping each (d+1)-tuple S to a coefficient array c of length (d+1),
        where c[k] is the coefficient for element S[k].
    """
    h, d = w.shape
    rank = d + 1  # affine matroid rank
    result: dict[tuple[int, ...], np.ndarray] = {}

    for subset in combinations(range(h), rank):
        # Submatrix of W for this subset: (d+1) x d
        sub_w = w[list(subset)]
        # We want c in R^(d+1) such that c @ sub_w = 0 (left null space).
        # sub_w is (d+1) x d with rank d, so left null space is 1-dimensional.
        # Full SVD: sub_w = U @ diag(S) @ V^T where U is (d+1) x (d+1).
        # The last column of U spans the left null space.
        u_full, _, _ = np.linalg.svd(sub_w, full_matrices=True)
        c = u_full[:, -1]

        # Normalize so the first nonzero coefficient is positive
        first_nonzero = np.argmax(np.abs(c) > 1e-12)
        if c[first_nonzero] < 0:
            c = -c

        result[subset] = c

    return result


def solve_bias_for_circuits(
    w: np.ndarray,
    target_circuits: list[tuple[int, ...]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Find biases that make the target subsets non-bases (circuits) of the affine matroid.

    For each target circuit S, the constraint is: sum_k c_S[k] * b[S[k]] = 0
    where c_S are the dependency coefficients. This gives one linear equation
    per target circuit.

    Uses direct substitution: start with random biases, then adjust one bias
    per constraint to satisfy it exactly. This is more numerically stable than
    SVD-based null space computation.

    Returns:
        Bias vector b of length H.
    """
    h = w.shape[0]
    deps = compute_dependency_coefficients(w)

    # Start with random biases (spread out, not near zero)
    b = np.array(rng.uniform(-2.0, 2.0, size=h))

    # For each target circuit, adjust one bias to satisfy the constraint exactly
    # Process constraints in order; for disjoint circuits this is independent
    for circuit in target_circuits:
        coeffs = deps[circuit]
        indices = list(circuit)

        # Current constraint violation: coeffs . b[indices]
        violation = sum(coeffs[k] * b[indices[k]] for k in range(len(indices)))

        # Adjust the element with largest |coefficient| for stability
        adj_k = int(np.argmax(np.abs(coeffs)))
        b[indices[adj_k]] -= violation / coeffs[adj_k]

    return b


def is_crossing_pair(
    c1: tuple[int, ...],
    c2: tuple[int, ...],
    n: int,
) -> bool:
    """Check if two subsets cross in cyclic order on [n].

    Two subsets cross if their elements interleave when placed on a circle
    labeled 0, 1, ..., n-1. Specifically, there is no arc of the circle
    containing all elements of one subset but none of the other.
    """
    s1 = set(c1)
    s2 = set(c2)

    # Place on circle. Check if s2 elements are all on one arc between
    # consecutive s1 elements (or vice versa). If so, they don't cross.
    # They cross iff neither can be separated.
    def is_separable(a: set[int], b: set[int]) -> bool:
        """Check if all elements of b lie in a single arc between consecutive a elements."""
        if not a or not b:
            return True
        a_sorted = sorted(a)
        # Check each arc between consecutive a elements
        for start_idx in range(len(a_sorted)):
            end_idx = (start_idx + 1) % len(a_sorted)
            arc_start = a_sorted[start_idx]
            arc_end = a_sorted[end_idx]

            # Elements in the arc (arc_start, arc_end) in cyclic order
            if arc_end > arc_start:
                arc = {x for x in range(arc_start + 1, arc_end)}
            else:
                arc = {x for x in range(arc_start + 1, n)} | {x for x in range(0, arc_end)}

            if b <= arc:
                return True
        return False

    # They cross if neither s1 separates s2 nor s2 separates s1
    return not is_separable(s1, s2) and not is_separable(s2, s1)


def find_crossing_pairs(
    n: int,
    circuit_size: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Find all pairs of subsets of [n] of given size that cross in cyclic order.

    For the counterexample search, circuit_size = rank of affine matroid = d+1.
    """
    all_subsets = list(combinations(range(n), circuit_size))
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    for i, c1 in enumerate(all_subsets):
        for c2 in all_subsets[i + 1 :]:
            if is_crossing_pair(c1, c2, n):
                pairs.append((c1, c2))

    return pairs


@dataclass
class CounterexampleResult:
    """Result of one counterexample search trial."""

    input_dim: int
    hidden_dim: int
    strategy: str
    tp_kernel: str
    target_circuits: list[tuple[int, ...]] | None

    is_weight_tp: bool
    is_uniform: bool
    is_positroid: bool
    num_bases: int
    total_possible_bases: int
    non_bases: list[tuple[int, ...]]
    grassmann_necklace: tuple[frozenset[int], ...] | None = None
    decorated_perm: list[int | None] | None = None

    @property
    def is_counterexample(self) -> bool:
        return self.is_weight_tp and not self.is_uniform and not self.is_positroid


@dataclass
class SearchSummary:
    """Aggregated search results."""

    results: list[CounterexampleResult] = field(default_factory=list)

    @property
    def num_trials(self) -> int:
        return len(self.results)

    @property
    def num_non_uniform(self) -> int:
        return sum(not r.is_uniform for r in self.results)

    @property
    def num_non_positroid(self) -> int:
        return sum(not r.is_positroid for r in self.results)

    @property
    def num_counterexamples(self) -> int:
        return sum(r.is_counterexample for r in self.results)


def _analyze_matroid(
    w: np.ndarray,
    b: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    strategy: str,
    tp_kernel: str,
    target_circuits: list[tuple[int, ...]] | None,
) -> CounterexampleResult | None:
    """Build the affine matroid from (W, b) and analyze it.

    Returns None if the matroid construction fails due to numerical issues
    (e.g., borderline dependencies that confuse the rank detector).
    """
    hyperplanes = [Hyperplane(normal=w[i].copy(), bias=float(b[i])) for i in range(hidden_dim)]
    arr = HyperplaneArrangement(hyperplanes)
    try:
        aff_mat = arr.affine_matroid()
    except ValueError:
        # Numerical issue in matroid construction (exchange axiom violation
        # due to borderline rank detection). Skip this trial.
        return None

    is_pos = is_positroid(aff_mat)
    necklace = None
    perm = None
    if is_pos:
        necklace = grassmann_necklace(aff_mat)
        perm = decorated_permutation(necklace, aff_mat.size)

    # Compute non-bases
    rank = aff_mat.rank
    all_subsets = {frozenset(s) for s in combinations(range(hidden_dim), rank)}
    non_bases = sorted(
        [tuple(sorted(s)) for s in all_subsets - aff_mat.bases],
    )

    from math import comb

    return CounterexampleResult(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        strategy=strategy,
        tp_kernel=tp_kernel,
        target_circuits=target_circuits,
        is_weight_tp=bool(is_totally_positive(w)),
        is_uniform=aff_mat.is_uniform(),
        is_positroid=is_pos,
        num_bases=len(aff_mat.bases),
        total_possible_bases=comb(hidden_dim, rank),
        non_bases=non_bases,
        grassmann_necklace=necklace,
        decorated_perm=perm,
    )


def targeted_search(
    w: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    tp_kernel: str,
    rng: np.random.Generator,
) -> list[CounterexampleResult]:
    """Search for counterexamples using targeted crossing-circuit biases."""
    rank = input_dim + 1  # affine matroid rank
    crossing_pairs = find_crossing_pairs(hidden_dim, rank)
    results: list[CounterexampleResult] = []

    for c1, c2 in crossing_pairs:
        targets = [c1, c2]
        b = solve_bias_for_circuits(w, targets, rng)
        result = _analyze_matroid(
            w,
            b,
            input_dim,
            hidden_dim,
            "targeted",
            tp_kernel,
            targets,
        )
        if result is not None:
            results.append(result)

    # Also try single crossing circuits (one non-basis)
    all_subsets = list(combinations(range(hidden_dim), rank))
    for subset in all_subsets[:20]:  # limit to first 20 for speed
        targets = [subset]
        b = solve_bias_for_circuits(w, targets, rng)
        result = _analyze_matroid(
            w,
            b,
            input_dim,
            hidden_dim,
            "targeted_single",
            tp_kernel,
            targets,
        )
        if result is not None and not result.is_uniform:
            results.append(result)

    return results


def random_search(
    w: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    tp_kernel: str,
    num_trials: int,
    rng: np.random.Generator,
) -> list[CounterexampleResult]:
    """Search with random biases — baseline comparison."""
    results: list[CounterexampleResult] = []

    for _ in range(num_trials):
        b = rng.uniform(-2.0, 2.0, size=hidden_dim)
        result = _analyze_matroid(
            w,
            b,
            input_dim,
            hidden_dim,
            "random",
            tp_kernel,
            None,
        )
        if result is not None:
            results.append(result)

    return results


def _make_tp_matrix(
    hidden_dim: int,
    input_dim: int,
    kernel: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a TP matrix using the specified kernel."""
    if kernel == "exponential":
        a = np.sort(rng.uniform(0.5, 2.0, size=hidden_dim))
        for i in range(1, hidden_dim):
            if a[i] <= a[i - 1]:
                a[i] = a[i - 1] + 0.01
        b = np.sort(rng.uniform(0.5, 2.0, size=input_dim))
        for i in range(1, input_dim):
            if b[i] <= b[i - 1]:
                b[i] = b[i - 1] + 0.01
        return tp_from_exponential_kernel(a, b)
    elif kernel == "cauchy":
        a = np.sort(rng.uniform(1.0, 5.0, size=hidden_dim))
        for i in range(1, hidden_dim):
            if a[i] <= a[i - 1]:
                a[i] = a[i - 1] + 0.1
        b = np.sort(rng.uniform(1.0, 5.0, size=input_dim))
        for i in range(1, input_dim):
            if b[i] <= b[i - 1]:
                b[i] = b[i - 1] + 0.1
        return tp_from_cauchy_kernel(a, b)
    else:
        return random_totally_positive(hidden_dim, input_dim, rng=rng)


def run_counterexample_search(
    configs: list[tuple[int, int]],
    num_matrices: int = 10,
    strategies: list[str] | None = None,
    num_random_trials: int = 50,
    seed: int = 42,
    kernels: list[str] | None = None,
) -> SearchSummary:
    """Run the full counterexample search.

    Args:
        configs: List of (input_dim, hidden_dim) pairs.
        num_matrices: Number of TP matrices to generate per config per kernel.
        strategies: Which strategies to use. Default: ["targeted", "random"].
        num_random_trials: Number of random bias trials per matrix.
        seed: Random seed.
        kernels: Which TP kernels to use. Default: ["exponential", "cauchy"].
    """
    if strategies is None:
        strategies = ["targeted", "random"]
    if kernels is None:
        kernels = ["exponential", "cauchy"]

    rng = np.random.default_rng(seed)
    summary = SearchSummary()

    for input_dim, hidden_dim in configs:
        for kernel in kernels:
            for _ in range(num_matrices):
                w = _make_tp_matrix(hidden_dim, input_dim, kernel, rng)

                if "targeted" in strategies:
                    results = targeted_search(
                        w,
                        input_dim,
                        hidden_dim,
                        kernel,
                        rng,
                    )
                    summary.results.extend(results)

                if "random" in strategies:
                    results = random_search(
                        w,
                        input_dim,
                        hidden_dim,
                        kernel,
                        num_random_trials,
                        rng,
                    )
                    summary.results.extend(results)

    return summary


def print_summary(summary: SearchSummary) -> None:
    """Print search summary."""
    print("\n" + "=" * 80)
    print("COUNTEREXAMPLE SEARCH RESULTS")
    print("=" * 80)
    print(f"\nTotal trials: {summary.num_trials}")
    print(f"Non-uniform matroids: {summary.num_non_uniform}")
    print(f"Non-positroid matroids: {summary.num_non_positroid}")
    print(f"COUNTEREXAMPLES (TP + non-uniform + non-positroid): {summary.num_counterexamples}")

    if summary.num_counterexamples > 0:
        print("\n*** CONJECTURE IS FALSE ***")
        print("\nCounterexample details:")
        for r in summary.results:
            if r.is_counterexample:
                print(
                    f"\n  d={r.input_dim}, H={r.hidden_dim}, "
                    f"kernel={r.tp_kernel}, strategy={r.strategy}"
                )
                print(f"  Bases: {r.num_bases}/{r.total_possible_bases}")
                print(f"  Non-bases: {r.non_bases}")
                if r.target_circuits:
                    print(f"  Target circuits: {r.target_circuits}")
    else:
        print("\nNo counterexamples found. Conjecture survives.")

    # Breakdown by strategy
    strategies = {r.strategy for r in summary.results}
    print("\nBreakdown by strategy:")
    for strat in sorted(strategies):
        strat_results = [r for r in summary.results if r.strategy == strat]
        non_unif = sum(not r.is_uniform for r in strat_results)
        non_pos = sum(not r.is_positroid for r in strat_results)
        cex = sum(r.is_counterexample for r in strat_results)
        print(
            f"  {strat:>20}: {len(strat_results):>5} trials, "
            f"{non_unif:>4} non-uniform, "
            f"{non_pos:>4} non-positroid, "
            f"{cex:>4} counterexamples"
        )


def print_detailed(summary: SearchSummary) -> None:
    """Print detailed results for all non-uniform matroids."""
    non_uniform = [r for r in summary.results if not r.is_uniform]
    if not non_uniform:
        print("\nNo non-uniform matroids found.")
        return

    print(f"\n{'=' * 80}")
    print(f"DETAILED: {len(non_uniform)} non-uniform matroids")
    print("=" * 80)

    for r in non_uniform:
        status = "COUNTEREXAMPLE" if r.is_counterexample else "positroid"
        print(
            f"\n--- [{status}] d={r.input_dim}, H={r.hidden_dim}, "
            f"kernel={r.tp_kernel}, strategy={r.strategy} ---"
        )
        print(f"  TP weight: {r.is_weight_tp}")
        print(f"  Positroid: {r.is_positroid}")
        print(f"  Bases: {r.num_bases}/{r.total_possible_bases}")
        print(f"  Non-bases: {r.non_bases}")
        if r.target_circuits:
            print(f"  Target circuits: {r.target_circuits}")
        if r.decorated_perm:
            print(f"  Decorated perm: {r.decorated_perm}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterexample Search for Activation Positroid Conjecture",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["2,5", "2,6", "2,8", "2,10"],
        help="Configs as 'input_dim,hidden_dim' pairs",
    )
    parser.add_argument("--num-matrices", type=int, default=10)
    parser.add_argument("--num-random", type=int, default=50)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["targeted", "random"],
        choices=["targeted", "random"],
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["exponential", "cauchy"],
        choices=["exponential", "cauchy"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    configs = [tuple(int(x) for x in c.split(",")) for c in args.configs]

    print("Searching for counterexamples to the Activation Positroid Conjecture...")
    print(f"Configs: {configs}")
    print(f"Matrices per config per kernel: {args.num_matrices}")
    print(f"Strategies: {args.strategies}")
    print(f"Kernels: {args.kernels}")

    summary = run_counterexample_search(
        configs=[(d, h) for d, h in configs],
        num_matrices=args.num_matrices,
        strategies=args.strategies,
        num_random_trials=args.num_random,
        seed=args.seed,
        kernels=args.kernels,
    )

    print_summary(summary)

    if args.detailed:
        print_detailed(summary)


if __name__ == "__main__":
    main()
