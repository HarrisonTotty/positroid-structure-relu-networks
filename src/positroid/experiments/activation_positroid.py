"""Activation Positroid Test — baseline experiment with random biases.

Originally tested the Activation Positroid Conjecture (TP weights imply
positroid affine matroid). That conjecture is FALSE — see counterexample_search
for the disproof. With random (non-trained) biases and TP weights, the affine
matroid is usually uniform and thus trivially a positroid, but deliberately
chosen biases can break the positroid property.

This experiment now serves as a baseline: it shows that random biases mostly
produce uniform matroids, in contrast with trained biases (see trained_positroid)
which produce richer non-uniform matroids that are still positroids.

Key insight: For a single hidden layer with H > n and TP weights, the LINEAR
matroid (from normal vectors alone) is always U(n, H), which is trivially a
positroid. The interesting test is the AFFINE matroid (augmenting normals with
biases), which captures the full combinatorial structure of the arrangement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from positroid.linalg.totally_positive import is_totally_positive
from positroid.matroid.positroid import decorated_permutation, grassmann_necklace, is_positroid


@dataclass
class TrialResult:
    """Result of a single trial of the Activation Positroid Test."""

    input_dim: int
    hidden_dim: int
    is_weight_tp: bool

    linear_matroid_rank: int
    linear_matroid_num_bases: int
    linear_matroid_is_uniform: bool
    linear_matroid_is_positroid: bool

    affine_matroid_rank: int
    affine_matroid_num_bases: int
    affine_matroid_is_uniform: bool
    affine_matroid_is_positroid: bool
    affine_grassmann_necklace: tuple[frozenset[int], ...] | None = None
    affine_decorated_permutation: list[int | None] | None = None


@dataclass
class ExperimentResult:
    """Aggregated results for one configuration."""

    input_dim: int
    hidden_dim: int
    weight_type: str  # 'tp' or 'random'
    num_trials: int
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def linear_positroid_count(self) -> int:
        return sum(t.linear_matroid_is_positroid for t in self.trials)

    @property
    def affine_positroid_count(self) -> int:
        return sum(t.affine_matroid_is_positroid for t in self.trials)

    @property
    def linear_positroid_rate(self) -> float:
        return self.linear_positroid_count / self.num_trials if self.num_trials > 0 else 0

    @property
    def affine_positroid_rate(self) -> float:
        return self.affine_positroid_count / self.num_trials if self.num_trials > 0 else 0


def run_single_trial(
    input_dim: int,
    hidden_dim: int,
    tp_weights: bool,
    rng: np.random.Generator,
) -> TrialResult:
    """Run a single trial: create network, extract arrangement, test positroid."""
    from positroid.arrangement.hyperplane import Hyperplane, HyperplaneArrangement
    from positroid.linalg.totally_positive import random_totally_positive

    # Generate weight matrix
    if tp_weights:
        w = random_totally_positive(hidden_dim, input_dim, rng=rng)
    else:
        w = rng.standard_normal((hidden_dim, input_dim))

    b = rng.uniform(-1.0, 1.0, size=hidden_dim)

    # Build hyperplane arrangement
    hyperplanes = [
        Hyperplane(normal=w[i].copy(), bias=float(b[i])) for i in range(hidden_dim)
    ]
    arr = HyperplaneArrangement(hyperplanes)

    # Linear matroid (normals only)
    lin_mat = arr.linear_matroid()
    lin_is_pos = is_positroid(lin_mat)

    # Affine matroid (normals + biases)
    aff_mat = arr.affine_matroid()
    aff_is_pos = is_positroid(aff_mat)

    # Compute necklace and permutation if positroid
    aff_necklace = None
    aff_perm = None
    if aff_is_pos:
        aff_necklace = grassmann_necklace(aff_mat)
        aff_perm = decorated_permutation(aff_necklace, aff_mat.size)

    return TrialResult(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        is_weight_tp=is_totally_positive(w) if tp_weights else False,
        linear_matroid_rank=lin_mat.rank,
        linear_matroid_num_bases=len(lin_mat.bases),
        linear_matroid_is_uniform=lin_mat.is_uniform(),
        linear_matroid_is_positroid=lin_is_pos,
        affine_matroid_rank=aff_mat.rank,
        affine_matroid_num_bases=len(aff_mat.bases),
        affine_matroid_is_uniform=aff_mat.is_uniform(),
        affine_matroid_is_positroid=aff_is_pos,
        affine_grassmann_necklace=aff_necklace,
        affine_decorated_permutation=aff_perm,
    )


def run_experiment(
    input_dim: int,
    hidden_dim: int,
    num_trials: int,
    tp_weights: bool,
    seed: int = 42,
) -> ExperimentResult:
    """Run the experiment for one configuration."""
    rng = np.random.default_rng(seed)
    weight_type = "tp" if tp_weights else "random"
    result = ExperimentResult(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        weight_type=weight_type,
        num_trials=num_trials,
    )

    for _ in range(num_trials):
        trial = run_single_trial(input_dim, hidden_dim, tp_weights, rng)
        result.trials.append(trial)

    return result


def print_results(results: list[ExperimentResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 80)
    print("ACTIVATION POSITROID TEST RESULTS")
    print("=" * 80)
    print()
    print(
        f"{'H':>3}  {'Type':>6}  {'Trials':>6}  "
        f"{'Lin Pos':>8}  {'Lin Rate':>8}  "
        f"{'Aff Pos':>8}  {'Aff Rate':>8}  "
        f"{'Aff Uniform':>11}"
    )
    print("-" * 80)

    for r in results:
        aff_uniform_count = sum(t.affine_matroid_is_uniform for t in r.trials)
        print(
            f"{r.hidden_dim:>3}  {r.weight_type:>6}  {r.num_trials:>6}  "
            f"{r.linear_positroid_count:>8}  {r.linear_positroid_rate:>8.1%}  "
            f"{r.affine_positroid_count:>8}  {r.affine_positroid_rate:>8.1%}  "
            f"{aff_uniform_count:>11}"
        )

    print()
    print("Legend:")
    print("  H          = hidden dimension")
    print("  Lin Pos    = # trials where linear matroid is a positroid")
    print("  Aff Pos    = # trials where affine matroid is a positroid")
    print("  Aff Uniform = # trials where affine matroid is the uniform matroid")
    print()


def print_detailed_results(results: list[ExperimentResult]) -> None:
    """Print detailed per-trial information for interesting cases."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS (non-positroid affine matroids)")
    print("=" * 80)

    found_any = False
    for r in results:
        for i, t in enumerate(r.trials):
            if not t.affine_matroid_is_positroid:
                found_any = True
                print(f"\n--- Trial {i} (H={t.hidden_dim}, {r.weight_type}) ---")
                print(f"  Weight TP: {t.is_weight_tp}")
                print(f"  Linear matroid: rank={t.linear_matroid_rank}, "
                      f"|bases|={t.linear_matroid_num_bases}, "
                      f"uniform={t.linear_matroid_is_uniform}, "
                      f"positroid={t.linear_matroid_is_positroid}")
                print(f"  Affine matroid: rank={t.affine_matroid_rank}, "
                      f"|bases|={t.affine_matroid_num_bases}, "
                      f"uniform={t.affine_matroid_is_uniform}, "
                      f"positroid={t.affine_matroid_is_positroid}")

    if not found_any:
        print("\n  All affine matroids were positroids!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation Positroid Test")
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[3, 4, 5, 6],
        help="Hidden dimensions to test",
    )
    parser.add_argument("--input-dim", type=int, default=2, help="Input dimension")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--detailed", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    all_results: list[ExperimentResult] = []

    for h in args.hidden_dims:
        print(f"Running H={h}, TP weights...", end=" ", flush=True)
        tp_result = run_experiment(
            args.input_dim, h, args.num_trials, tp_weights=True, seed=args.seed,
        )
        print(f"done. Affine positroid rate: {tp_result.affine_positroid_rate:.1%}")
        all_results.append(tp_result)

        print(f"Running H={h}, random weights...", end=" ", flush=True)
        rand_result = run_experiment(
            args.input_dim, h, args.num_trials, tp_weights=False, seed=args.seed + 1,
        )
        print(f"done. Affine positroid rate: {rand_result.affine_positroid_rate:.1%}")
        all_results.append(rand_result)

    print_results(all_results)

    if args.detailed:
        print_detailed_results(all_results)


if __name__ == "__main__":
    main()
