"""Non-TP Baseline Experiment.

Tests whether the positroid phenomenon in trained networks is about TP structure,
gradient descent dynamics, or their interaction. Compares:
1. TP-constrained training (exponential/Cauchy kernels)
2. Non-TP kernel training (sinusoidal/quadratic distance)
3. Unconstrained training (Xavier init, general position)

Key question: if non-TP constrained trained networks also only produce positroids,
the phenomenon is purely about gradient descent. If they sometimes produce
non-positroids, TP structure matters jointly with training dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from positroid.datasets import DATASETS
from positroid.linalg.totally_positive import is_totally_positive
from positroid.matroid.positroid import (
    decorated_permutation,
    grassmann_necklace,
    is_cyclic_interval,
    is_positroid,
)
from positroid.network.relu_network import ReluNetwork
from positroid.network.train import TrainConfig, train

_MODE_SEED_OFFSET: dict[str, int] = {
    "tp_exponential": 0,
    "tp_cauchy": 1000,
    "sinusoidal": 2000,
    "quadratic_distance": 3000,
    "unconstrained": 4000,
    "permuted_exponential": 5000,
    "negated_bidiagonal": 6000,
    "fixed_convergent_bias_only": 7000,
}


@dataclass
class BaselineTrialResult:
    """Result of one non-TP baseline trial."""

    dataset_name: str
    hidden_dim: int
    param_mode: str

    final_loss: float
    final_accuracy: float

    is_weight_tp: bool

    affine_matroid_rank: int
    affine_matroid_num_bases: int
    affine_matroid_is_uniform: bool
    affine_matroid_is_positroid: bool

    non_bases: list[tuple[int, ...]]
    has_non_interval_nonbases: bool

    affine_grassmann_necklace: tuple[frozenset[int], ...] | None = None
    affine_decorated_permutation: list[int | None] | None = None


@dataclass
class BaselineExperimentResult:
    """Aggregated results across parameter modes."""

    dataset_name: str
    hidden_dim: int
    num_trials: int
    trials_by_mode: dict[str, list[BaselineTrialResult]] = field(default_factory=dict)

    def positroid_rate(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return sum(t.affine_matroid_is_positroid for t in trials) / len(trials)

    def uniform_rate(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return sum(t.affine_matroid_is_uniform for t in trials) / len(trials)

    def mean_accuracy(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return float(np.mean([t.final_accuracy for t in trials]))

    def non_interval_rate(self, mode: str) -> float:
        """Rate of trials with at least one non-cyclic-interval non-basis."""
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return sum(t.has_non_interval_nonbases for t in trials) / len(trials)


def analyze_trial(
    net: ReluNetwork,
    dataset_name: str,
    param_mode: str,
    final_loss: float,
    final_accuracy: float,
) -> BaselineTrialResult | None:
    """Analyze a trained network's arrangement matroid.

    Returns None if matroid construction fails due to numerical issues.
    """
    arr = net.hyperplane_arrangement(layer_idx=0)
    hidden_dim = net.hidden_dims[0]

    try:
        aff_mat = arr.affine_matroid()
    except ValueError:
        return None

    aff_is_pos = is_positroid(aff_mat)
    aff_necklace = None
    aff_perm = None
    if aff_is_pos:
        aff_necklace = grassmann_necklace(aff_mat)
        aff_perm = decorated_permutation(aff_necklace, aff_mat.size)

    # Compute non-bases
    rank = aff_mat.rank
    all_subsets = {frozenset(s) for s in combinations(range(hidden_dim), rank)}
    non_bases_frozen = all_subsets - aff_mat.bases
    non_bases = sorted([tuple(sorted(s)) for s in non_bases_frozen])

    # Check if any non-basis is NOT a cyclic interval
    has_non_interval = False
    for nb in non_bases_frozen:
        if not is_cyclic_interval(nb, hidden_dim):
            has_non_interval = True
            break

    w1 = net.layers[0].weight
    weight_is_tp = bool(is_totally_positive(w1))

    return BaselineTrialResult(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        param_mode=param_mode,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        is_weight_tp=weight_is_tp,
        affine_matroid_rank=rank,
        affine_matroid_num_bases=len(aff_mat.bases),
        affine_matroid_is_uniform=aff_mat.is_uniform(),
        affine_matroid_is_positroid=aff_is_pos,
        non_bases=non_bases,
        has_non_interval_nonbases=has_non_interval,
        affine_grassmann_necklace=aff_necklace,
        affine_decorated_permutation=aff_perm,
    )


def run_single_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    param_mode: str,
) -> BaselineTrialResult | None:
    """Train a network and analyze its matroid."""
    net, history = train(x, y, config)
    final_loss = history.losses[-1] if history.losses else float("inf")
    final_acc = history.accuracies[-1] if history.accuracies else 0.0
    return analyze_trial(net, dataset_name, param_mode, final_loss, final_acc)


def run_baseline_experiment(
    dataset_name: str,
    hidden_dim: int,
    num_trials: int,
    param_modes: list[str],
    n_samples: int = 200,
    epochs: int = 200,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> BaselineExperimentResult:
    """Run the non-TP baseline experiment for one configuration."""
    result = BaselineExperimentResult(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        num_trials=num_trials,
    )
    for mode in param_modes:
        result.trials_by_mode[mode] = []

    dataset_fn = DATASETS[dataset_name]

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        data_rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=data_rng)

        for mode in param_modes:
            config = TrainConfig(
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                epochs=epochs,
                param_mode=mode,
                seed=trial_seed + _MODE_SEED_OFFSET.get(mode, 0),
            )
            trial_result = run_single_trial(dataset_name, x, y, config, mode)
            if trial_result is not None:
                result.trials_by_mode[mode].append(trial_result)

    return result


def print_results(results: list[BaselineExperimentResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 100)
    print("NON-TP BASELINE EXPERIMENT RESULTS")
    print("=" * 100)
    print()
    print(
        f"{'Dataset':>10}  {'H':>3}  {'Mode':>20}  "
        f"{'Pos%':>6}  {'Unif%':>6}  {'NonInt%':>8}  {'Acc%':>6}"
    )
    print("-" * 100)

    for r in results:
        for mode in sorted(r.trials_by_mode.keys()):
            trials = r.trials_by_mode[mode]
            if not trials:
                continue
            print(
                f"{r.dataset_name:>10}  {r.hidden_dim:>3}  {mode:>20}  "
                f"{r.positroid_rate(mode):>6.1%}  "
                f"{r.uniform_rate(mode):>6.1%}  "
                f"{r.non_interval_rate(mode):>8.1%}  "
                f"{r.mean_accuracy(mode):>6.1%}"
            )

    print()
    print("Legend:")
    print("  H        = hidden dimension")
    print("  Mode     = parameter mode (tp_exponential, sinusoidal, etc.)")
    print("  Pos%     = fraction of trials producing positroid matroids")
    print("  Unif%    = fraction of trials producing uniform (trivial) matroids")
    print("  NonInt%  = fraction with at least one non-cyclic-interval non-basis")
    print("  Acc%     = mean training accuracy")
    print()


def print_detailed_results(results: list[BaselineExperimentResult]) -> None:
    """Print per-trial detail for non-uniform or non-positroid cases."""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS (non-uniform and/or non-positroid trials)")
    print("=" * 100)

    any_printed = False
    for r in results:
        for mode, trials in r.trials_by_mode.items():
            for i, t in enumerate(trials):
                if not t.affine_matroid_is_uniform or not t.affine_matroid_is_positroid:
                    any_printed = True
                    status = "NON-POSITROID" if not t.affine_matroid_is_positroid else "non-uniform"
                    print(
                        f"\n--- [{status}] {r.dataset_name} H={t.hidden_dim} {mode} trial {i} ---"
                    )
                    print(f"  Accuracy: {t.final_accuracy:.1%}")
                    print(f"  Loss: {t.final_loss:.4f}")
                    print(f"  Weight TP: {t.is_weight_tp}")
                    print(
                        f"  Affine matroid: rank={t.affine_matroid_rank}, "
                        f"|bases|={t.affine_matroid_num_bases}, "
                        f"uniform={t.affine_matroid_is_uniform}, "
                        f"positroid={t.affine_matroid_is_positroid}"
                    )
                    print(f"  Non-bases: {t.non_bases}")
                    print(f"  Has non-interval non-bases: {t.has_non_interval_nonbases}")
                    if t.affine_decorated_permutation is not None:
                        print(f"  Decorated perm: {t.affine_decorated_permutation}")

    if not any_printed:
        all_uniform = all(
            t.affine_matroid_is_uniform
            for r in results
            for trials in r.trials_by_mode.values()
            for t in trials
        )
        if all_uniform:
            print("\n  All affine matroids were uniform (trivially positroids).")
        else:
            print("\n  All non-uniform affine matroids were positroids!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-TP Baseline Experiment: TP vs non-TP kernel vs unconstrained",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["moons", "circles"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[6, 8, 10],
    )
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--param-modes",
        nargs="+",
        default=["tp_exponential", "sinusoidal", "quadratic_distance", "unconstrained"],
    )
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    all_results: list[BaselineExperimentResult] = []

    for dataset_name in args.datasets:
        for h in args.hidden_dims:
            print(f"Running {dataset_name}, H={h}...", flush=True)
            result = run_baseline_experiment(
                dataset_name=dataset_name,
                hidden_dim=h,
                num_trials=args.num_trials,
                param_modes=args.param_modes,
                n_samples=args.n_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
            )
            for mode in args.param_modes:
                pos = result.positroid_rate(mode)
                unif = result.uniform_rate(mode)
                print(f"  {mode}: positroid={pos:.1%}, uniform={unif:.1%}")
            all_results.append(result)

    print_results(all_results)

    if args.detailed:
        print_detailed_results(all_results)


if __name__ == "__main__":
    main()
