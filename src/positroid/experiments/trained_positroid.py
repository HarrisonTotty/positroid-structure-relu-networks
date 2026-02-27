"""Trained Network Positroid Experiment.

Tests the Activation Positroid Conjecture on networks TRAINED on real data,
where weight-bias correlations create non-uniform (interesting) matroids.

Compares:
1. Unconstrained training (arbitrary weights)
2. TP-constrained training (weights constrained to be totally positive)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from positroid.datasets import DATASETS
from positroid.linalg.totally_positive import is_totally_positive
from positroid.matroid.positroid import (
    decorated_permutation,
    grassmann_necklace,
    is_positroid,
)
from positroid.network.relu_network import ReluNetwork
from positroid.network.train import TrainConfig, TrainHistory, train


@dataclass
class TrainedTrialResult:
    """Result of one trained-network positroid trial."""

    dataset_name: str
    hidden_dim: int
    tp_constrained: bool

    final_loss: float
    final_accuracy: float

    is_weight_tp: bool

    affine_matroid_rank: int
    affine_matroid_num_bases: int
    affine_matroid_is_uniform: bool
    affine_matroid_is_positroid: bool
    affine_matroid_num_circuits: int
    affine_grassmann_necklace: tuple[frozenset[int], ...] | None = None
    affine_decorated_permutation: list[int | None] | None = None

    linear_matroid_rank: int = 0
    linear_matroid_is_uniform: bool = True
    linear_matroid_is_positroid: bool = True

    matroid_evolution: list[tuple[int, bool, bool]] | None = None


@dataclass
class TrainedExperimentResult:
    """Aggregated results for one configuration."""

    dataset_name: str
    hidden_dim: int
    num_trials: int
    tp_trials: list[TrainedTrialResult] = field(default_factory=list)
    unconstrained_trials: list[TrainedTrialResult] = field(default_factory=list)

    @property
    def tp_positroid_rate(self) -> float:
        if not self.tp_trials:
            return 0.0
        return sum(t.affine_matroid_is_positroid for t in self.tp_trials) / len(self.tp_trials)

    @property
    def unconstrained_positroid_rate(self) -> float:
        if not self.unconstrained_trials:
            return 0.0
        return sum(t.affine_matroid_is_positroid for t in self.unconstrained_trials) / len(
            self.unconstrained_trials
        )

    @property
    def tp_uniform_rate(self) -> float:
        if not self.tp_trials:
            return 0.0
        return sum(t.affine_matroid_is_uniform for t in self.tp_trials) / len(self.tp_trials)

    @property
    def unconstrained_uniform_rate(self) -> float:
        if not self.unconstrained_trials:
            return 0.0
        return sum(t.affine_matroid_is_uniform for t in self.unconstrained_trials) / len(
            self.unconstrained_trials
        )


def analyze_network(
    net: ReluNetwork,
    dataset_name: str,
    tp_constrained: bool,
    final_loss: float,
    final_accuracy: float,
    history: TrainHistory | None = None,
) -> TrainedTrialResult:
    """Analyze a trained network's arrangement matroid."""
    arr = net.hyperplane_arrangement(layer_idx=0)

    lin_mat = arr.linear_matroid()
    aff_mat = arr.affine_matroid()

    aff_is_pos = is_positroid(aff_mat)
    aff_necklace = None
    aff_perm = None
    if aff_is_pos:
        aff_necklace = grassmann_necklace(aff_mat)
        aff_perm = decorated_permutation(aff_necklace, aff_mat.size)

    circuits = aff_mat.circuits()
    w1 = net.layers[0].weight
    weight_is_tp = bool(is_totally_positive(w1))

    # Track matroid evolution from training snapshots
    evolution = None
    if history and history.snapshots:
        evolution = []
        for epoch in sorted(history.snapshots):
            snap_net = history.snapshots[epoch]
            snap_arr = snap_net.hyperplane_arrangement(layer_idx=0)
            snap_aff = snap_arr.affine_matroid()
            evolution.append(
                (epoch, snap_aff.is_uniform(), is_positroid(snap_aff)),
            )

    return TrainedTrialResult(
        dataset_name=dataset_name,
        hidden_dim=net.hidden_dims[0],
        tp_constrained=tp_constrained,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        is_weight_tp=weight_is_tp,
        affine_matroid_rank=aff_mat.rank,
        affine_matroid_num_bases=len(aff_mat.bases),
        affine_matroid_is_uniform=aff_mat.is_uniform(),
        affine_matroid_is_positroid=aff_is_pos,
        affine_matroid_num_circuits=len(circuits),
        affine_grassmann_necklace=aff_necklace,
        affine_decorated_permutation=aff_perm,
        linear_matroid_rank=lin_mat.rank,
        linear_matroid_is_uniform=lin_mat.is_uniform(),
        linear_matroid_is_positroid=is_positroid(lin_mat),
        matroid_evolution=evolution,
    )


def run_single_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    snapshot_epochs: list[int] | None = None,
) -> TrainedTrialResult:
    """Train a network and analyze its matroid."""
    net, history = train(x, y, config, snapshot_epochs=snapshot_epochs)

    final_loss = history.losses[-1] if history.losses else float("inf")
    final_acc = history.accuracies[-1] if history.accuracies else 0.0

    return analyze_network(
        net=net,
        dataset_name=dataset_name,
        tp_constrained=config.tp_constrained,
        final_loss=final_loss,
        final_accuracy=final_acc,
        history=history,
    )


def run_experiment(
    dataset_name: str,
    hidden_dim: int,
    num_trials: int,
    n_samples: int = 200,
    epochs: int = 200,
    learning_rate: float = 0.01,
    seed: int = 42,
    track_evolution: bool = False,
    tp_kernel: str = "exponential",
) -> TrainedExperimentResult:
    """Run the trained positroid experiment for one configuration."""
    result = TrainedExperimentResult(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        num_trials=num_trials,
    )

    snapshot_epochs = None
    if track_evolution:
        snapshot_epochs = list(range(0, epochs, max(1, epochs // 10)))

    dataset_fn = DATASETS[dataset_name]

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        data_rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=data_rng)

        # TP-constrained trial
        tp_config = TrainConfig(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            tp_constrained=True,
            tp_kernel=tp_kernel,
            seed=trial_seed + 1000,
        )
        tp_trial = run_single_trial(
            dataset_name,
            x,
            y,
            tp_config,
            snapshot_epochs,
        )
        result.tp_trials.append(tp_trial)

        # Unconstrained trial
        uc_config = TrainConfig(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            tp_constrained=False,
            seed=trial_seed + 2000,
        )
        uc_trial = run_single_trial(
            dataset_name,
            x,
            y,
            uc_config,
            snapshot_epochs,
        )
        result.unconstrained_trials.append(uc_trial)

    return result


def print_results(results: list[TrainedExperimentResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 95)
    print("TRAINED NETWORK POSITROID TEST RESULTS")
    print("=" * 95)
    print()
    print(
        f"{'Dataset':>10}  {'H':>3}  "
        f"{'TP Pos':>7}  {'TP Unif':>8}  {'TP Acc':>7}  "
        f"{'UC Pos':>7}  {'UC Unif':>8}  {'UC Acc':>7}"
    )
    print("-" * 95)

    for r in results:
        tp_acc = np.mean([t.final_accuracy for t in r.tp_trials]) if r.tp_trials else 0.0
        uc_acc = (
            np.mean([t.final_accuracy for t in r.unconstrained_trials])
            if r.unconstrained_trials
            else 0.0
        )

        print(
            f"{r.dataset_name:>10}  {r.hidden_dim:>3}  "
            f"{r.tp_positroid_rate:>7.1%}  "
            f"{r.tp_uniform_rate:>8.1%}  "
            f"{tp_acc:>7.1%}  "
            f"{r.unconstrained_positroid_rate:>7.1%}  "
            f"{r.unconstrained_uniform_rate:>8.1%}  "
            f"{uc_acc:>7.1%}"
        )

    print()
    print("Legend:")
    print("  H        = hidden dimension")
    print("  TP Pos   = positroid rate for TP-constrained training")
    print("  TP Unif  = uniform matroid rate for TP-constrained")
    print("  TP Acc   = mean training accuracy for TP-constrained")
    print("  UC Pos   = positroid rate for unconstrained training")
    print("  UC Unif  = uniform matroid rate for unconstrained")
    print("  UC Acc   = mean training accuracy for unconstrained")
    print()


def print_detailed_results(results: list[TrainedExperimentResult]) -> None:
    """Print per-trial details for interesting cases."""
    print("\n" + "=" * 95)
    print("DETAILED RESULTS")
    print("=" * 95)

    for r in results:
        for label, trials in [("TP", r.tp_trials), ("UC", r.unconstrained_trials)]:
            for i, t in enumerate(trials):
                if not t.affine_matroid_is_uniform or not t.affine_matroid_is_positroid:
                    print(f"\n--- {r.dataset_name} H={t.hidden_dim} {label} trial {i} ---")
                    print(f"  Accuracy: {t.final_accuracy:.1%}")
                    print(f"  Loss: {t.final_loss:.4f}")
                    print(f"  Weight TP: {t.is_weight_tp}")
                    print(
                        f"  Affine matroid: rank={t.affine_matroid_rank}, "
                        f"|bases|={t.affine_matroid_num_bases}, "
                        f"uniform={t.affine_matroid_is_uniform}, "
                        f"positroid={t.affine_matroid_is_positroid}"
                    )
                    print(f"  Circuits: {t.affine_matroid_num_circuits}")
                    if t.affine_decorated_permutation is not None:
                        print(f"  Decorated perm: {t.affine_decorated_permutation}")
                    if t.matroid_evolution:
                        print("  Evolution:")
                        for epoch, unif, pos in t.matroid_evolution:
                            print(f"    epoch {epoch:>4}: uniform={unif}, positroid={pos}")

    # Check if all were trivial
    all_uniform = all(
        t.affine_matroid_is_uniform for r in results for t in r.tp_trials + r.unconstrained_trials
    )
    if all_uniform:
        print("\n  All affine matroids were uniform (trivially positroids).")

    all_positroid = all(
        t.affine_matroid_is_positroid for r in results for t in r.tp_trials + r.unconstrained_trials
    )
    if all_positroid and not all_uniform:
        print("\n  All affine matroids were positroids!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trained Network Positroid Test",
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
    parser.add_argument("--track-evolution", action="store_true")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument(
        "--tp-kernel",
        default="exponential",
        choices=["exponential", "cauchy"],
    )
    args = parser.parse_args()

    all_results: list[TrainedExperimentResult] = []

    for dataset_name in args.datasets:
        for h in args.hidden_dims:
            print(f"Running {dataset_name}, H={h}...", flush=True)
            result = run_experiment(
                dataset_name=dataset_name,
                hidden_dim=h,
                num_trials=args.num_trials,
                n_samples=args.n_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                track_evolution=args.track_evolution,
                tp_kernel=args.tp_kernel,
            )
            tp_pos = result.tp_positroid_rate
            uc_pos = result.unconstrained_positroid_rate
            print(f"  TP positroid: {tp_pos:.1%}, UC positroid: {uc_pos:.1%}")
            all_results.append(result)

    print_results(all_results)

    if args.detailed:
        print_detailed_results(all_results)


if __name__ == "__main__":
    main()
