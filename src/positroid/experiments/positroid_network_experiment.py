"""Positroid Network Experiment.

Trains a positroid cell network on binary classification datasets and
compares to an equivalently-parameterized ReLU baseline.

Tests:
1. Fixed vs learnable encoding (is the plabic graph or the encoding doing the work?)
2. k=2 (linear) vs k=3 (quadratic) readout
3. Comparison to ReLU baseline with matched parameter count
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from positroid.datasets import DATASETS
from positroid.network.positroid_network import (
    PositroidNetwork,
    PositroidTrainConfig,
    train_positroid,
)
from positroid.network.train import TrainConfig, train
from positroid.positroid_cell.boundary_map import plucker_coordinates


@dataclass
class TrialResult:
    """Result of one trial."""

    mode: str  # 'positroid_fixed', 'positroid_learnable', 'relu'
    k: int
    n: int
    num_params: int
    final_loss: float
    final_accuracy: float
    readout: str = "det"
    all_plucker_positive: bool = True


@dataclass
class ExperimentResult:
    """Aggregated results for one configuration."""

    dataset_name: str
    num_trials: int
    trials: list[TrialResult] = field(default_factory=list)


def _relu_hidden_dim_for_params(target_params: int, input_dim: int) -> int:
    """Find hidden dim H such that a ReLU net has ~target_params parameters.

    ReLU net params = H*d + H + H + 1 = H*(d+2) + 1.
    """
    # H*(d+2) + 1 ~= target_params => H = (target_params - 1) / (d + 2)
    h = max(1, round((target_params - 1) / (input_dim + 2)))
    return h


def run_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    k: int,
    n: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    readout: str = "det",
) -> TrialResult:
    """Run a single trial."""
    input_dim = x.shape[1]

    if mode.startswith("positroid"):
        encoding = "learnable" if mode == "positroid_learnable" else "fixed"
        config = PositroidTrainConfig(
            k=k,
            n=n,
            encoding=encoding,
            readout=readout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=32,
            seed=seed,
        )
        net, history = train_positroid(x, y, config)

        # Check Plucker positivity
        weights = np.exp(net.face_weights_raw)
        from positroid.positroid_cell.boundary_map import boundary_measurement_matrix

        bnd = boundary_measurement_matrix(weights, k, n)
        pluckers = plucker_coordinates(bnd)
        all_pos = all(v > 0 for v in pluckers.values())

        return TrialResult(
            mode=mode,
            k=k,
            n=n,
            num_params=net.num_params,
            final_loss=history.losses[-1] if history.losses else float("inf"),
            final_accuracy=history.accuracies[-1] if history.accuracies else 0.0,
            readout=readout,
            all_plucker_positive=all_pos,
        )

    else:
        # ReLU baseline: match parameter count to positroid_fixed
        pos_net_temp = PositroidNetwork(k=k, n=n, input_dim=input_dim, encoding="fixed")
        target_params = pos_net_temp.num_params
        h = _relu_hidden_dim_for_params(target_params, input_dim)

        relu_config = TrainConfig(
            hidden_dim=h,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=32,
            param_mode="unconstrained",
            seed=seed,
        )
        _, relu_history = train(x, y, relu_config)
        actual_params = h * (input_dim + 2) + 1

        return TrialResult(
            mode="relu",
            k=k,
            n=n,
            num_params=actual_params,
            final_loss=relu_history.losses[-1] if relu_history.losses else float("inf"),
            final_accuracy=(relu_history.accuracies[-1] if relu_history.accuracies else 0.0),
        )


def run_experiment(
    dataset_name: str,
    k: int,
    n: int,
    num_trials: int,
    n_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    modes: list[str],
    readout: str = "det",
) -> ExperimentResult:
    """Run experiment for one (dataset, k, n) configuration."""
    result = ExperimentResult(dataset_name=dataset_name, num_trials=num_trials)
    dataset_fn = DATASETS[dataset_name]

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        data_rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=data_rng)

        for mode in modes:
            trial = run_trial(
                dataset_name=dataset_name,
                x=x,
                y=y,
                mode=mode,
                k=k,
                n=n,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=trial_seed + hash(mode) % 10000,
                readout=readout,
            )
            result.trials.append(trial)

    return result


def print_results(results: list[ExperimentResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 90)
    print("POSITROID NETWORK EXPERIMENT RESULTS")
    print("=" * 90)
    print()
    print(
        f"{'Dataset':>10}  {'Mode':>22}  {'k':>2}  {'n':>2}  "
        f"{'Params':>6}  {'Acc%':>6}  {'Loss':>8}  {'Pluck+':>6}"
    )
    print("-" * 90)

    for r in results:
        # Group trials by mode
        by_mode: dict[str, list[TrialResult]] = {}
        for t in r.trials:
            by_mode.setdefault(t.mode, []).append(t)

        for mode, trials in sorted(by_mode.items()):
            mean_acc = np.mean([t.final_accuracy for t in trials])
            mean_loss = np.mean([t.final_loss for t in trials])
            params = trials[0].num_params
            k = trials[0].k
            n = trials[0].n
            plucker_rate = np.mean([t.all_plucker_positive for t in trials])

            plucker_str = f"{plucker_rate:.0%}" if mode != "relu" else "n/a"
            print(
                f"{r.dataset_name:>10}  {mode:>22}  {k:>2}  {n:>2}  "
                f"{params:>6}  {mean_acc:>5.1%}  {mean_loss:>8.4f}  {plucker_str:>6}"
            )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Positroid Network Experiment")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["moons"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["positroid_fixed", "positroid_learnable", "relu"],
        choices=["positroid_fixed", "positroid_learnable", "relu"],
    )
    parser.add_argument(
        "--readout",
        default="det",
        choices=["det", "plucker_ratio", "canonical_residue"],
    )
    args = parser.parse_args()

    all_results: list[ExperimentResult] = []

    for dataset_name in args.datasets:
        for k in args.k:
            print(f"Running {dataset_name}, k={k}, n={args.n}...", flush=True)
            result = run_experiment(
                dataset_name=dataset_name,
                k=k,
                n=args.n,
                num_trials=args.num_trials,
                n_samples=args.n_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                modes=args.modes,
                readout=args.readout,
            )
            all_results.append(result)

            # Quick summary
            by_mode: dict[str, list[TrialResult]] = {}
            for t in result.trials:
                by_mode.setdefault(t.mode, []).append(t)
            for mode, trials in sorted(by_mode.items()):
                mean_acc = np.mean([t.final_accuracy for t in trials])
                print(f"  {mode}: {mean_acc:.1%}")

    print_results(all_results)


if __name__ == "__main__":
    main()
