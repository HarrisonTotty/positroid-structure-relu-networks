"""MNIST Experiment: Positroid network on multiclass digit classification.

Tests positroid cell networks on 10-class sklearn digits (8x8) and full MNIST
(28x28) with PCA reduction, comparing against parameter-matched ReLU baselines.

Usage:
    # Quick test on sklearn digits
    python -m positroid.experiments.mnist_experiment --datasets digits_10class_pca10

    # Full run
    python -m positroid.experiments.mnist_experiment

    # Custom settings
    python -m positroid.experiments.mnist_experiment --k 2 3 --epochs 300 --num-trials 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from time import time

import numpy as np

from positroid.datasets import DATASETS
from positroid.network.positroid_network import (
    PositroidNetwork,
    PositroidTrainConfig,
    train_positroid,
)
from positroid.network.train import TrainConfig, train_multiclass


@dataclass
class TrialResult:
    """Result of one trial."""

    dataset: str
    mode: str  # 'positroid_fixed', 'positroid_learnable', 'relu'
    k: int
    n: int
    num_classes: int
    num_params: int
    final_loss: float
    final_accuracy: float
    elapsed: float  # seconds


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    datasets: list[str] = field(
        default_factory=lambda: ["digits_10class_pca10", "digits_10class_pca20"]
    )
    k_values: list[int] = field(default_factory=lambda: [2, 3])
    num_trials: int = 3
    n_samples: int = 1000
    epochs: int = 200
    learning_rate: float = 0.01
    seed: int = 42
    modes: list[str] = field(
        default_factory=lambda: ["positroid_fixed", "positroid_learnable", "relu"]
    )


# Recommended (k, n) for different input dimensions
_KN_TABLE: dict[tuple[int, int], int] = {
    # (k, d) -> n
    (2, 10): 12,
    (2, 20): 16,
    (2, 50): 20,
    (3, 10): 8,
    (3, 20): 10,
    (3, 50): 12,
}


def _select_n(k: int, d: int) -> int:
    """Select n for given (k, d) from table or heuristic."""
    key = (k, d)
    if key in _KN_TABLE:
        return _KN_TABLE[key]
    # Heuristic: n ~ d + k for k=2, n ~ d/2 + k for k=3
    if k == 2:
        return min(d + k, 30)
    return min(d // 2 + k, 20)


def run_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    k: int,
    n: int,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    target_params: int | None = None,
) -> TrialResult:
    """Run a single trial."""
    d = x.shape[1]
    t0 = time()

    if mode.startswith("positroid"):
        encoding = "learnable" if mode == "positroid_learnable" else "fixed"
        config = PositroidTrainConfig(
            k=k,
            n=n,
            encoding=encoding,
            readout="det",
            num_classes=num_classes,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=64,
            seed=seed,
        )
        net, history = train_positroid(x, y, config)

        return TrialResult(
            dataset=dataset_name,
            mode=mode,
            k=k,
            n=n,
            num_classes=num_classes,
            num_params=net.num_params,
            final_loss=history.losses[-1] if history.losses else float("inf"),
            final_accuracy=history.accuracies[-1] if history.accuracies else 0.0,
            elapsed=time() - t0,
        )

    else:
        # ReLU baseline: match parameter count
        assert target_params is not None
        h = max(1, round((target_params - num_classes) / (d + 1 + num_classes)))

        relu_config = TrainConfig(
            hidden_dim=h,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=64,
            param_mode="unconstrained",
            seed=seed,
        )
        _, relu_history = train_multiclass(x, y, relu_config, num_classes=num_classes)
        actual_params = h * (d + 1 + num_classes) + num_classes

        return TrialResult(
            dataset=dataset_name,
            mode="relu",
            k=k,
            n=n,
            num_classes=num_classes,
            num_params=actual_params,
            final_loss=relu_history.losses[-1] if relu_history.losses else float("inf"),
            final_accuracy=(relu_history.accuracies[-1] if relu_history.accuracies else 0.0),
            elapsed=time() - t0,
        )


def run_experiment(exp_config: ExperimentConfig) -> list[TrialResult]:
    """Run the full experiment."""
    results: list[TrialResult] = []

    for dataset_name in exp_config.datasets:
        if dataset_name not in DATASETS:
            print(f"WARNING: {dataset_name} not found in DATASETS registry, skipping.")
            continue

        dataset_fn = DATASETS[dataset_name]

        for k in exp_config.k_values:
            # Load a sample to get dimensions
            rng = np.random.default_rng(exp_config.seed)
            x_sample, y_sample = dataset_fn(n_samples=10, rng=rng)
            d = x_sample.shape[1]
            num_classes = len(np.unique(y_sample))
            # For registry datasets, we know it's 10 classes
            if "10class" in dataset_name:
                num_classes = 10

            n = _select_n(k, d)
            print(f"\n{'=' * 60}\n{dataset_name}  k={k} n={n}  d={d}  C={num_classes}\n{'=' * 60}")

            # Get param count from positroid_fixed to match ReLU
            temp_net = PositroidNetwork(
                k=k, n=n, input_dim=d, encoding="fixed", num_classes=num_classes
            )
            target_params_fixed = temp_net.num_params

            for trial_idx in range(exp_config.num_trials):
                trial_seed = exp_config.seed + trial_idx
                data_rng = np.random.default_rng(trial_seed)
                x, y = dataset_fn(n_samples=exp_config.n_samples, rng=data_rng)

                for mode in exp_config.modes:
                    trial = run_trial(
                        dataset_name=dataset_name,
                        x=x,
                        y=y,
                        mode=mode,
                        k=k,
                        n=n,
                        num_classes=num_classes,
                        epochs=exp_config.epochs,
                        learning_rate=exp_config.learning_rate,
                        seed=trial_seed + hash(mode) % 10000,
                        target_params=target_params_fixed,
                    )
                    results.append(trial)
                    print(
                        f"  trial {trial_idx}  {mode:>22}  "
                        f"params={trial.num_params:>5}  "
                        f"acc={trial.final_accuracy:>5.1%}  "
                        f"loss={trial.final_loss:.4f}  "
                        f"({trial.elapsed:.1f}s)"
                    )

    return results


def print_summary(results: list[TrialResult]) -> None:
    """Print aggregated results table."""
    print("\n" + "=" * 95)
    print("MNIST EXPERIMENT RESULTS (averaged over trials)")
    print("=" * 95)
    print(
        f"{'Dataset':>25}  {'Mode':>22}  {'k':>2}  {'n':>2}  "
        f"{'Params':>6}  {'Acc%':>6}  {'Loss':>8}  {'Time':>6}"
    )
    print("-" * 95)

    # Group by (dataset, k, n, mode)
    groups: dict[tuple[str, int, int, str], list[TrialResult]] = {}
    for r in results:
        key = (r.dataset, r.k, r.n, r.mode)
        groups.setdefault(key, []).append(r)

    for key, trials in sorted(groups.items()):
        dataset, k, n, mode = key
        mean_acc = np.mean([t.final_accuracy for t in trials])
        std_acc = np.std([t.final_accuracy for t in trials])
        mean_loss = np.mean([t.final_loss for t in trials])
        mean_time = np.mean([t.elapsed for t in trials])
        params = trials[0].num_params

        print(
            f"{dataset:>25}  {mode:>22}  {k:>2}  {n:>2}  "
            f"{params:>6}  {mean_acc:>5.1%}  {mean_loss:>8.4f}  {mean_time:>5.1f}s"
        )
        if std_acc > 0.001:
            print(f"{'':>25}  {'':>22}  {'':>2}  {'':>2}  {'':>6}  +/-{std_acc:.1%}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST Positroid Experiment")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["digits_10class_pca10", "digits_10class_pca20"],
    )
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["positroid_fixed", "positroid_learnable", "relu"],
        choices=["positroid_fixed", "positroid_learnable", "relu"],
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(
        datasets=args.datasets,
        k_values=args.k,
        num_trials=args.num_trials,
        n_samples=args.n_samples,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        modes=args.modes,
    )

    results = run_experiment(exp_config)
    print_summary(results)


if __name__ == "__main__":
    main()
