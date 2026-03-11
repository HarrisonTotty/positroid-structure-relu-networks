"""Matroid Evolution Experiment.

Instruments training to track matroid evolution epoch-by-epoch, correlating
matroid metrics with training metrics. Key question: do matroids *transition*
during training? E.g.:
- uniform → contiguous non-bases → positroid (TP)
- uniform → gapped non-bases → non-positroid (negated bidiagonal)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from positroid.datasets import DATASETS
from positroid.matroid.positroid import (
    is_cyclic_interval,
    is_positroid,
)
from positroid.network.relu_network import ReluNetwork
from positroid.network.train import TrainConfig, TrainHistory, train

_MODE_SEED_OFFSET: dict[str, int] = {
    "tp_exponential": 0,
    "tp_cauchy": 1000,
    "negated_bidiagonal": 6000,
}


@dataclass
class EpochSnapshot:
    """Per-epoch matroid metrics."""

    epoch: int
    loss: float
    accuracy: float

    is_uniform: bool
    is_positroid: bool
    num_bases: int
    num_non_bases: int

    support: tuple[int, ...]
    support_size: int
    support_is_interval: bool
    support_rank_deficiency: int

    all_nonbases_are_intervals: bool
    contiguous_support_condition: bool


@dataclass
class EvolutionTrialResult:
    """One trial's full timeline."""

    dataset_name: str
    hidden_dim: int
    param_mode: str
    seed: int

    final_loss: float
    final_accuracy: float
    final_is_positroid: bool

    snapshots: list[EpochSnapshot]

    first_nonuniform_epoch: int | None = None
    first_nonpositroid_epoch: int | None = None
    first_noninterval_epoch: int | None = None


@dataclass
class EvolutionExperimentResult:
    """Aggregated results across parameter modes."""

    trials_by_mode: dict[str, list[EvolutionTrialResult]] = field(default_factory=dict)

    def always_positroid_rate(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return sum(all(s.is_positroid for s in t.snapshots) for t in trials) / len(trials)

    def ever_nonuniform_rate(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return sum(any(not s.is_uniform for s in t.snapshots) for t in trials) / len(trials)

    def ever_noninterval_rate(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return sum(any(not s.support_is_interval for s in t.snapshots) for t in trials) / len(
            trials
        )

    def mean_accuracy(self, mode: str) -> float:
        trials = self.trials_by_mode.get(mode, [])
        if not trials:
            return 0.0
        return float(np.mean([t.final_accuracy for t in trials]))


def analyze_snapshot(
    net: ReluNetwork,
    epoch: int,
    loss: float,
    accuracy: float,
) -> EpochSnapshot | None:
    """Analyze a network snapshot's matroid at a given epoch.

    Returns None if matroid construction fails due to numerical issues.
    """
    arr = net.hyperplane_arrangement(layer_idx=0)

    try:
        aff_mat = arr.affine_matroid()
    except ValueError:
        return None

    n = aff_mat.size
    k = aff_mat.rank
    all_k_subsets = frozenset(frozenset(s) for s in combinations(range(n), k))
    non_bases = all_k_subsets - aff_mat.bases

    # Compute support once from local non_bases, derive everything from it
    support = frozenset().union(*non_bases) if non_bases else frozenset()
    support_is_interval = True if not support else is_cyclic_interval(support, n)
    rank_def = 0 if not support else k - aff_mat.rank_of(support)

    all_intervals = all(is_cyclic_interval(nb, n) for nb in non_bases)
    cst_condition = support_is_interval and rank_def > 0

    aff_is_pos = is_positroid(aff_mat)

    # CST theorem invariant: contiguous rank-deficient support implies positroid.
    # Violation here would indicate a bug in either is_positroid or the CST check.
    assert not cst_condition or aff_is_pos, (
        f"CST invariant violated at epoch {epoch}: support={sorted(support)}, "
        f"rank_def={rank_def}, but is_positroid=False"
    )

    return EpochSnapshot(
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        is_uniform=aff_mat.is_uniform(),
        is_positroid=aff_is_pos,
        num_bases=len(aff_mat.bases),
        num_non_bases=len(non_bases),
        support=tuple(sorted(support)),
        support_size=len(support),
        support_is_interval=support_is_interval,
        support_rank_deficiency=rank_def,
        all_nonbases_are_intervals=all_intervals,
        contiguous_support_condition=cst_condition,
    )


def analyze_snapshot_evolution(
    history: TrainHistory,
    dataset_name: str,
    hidden_dim: int,
    param_mode: str,
    seed: int,
) -> EvolutionTrialResult:
    """Analyze all snapshots from a training history."""
    snapshots: list[EpochSnapshot] = []

    for epoch in sorted(history.snapshots.keys()):
        net = history.snapshots[epoch]
        loss = history.losses[epoch] if epoch < len(history.losses) else float("inf")
        acc = history.accuracies[epoch] if epoch < len(history.accuracies) else 0.0
        snap = analyze_snapshot(net, epoch, loss, acc)
        if snap is not None:
            snapshots.append(snap)

    # Compute transition epochs
    first_nonuniform = None
    first_nonpositroid = None
    first_noninterval = None
    for s in snapshots:
        if not s.is_uniform and first_nonuniform is None:
            first_nonuniform = s.epoch
        if not s.is_positroid and first_nonpositroid is None:
            first_nonpositroid = s.epoch
        if not s.support_is_interval and first_noninterval is None:
            first_noninterval = s.epoch

    final_loss = history.losses[-1] if history.losses else float("inf")
    final_acc = history.accuracies[-1] if history.accuracies else 0.0
    final_pos = snapshots[-1].is_positroid if snapshots else True

    return EvolutionTrialResult(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        param_mode=param_mode,
        seed=seed,
        final_loss=final_loss,
        final_accuracy=final_acc,
        final_is_positroid=final_pos,
        snapshots=snapshots,
        first_nonuniform_epoch=first_nonuniform,
        first_nonpositroid_epoch=first_nonpositroid,
        first_noninterval_epoch=first_noninterval,
    )


def run_single_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    param_mode: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    snapshot_interval: int,
    seed: int,
) -> EvolutionTrialResult:
    """Train a network and analyze matroid evolution across epochs."""
    snapshot_epochs = list(range(0, epochs, snapshot_interval))
    if (epochs - 1) not in snapshot_epochs:
        snapshot_epochs.append(epochs - 1)

    config = TrainConfig(
        hidden_dim=hidden_dim,
        learning_rate=lr,
        epochs=epochs,
        param_mode=param_mode,
        seed=seed,
    )
    _, history = train(x, y, config, snapshot_epochs=snapshot_epochs)

    return analyze_snapshot_evolution(
        history,
        dataset_name,
        hidden_dim,
        param_mode,
        seed,
    )


def run_evolution_experiment(
    dataset_name: str,
    hidden_dim: int,
    num_trials: int,
    param_modes: list[str],
    n_samples: int = 200,
    epochs: int = 200,
    learning_rate: float = 0.01,
    snapshot_interval: int = 10,
    seed: int = 42,
) -> EvolutionExperimentResult:
    """Run the matroid evolution experiment for one configuration."""
    result = EvolutionExperimentResult()
    for mode in param_modes:
        result.trials_by_mode[mode] = []

    dataset_fn = DATASETS[dataset_name]

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        data_rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=data_rng)

        for mode in param_modes:
            mode_seed = trial_seed + _MODE_SEED_OFFSET.get(mode, 0)
            trial_result = run_single_trial(
                dataset_name,
                x,
                y,
                mode,
                hidden_dim,
                epochs,
                learning_rate,
                snapshot_interval,
                mode_seed,
            )
            result.trials_by_mode[mode].append(trial_result)

    return result


def print_results(results: list[EvolutionExperimentResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 100)
    print("MATROID EVOLUTION EXPERIMENT RESULTS")
    print("=" * 100)
    print()
    print(f"{'Mode':>22}  {'AlwPos%':>8}  {'EvrNonU%':>9}  {'EvrNonI%':>9}  {'Acc%':>6}")
    print("-" * 100)

    for r in results:
        for mode in sorted(r.trials_by_mode.keys()):
            trials = r.trials_by_mode[mode]
            if not trials:
                continue
            ds = trials[0].dataset_name
            h = trials[0].hidden_dim
            print(
                f"{ds:>8} H={h:<3} {mode:>22}  "
                f"{r.always_positroid_rate(mode):>8.1%}  "
                f"{r.ever_nonuniform_rate(mode):>9.1%}  "
                f"{r.ever_noninterval_rate(mode):>9.1%}  "
                f"{r.mean_accuracy(mode):>6.1%}"
            )

    print()
    print("Legend:")
    print("  AlwPos%   = fraction of trials where matroid was ALWAYS positroid")
    print("  EvrNonU%  = fraction of trials that were ever non-uniform")
    print("  EvrNonI%  = fraction of trials where support was ever non-interval")
    print("  Acc%      = mean final training accuracy")
    print()


def print_detailed_results(results: list[EvolutionExperimentResult]) -> None:
    """Print epoch-by-epoch evolution for non-uniform/non-positroid trials."""
    print("\n" + "=" * 100)
    print("DETAILED EVOLUTION (non-uniform and/or non-positroid trials)")
    print("=" * 100)

    any_printed = False
    for r in results:
        for mode, trials in r.trials_by_mode.items():
            for i, t in enumerate(trials):
                has_nonuniform = any(not s.is_uniform for s in t.snapshots)
                has_nonpositroid = any(not s.is_positroid for s in t.snapshots)
                if not has_nonuniform and not has_nonpositroid:
                    continue
                any_printed = True

                tag = "NON-POS" if has_nonpositroid else "non-unif"
                print(
                    f"\n--- [{tag}] {t.dataset_name} H={t.hidden_dim} "
                    f"{mode} trial {i} (seed={t.seed}) ---"
                )
                if t.first_nonuniform_epoch is not None:
                    print(f"  First non-uniform: epoch {t.first_nonuniform_epoch}")
                if t.first_nonpositroid_epoch is not None:
                    print(f"  First non-positroid: epoch {t.first_nonpositroid_epoch}")
                if t.first_noninterval_epoch is not None:
                    print(f"  First non-interval support: epoch {t.first_noninterval_epoch}")

                print(
                    f"  {'epoch':>5}  {'loss':>8}  {'acc':>6}  "
                    f"{'#nb':>4}  {'support':>16}  {'intv':>5}  "
                    f"{'rdef':>4}  {'status':>10}  {'flags':>16}"
                )
                print("  " + "-" * 90)
                for s in t.snapshots:
                    status = "pos" if s.is_positroid else "NON-POS"
                    flags = []
                    if s.all_nonbases_are_intervals:
                        flags.append("all-interval")
                    else:
                        flags.append("gapped")
                    if s.contiguous_support_condition:
                        flags.append("CST")
                    flag_str = " ".join(flags)

                    support_str = (
                        "{}" if not s.support else "{" + ",".join(str(e) for e in s.support) + "}"
                    )
                    print(
                        f"  {s.epoch:>5}  {s.loss:>8.4f}  {s.accuracy:>6.1%}  "
                        f"{s.num_non_bases:>4}  {support_str:>16}  "
                        f"{'Y' if s.support_is_interval else 'N':>5}  "
                        f"{s.support_rank_deficiency:>4}  "
                        f"{status:>10}  {flag_str:>16}"
                    )

    if not any_printed:
        print("\n  All matroids were uniform positroids at every snapshot.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matroid Evolution Experiment: track matroid changes epoch-by-epoch",
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
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--param-modes",
        nargs="+",
        default=["tp_exponential", "negated_bidiagonal"],
    )
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    all_results: list[EvolutionExperimentResult] = []

    for dataset_name in args.datasets:
        for h in args.hidden_dims:
            print(f"Running {dataset_name}, H={h}...", flush=True)
            result = run_evolution_experiment(
                dataset_name=dataset_name,
                hidden_dim=h,
                num_trials=args.num_trials,
                param_modes=args.param_modes,
                n_samples=args.n_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                snapshot_interval=args.snapshot_interval,
                seed=args.seed,
            )
            for mode in args.param_modes:
                alw = result.always_positroid_rate(mode)
                enu = result.ever_nonuniform_rate(mode)
                print(f"  {mode}: always-pos={alw:.1%}, ever-nonuniform={enu:.1%}")
            all_results.append(result)

    print_results(all_results)

    if args.detailed:
        print_detailed_results(all_results)


if __name__ == "__main__":
    main()
