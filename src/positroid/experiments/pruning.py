"""Pruning Experiment.

Finding 005 showed that at H=200, ~98% of neurons are in the rank-deficient
tail support — only the first k~3 carry independent matroid rank. This
experiment tests whether that matroid-theoretic redundancy translates to
functional redundancy: can we remove tail neurons without losing accuracy?

Strategies:
- full_removal: delete matroid-identified tail neurons (shrinks network)
- random_removal: delete the same NUMBER of randomly-chosen neurons (control)
- direction_replacement: project tail W1 rows onto essential subspace,
  keep biases and W2 unchanged (validates matroid computation)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from positroid.datasets import DATASETS
from positroid.experiments.scale_evolution import get_augmented_matrix, rank_deficiency_support
from positroid.network.relu_network import ReluLayer, ReluNetwork
from positroid.network.train import TrainConfig, forward_pass, train

_MODE_SEED_OFFSET: dict[str, int] = {
    "tp_exponential": 0,
    "tp_cauchy": 1000,
    "unconstrained": 5000,
    "negated_bidiagonal": 6000,
}


# ---------------------------------------------------------------------------
# Core pruning functions
# ---------------------------------------------------------------------------


def identify_essential_and_tail(
    net: ReluNetwork,
) -> tuple[list[int], list[int], int]:
    """Partition neurons into essential and tail (rank-deficient support).

    Returns (essential_indices, tail_indices, rank) where both lists are
    sorted and partition [0, H).
    """
    aug, k = get_augmented_matrix(net)
    support = rank_deficiency_support(aug, k)
    h = aug.shape[0]
    tail = sorted(support)
    essential = sorted(set(range(h)) - support)
    return essential, tail, k


def prune_full_removal(
    net: ReluNetwork,
    neurons_to_remove: list[int],
) -> ReluNetwork:
    """Remove neurons entirely — delete W1 rows, b1 elements, W2 columns.

    Network shrinks from H to H-|removed|.
    """
    if not neurons_to_remove:
        return ReluNetwork(
            [
                ReluLayer(net.layers[0].weight.copy(), net.layers[0].bias.copy()),
                ReluLayer(net.layers[1].weight.copy(), net.layers[1].bias.copy()),
            ]
        )

    keep = sorted(set(range(net.layers[0].weight.shape[0])) - set(neurons_to_remove))
    w1 = net.layers[0].weight[keep]
    b1 = net.layers[0].bias[keep]
    w2 = net.layers[1].weight[:, keep]
    b2 = net.layers[1].bias.copy()
    return ReluNetwork([ReluLayer(w1, b1), ReluLayer(w2, b2)])


def prune_direction_replacement(
    net: ReluNetwork,
    neurons_to_replace: list[int],
    essential_indices: list[int],
) -> ReluNetwork:
    """Project replaced neurons' W1 rows onto the essential subspace.

    Keeps all biases and W2 unchanged. Network size is preserved.
    Raises ValueError if essential_indices is empty.
    """
    if not essential_indices:
        raise ValueError("Cannot project onto empty essential subspace")

    if not neurons_to_replace:
        return ReluNetwork(
            [
                ReluLayer(net.layers[0].weight.copy(), net.layers[0].bias.copy()),
                ReluLayer(net.layers[1].weight.copy(), net.layers[1].bias.copy()),
            ]
        )

    w1 = net.layers[0].weight.copy()

    # Build orthonormal basis for the essential rows' subspace via SVD
    essential_rows = w1[essential_indices]  # (n_ess, d)
    _u, s, vt = np.linalg.svd(essential_rows, full_matrices=False)
    # Keep only the non-negligible singular vectors
    tol = max(essential_rows.shape) * np.finfo(float).eps * s[0] if s[0] > 0 else 0
    r = int(np.sum(s > tol))
    basis = vt[:r]  # (r, d) — orthonormal rows spanning essential subspace

    # Project each replaced row: row -> basis^T @ (basis @ row)
    for idx in neurons_to_replace:
        row = w1[idx]
        coeffs = basis @ row  # (r,)
        w1[idx] = coeffs @ basis  # (d,)

    b1 = net.layers[0].bias.copy()
    w2 = net.layers[1].weight.copy()
    b2 = net.layers[1].bias.copy()
    return ReluNetwork([ReluLayer(w1, b1), ReluLayer(w2, b2)])


def evaluate_network(
    net: ReluNetwork,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate a trained network on data.

    Returns (accuracy, sigmoid_predictions).
    """
    w1 = net.layers[0].weight
    b1 = net.layers[0].bias
    w2 = net.layers[1].weight
    b2 = net.layers[1].bias
    y_pred, _, _, _ = forward_pass(x, w1, b1, w2, b2)
    preds_binary = (y_pred.ravel() > 0.5).astype(float)
    accuracy = float(np.mean(preds_binary == y))
    return accuracy, y_pred.ravel()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PrunePoint:
    """Results at one pruning level."""

    prune_fraction: float
    num_pruned: int
    num_kept: int
    pruned_accuracy: float
    accuracy_delta: float
    prediction_l2_dist: float


@dataclass
class PruningTrialResult:
    """One trial across all pruning levels."""

    dataset_name: str
    hidden_dim: int
    input_dim: int
    param_mode: str
    strategy: str
    seed: int
    original_accuracy: float
    support_size: int
    essential_size: int
    prune_curve: list[PrunePoint]


@dataclass
class PruningExperimentResult:
    """Aggregated results across strategies and modes."""

    trials_by_key: dict[str, list[PruningTrialResult]] = field(default_factory=dict)

    def add_trial(self, trial: PruningTrialResult) -> None:
        key = f"{trial.param_mode}:{trial.strategy}"
        if key not in self.trials_by_key:
            self.trials_by_key[key] = []
        self.trials_by_key[key].append(trial)

    def trials(self, mode: str, strategy: str) -> list[PruningTrialResult]:
        return self.trials_by_key.get(f"{mode}:{strategy}", [])

    def mean_accuracy_at_fraction(
        self,
        mode: str,
        strategy: str,
        fraction: float,
    ) -> float:
        ts = self.trials(mode, strategy)
        if not ts:
            return 0.0
        accs = []
        for t in ts:
            for p in t.prune_curve:
                if abs(p.prune_fraction - fraction) < 1e-9:
                    accs.append(p.pruned_accuracy)
                    break
        return float(np.mean(accs)) if accs else 0.0

    def mean_delta_at_fraction(
        self,
        mode: str,
        strategy: str,
        fraction: float,
    ) -> float:
        ts = self.trials(mode, strategy)
        if not ts:
            return 0.0
        deltas = []
        for t in ts:
            for p in t.prune_curve:
                if abs(p.prune_fraction - fraction) < 1e-9:
                    deltas.append(p.accuracy_delta)
                    break
        return float(np.mean(deltas)) if deltas else 0.0

    def mean_original_accuracy(self, mode: str, strategy: str) -> float:
        ts = self.trials(mode, strategy)
        if not ts:
            return 0.0
        return float(np.mean([t.original_accuracy for t in ts]))


# ---------------------------------------------------------------------------
# Trial & experiment runners
# ---------------------------------------------------------------------------


def run_pruning_trial(
    net: ReluNetwork,
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    param_mode: str,
    strategy: str,
    seed: int,
    prune_fractions: list[float] | None = None,
    n_random_samples: int = 20,
) -> PruningTrialResult | None:
    """Run one pruning trial: identify essential/tail, prune at each fraction.

    For random_removal, removes the same number of neurons as matroid-guided
    would, but selected randomly from all H neurons (averaged over
    n_random_samples draws).

    Returns None on numerical failure (e.g., augmented matrix extraction).
    """
    if prune_fractions is None:
        prune_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

    try:
        essential, tail, _rank = identify_essential_and_tail(net)
    except ValueError, np.linalg.LinAlgError:
        return None

    h = net.layers[0].weight.shape[0]
    input_dim = net.layers[0].weight.shape[1]
    original_acc, original_preds = evaluate_network(net, x, y)

    prune_curve: list[PrunePoint] = []

    for frac in prune_fractions:
        n_to_prune = int(round(frac * len(tail)))

        if strategy == "full_removal":
            # Prune from support head (lowest tail index) inward
            neurons_to_prune = tail[:n_to_prune]
            pruned_net = prune_full_removal(net, neurons_to_prune)
            pruned_acc, pruned_preds = evaluate_network(pruned_net, x, y)
            delta = pruned_acc - original_acc
            l2_dist = float(np.linalg.norm(pruned_preds - original_preds))
            num_kept = h - n_to_prune

        elif strategy == "random_removal":
            # Remove the same count as matroid-guided, but random neurons
            if n_to_prune == 0:
                pruned_acc = original_acc
                delta = 0.0
                l2_dist = 0.0
                num_kept = h
            else:
                rng = np.random.default_rng(seed * 1000 + int(frac * 1000))
                accs = []
                l2s = []
                for _ in range(n_random_samples):
                    random_neurons = rng.choice(
                        h,
                        size=min(n_to_prune, h),
                        replace=False,
                    ).tolist()
                    pruned_net = prune_full_removal(net, random_neurons)
                    acc_i, preds_i = evaluate_network(pruned_net, x, y)
                    accs.append(acc_i)
                    l2s.append(float(np.linalg.norm(preds_i - original_preds)))
                pruned_acc = float(np.mean(accs))
                delta = pruned_acc - original_acc
                l2_dist = float(np.mean(l2s))
                num_kept = h - n_to_prune

        elif strategy == "direction_replacement":
            neurons_to_prune = tail[:n_to_prune]
            if not essential and neurons_to_prune:
                # No basis to project onto — record original accuracy
                prune_curve.append(
                    PrunePoint(
                        prune_fraction=frac,
                        num_pruned=0,
                        num_kept=h,
                        pruned_accuracy=original_acc,
                        accuracy_delta=0.0,
                        prediction_l2_dist=0.0,
                    )
                )
                continue
            pruned_net = prune_direction_replacement(net, neurons_to_prune, essential)
            pruned_acc, pruned_preds = evaluate_network(pruned_net, x, y)
            delta = pruned_acc - original_acc
            l2_dist = float(np.linalg.norm(pruned_preds - original_preds))
            num_kept = h

        elif strategy == "magnitude_pruning":
            # Remove neurons with smallest L2 norm of W1 rows
            norms = np.linalg.norm(net.layers[0].weight, axis=1)
            order = np.argsort(norms).tolist()  # ascending: smallest first
            neurons_to_prune = order[:n_to_prune]
            pruned_net = prune_full_removal(net, neurons_to_prune)
            pruned_acc, pruned_preds = evaluate_network(pruned_net, x, y)
            delta = pruned_acc - original_acc
            l2_dist = float(np.linalg.norm(pruned_preds - original_preds))
            num_kept = h - n_to_prune

        elif strategy == "activation_pruning":
            # Remove neurons with lowest mean activation (APoZ-like)
            w1 = net.layers[0].weight
            b1_vec = net.layers[0].bias
            activations = x @ w1.T + b1_vec  # (n_samples, H)
            mean_acts = np.maximum(activations, 0).mean(axis=0)  # (H,)
            order = np.argsort(mean_acts).tolist()  # ascending
            neurons_to_prune = order[:n_to_prune]
            pruned_net = prune_full_removal(net, neurons_to_prune)
            pruned_acc, pruned_preds = evaluate_network(pruned_net, x, y)
            delta = pruned_acc - original_acc
            l2_dist = float(np.linalg.norm(pruned_preds - original_preds))
            num_kept = h - n_to_prune

        elif strategy == "sensitivity_pruning":
            # Remove neurons with smallest |W2[0,i]| * ||W1[i,:]||₂
            w1_norms = np.linalg.norm(net.layers[0].weight, axis=1)
            w2_abs = np.abs(net.layers[1].weight[0])
            sensitivity = w1_norms * w2_abs
            order = np.argsort(sensitivity).tolist()  # ascending
            neurons_to_prune = order[:n_to_prune]
            pruned_net = prune_full_removal(net, neurons_to_prune)
            pruned_acc, pruned_preds = evaluate_network(pruned_net, x, y)
            delta = pruned_acc - original_acc
            l2_dist = float(np.linalg.norm(pruned_preds - original_preds))
            num_kept = h - n_to_prune

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        prune_curve.append(
            PrunePoint(
                prune_fraction=frac,
                num_pruned=n_to_prune,
                num_kept=num_kept,
                pruned_accuracy=pruned_acc,
                accuracy_delta=delta,
                prediction_l2_dist=l2_dist,
            )
        )

    return PruningTrialResult(
        dataset_name=dataset_name,
        hidden_dim=h,
        input_dim=input_dim,
        param_mode=param_mode,
        strategy=strategy,
        seed=seed,
        original_accuracy=original_acc,
        support_size=len(tail),
        essential_size=len(essential),
        prune_curve=prune_curve,
    )


def run_pruning_experiment(
    dataset_name: str,
    hidden_dim: int,
    num_trials: int,
    param_modes: list[str],
    strategies: list[str],
    n_samples: int = 200,
    epochs: int = 100,
    learning_rate: float = 0.01,
    prune_fractions: list[float] | None = None,
    n_random_samples: int = 20,
    seed: int = 42,
) -> PruningExperimentResult:
    """Run pruning experiment: train networks, prune, evaluate.

    Triple-nested loop: trials -> modes -> strategies.
    Same (x, y) within a trial. Same trained network across strategies.
    """
    result = PruningExperimentResult()
    dataset_fn = DATASETS[dataset_name]

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        data_rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=data_rng)

        for mode in param_modes:
            mode_seed = trial_seed + _MODE_SEED_OFFSET.get(mode, 0)
            config = TrainConfig(
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                epochs=epochs,
                param_mode=mode,
                seed=mode_seed,
            )
            net, _history = train(x, y, config)

            for strategy in strategies:
                trial_result = run_pruning_trial(
                    net,
                    x,
                    y,
                    dataset_name,
                    mode,
                    strategy,
                    mode_seed,
                    prune_fractions=prune_fractions,
                    n_random_samples=n_random_samples,
                )
                if trial_result is not None:
                    result.add_trial(trial_result)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_results(
    results: list[PruningExperimentResult],
    param_modes: list[str],
    strategies: list[str],
    prune_fractions: list[float],
) -> None:
    """Print summary table."""
    print("\n" + "=" * 130)
    print("PRUNING EXPERIMENT RESULTS")
    print("=" * 130)
    print()
    print(
        f"{'Dataset':>8}  {'d':>2}  {'H':>4}  {'Mode':>22}  "
        f"{'Strategy':>24}  {'Prune%':>7}  {'OrigAcc%':>9}  "
        f"{'PrunAcc%':>9}  {'Delta%':>7}  {'#Kept/#Tot':>11}"
    )
    print("-" * 130)

    for r in results:
        for mode in param_modes:
            for strategy in strategies:
                ts = r.trials(mode, strategy)
                if not ts:
                    continue
                ds = ts[0].dataset_name
                d = ts[0].input_dim
                h = ts[0].hidden_dim
                orig_acc = r.mean_original_accuracy(mode, strategy)

                for frac in prune_fractions:
                    mean_acc = r.mean_accuracy_at_fraction(mode, strategy, frac)
                    mean_delta = r.mean_delta_at_fraction(mode, strategy, frac)

                    # Average kept/total
                    kept_vals = []
                    for t in ts:
                        for p in t.prune_curve:
                            if abs(p.prune_fraction - frac) < 1e-9:
                                kept_vals.append(p.num_kept)
                                break
                    avg_kept = np.mean(kept_vals) if kept_vals else h

                    print(
                        f"{ds:>8}  {d:>2}  {h:>4}  {mode:>22}  "
                        f"{strategy:>24}  {frac:>6.0%}  {orig_acc:>8.1%}  "
                        f"{mean_acc:>8.1%}  {mean_delta:>+6.1%}  "
                        f"{avg_kept:>5.0f}/{h:<5}"
                    )

    print()
    print("Legend:")
    print("  Prune%    = fraction of tail (support) neurons pruned")
    print("  OrigAcc%  = original network accuracy (before pruning)")
    print("  PrunAcc%  = accuracy after pruning")
    print("  Delta%    = PrunAcc% - OrigAcc%")
    print("  #Kept/#Tot= average neurons kept / total hidden dim")
    print()


def print_detailed_results(
    results: list[PruningExperimentResult],
    param_modes: list[str],
    strategies: list[str],
) -> None:
    """Print per-trial pruning curves."""
    print("\n" + "=" * 130)
    print("DETAILED PRUNING RESULTS (per-trial)")
    print("=" * 130)

    for r in results:
        for mode in param_modes:
            for strategy in strategies:
                ts = r.trials(mode, strategy)
                if not ts:
                    continue
                for i, t in enumerate(ts):
                    print(
                        f"\n--- {t.dataset_name} d={t.input_dim} H={t.hidden_dim} "
                        f"{mode} {strategy} trial {i} "
                        f"(seed={t.seed}) ---"
                    )
                    print(
                        f"  OrigAcc={t.original_accuracy:.1%}  "
                        f"Essential={t.essential_size}  "
                        f"Support={t.support_size}"
                    )
                    print(
                        f"  {'Prune%':>7}  {'#Pruned':>8}  {'#Kept':>6}  "
                        f"{'Acc%':>7}  {'Delta%':>7}  {'L2dist':>8}"
                    )
                    print("  " + "-" * 55)
                    for p in t.prune_curve:
                        print(
                            f"  {p.prune_fraction:>6.0%}  {p.num_pruned:>8}  "
                            f"{p.num_kept:>6}  {p.pruned_accuracy:>6.1%}  "
                            f"{p.accuracy_delta:>+6.1%}  {p.prediction_l2_dist:>8.3f}"
                        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pruning Experiment: test whether matroid-theoretic tail "
        "redundancy translates to functional redundancy",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["moons"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[6, 20, 50, 100, 200],
    )
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--param-modes",
        nargs="+",
        default=["tp_exponential", "negated_bidiagonal"],
    )
    parser.add_argument(
        "--prune-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "full_removal",
            "random_removal",
            "magnitude_pruning",
            "activation_pruning",
            "sensitivity_pruning",
        ],
    )
    parser.add_argument("--n-random-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    all_results: list[PruningExperimentResult] = []

    for dataset_name in args.datasets:
        for h in args.hidden_dims:
            print(f"Running {dataset_name}, H={h}...", flush=True)
            result = run_pruning_experiment(
                dataset_name=dataset_name,
                hidden_dim=h,
                num_trials=args.num_trials,
                param_modes=args.param_modes,
                strategies=args.strategies,
                n_samples=args.n_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                prune_fractions=args.prune_fractions,
                n_random_samples=args.n_random_samples,
                seed=args.seed,
            )
            # Quick progress
            for mode in args.param_modes:
                for strategy in args.strategies:
                    ts = result.trials(mode, strategy)
                    if ts:
                        orig = result.mean_original_accuracy(mode, strategy)
                        d100 = result.mean_delta_at_fraction(mode, strategy, 1.0)
                        print(f"  {mode} {strategy}: orig={orig:.1%}, delta@100%={d100:+.1%}")
            all_results.append(result)

    print_results(
        all_results,
        args.param_modes,
        args.strategies,
        args.prune_fractions,
    )

    if args.detailed:
        print_detailed_results(all_results, args.param_modes, args.strategies)


if __name__ == "__main__":
    main()
