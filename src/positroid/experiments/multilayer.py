"""Multi-layer positroid experiment.

Tests whether positroid structure survives ReLU composition in deeper networks.
For a 2-hidden-layer network [d, H1, H2, 1], computes "effective matroids"
at layer 1 by folding the layer-0 activation pattern into the weight matrices:

    W_eff = W2 @ diag(D) @ W1
    b_eff = W2 @ diag(D) @ b1 + b2

The matroid of [W_eff | b_eff] has H2 elements and rank <= d+1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from positroid.datasets import DATASETS
from positroid.matroid.linear_matroid import linear_matroid_from_vectors
from positroid.matroid.matroid import Matroid
from positroid.matroid.positroid import is_positroid, support_is_cyclic_interval
from positroid.network.relu_network import ReluNetwork
from positroid.network.train_multilayer import (
    MultiLayerTrainConfig,
    train_multilayer,
)

_MODE_SEED_OFFSET: dict[str, int] = {
    "tp_exponential": 0,
    "unconstrained": 2000,
}

MAX_EXACT_SUBSETS = 50_000


# ── Effective Matrix Computation ──


def _row_normalize(aug: np.ndarray) -> np.ndarray:
    """Two-pass overflow-safe row normalization."""
    row_max = np.abs(aug).max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    scaled = aug / row_max
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return scaled / norms


def compute_effective_matrix(
    net: ReluNetwork,
    activation_pattern: np.ndarray,
) -> np.ndarray:
    """Compute effective augmented matrix [W_eff | b_eff] for a 2-hidden-layer net.

    For activation pattern D (binary vector over H1 neurons):
        W_eff = W2 @ diag(D) @ W1    in R^{H2 x d}
        b_eff = W2 @ diag(D) @ b1 + b2   in R^{H2}

    Returns row-normalized augmented matrix of shape (H2, d+1).
    """
    w1 = net.layers[0].weight  # (H1, d)
    b1 = net.layers[0].bias  # (H1,)
    w2 = net.layers[1].weight  # (H2, H1)
    b2 = net.layers[1].bias  # (H2,)

    d = activation_pattern.astype(float)  # (H1,)
    # W_eff = W2 @ diag(D) @ W1
    w_eff = w2 @ (d[:, None] * w1)  # (H2, d)
    # b_eff = W2 @ (D * b1) + b2
    b_eff = w2 @ (d * b1) + b2  # (H2,)

    aug = np.hstack([w_eff, b_eff.reshape(-1, 1)])  # (H2, d+1)
    return _row_normalize(aug)


# ── Activation Region Sampling ──


def sample_activation_regions(
    net: ReluNetwork,
    x: np.ndarray,
    max_regions: int = 200,
) -> dict[tuple[int, ...], list[int]]:
    """Group data points by their layer-0 activation pattern.

    Returns dict mapping pattern (as tuple of 0/1) to list of sample indices.
    Capped at max_regions most-populated regions.
    """
    patterns = net.activation_pattern(x)
    layer0_patterns = patterns[0]  # (n_samples, H1) boolean

    region_map: dict[tuple[int, ...], list[int]] = {}
    for i in range(x.shape[0]):
        key = tuple(int(v) for v in layer0_patterns[i])
        if key not in region_map:
            region_map[key] = []
        region_map[key].append(i)

    if len(region_map) > max_regions:
        # Keep most-populated regions
        sorted_keys = sorted(region_map.keys(), key=lambda k: len(region_map[k]), reverse=True)
        region_map = {k: region_map[k] for k in sorted_keys[:max_regions]}

    return region_map


def activation_pattern_matroid(
    regions: dict[tuple[int, ...], list[int]],
) -> Matroid | None:
    """Column matroid of K×H1 binary pattern matrix.

    Rows = observed activation patterns, columns = neurons.
    The matroid on [H1] captures which subsets of neurons
    are "independently distinguishable" across regions.

    Returns None if there are fewer than 2 patterns.
    """
    patterns = list(regions.keys())
    if len(patterns) < 2:
        return None
    mat = np.array(patterns, dtype=float)  # K × H1
    # Column matroid: vectors are columns of mat = rows of mat.T
    return linear_matroid_from_vectors(mat.T)


def region_adjacency_graph(
    regions: dict[tuple[int, ...], list[int]],
) -> tuple[dict[tuple[int, ...], list[tuple[int, ...]]], int]:
    """Build adjacency graph: edges between patterns at Hamming distance 1.

    Returns:
        adjacency: dict mapping pattern → list of adjacent patterns
        n_edges: total edge count
    """
    patterns = list(regions.keys())
    adjacency: dict[tuple[int, ...], list[tuple[int, ...]]] = {p: [] for p in patterns}
    n_edges = 0
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            # Hamming distance
            dist = sum(a != b for a, b in zip(patterns[i], patterns[j], strict=True))
            if dist == 1:
                adjacency[patterns[i]].append(patterns[j])
                adjacency[patterns[j]].append(patterns[i])
                n_edges += 1
    return adjacency, n_edges


def is_activation_contiguous(pattern: tuple[int, ...] | np.ndarray) -> bool:
    """Check if active neurons form a contiguous block (no gaps).

    All-active and all-inactive are considered contiguous.
    """
    active = [i for i, v in enumerate(pattern) if v]
    if len(active) <= 1:
        return True
    return active[-1] - active[0] == len(active) - 1


# ── Data Structures ──


@dataclass
class EffectiveMatroidResult:
    """Result for one activation region's effective matroid."""

    activation_pattern: tuple[int, ...]
    n_samples_in_region: int
    pattern_is_contiguous: bool
    n_active: int
    effective_rank: int
    is_uniform: bool | None
    is_positroid: bool | None
    num_non_bases: int
    support_is_interval: bool


@dataclass
class MultilayerTrialResult:
    """Result for one training trial."""

    dataset_name: str
    layer_dims: list[int]
    param_mode: str
    seed: int
    final_loss: float
    final_accuracy: float
    layer0_is_positroid: bool | None
    layer0_is_uniform: bool | None
    n_activation_regions: int
    effective_results: list[EffectiveMatroidResult]
    all_effective_positroid: bool
    any_effective_non_positroid: bool
    contiguous_patterns_all_positroid: bool
    # Activation pattern matroid fields
    pattern_matroid_is_positroid: bool | None = None
    pattern_matroid_rank: int = 0
    pattern_matroid_num_non_bases: int = 0
    n_adjacent_pairs: int = 0


@dataclass
class MultilayerExperimentResult:
    """Aggregated experiment results."""

    trials_by_mode: dict[str, list[MultilayerTrialResult]] = field(
        default_factory=dict,
    )

    def add_trial(self, mode: str, trial: MultilayerTrialResult) -> None:
        if mode not in self.trials_by_mode:
            self.trials_by_mode[mode] = []
        self.trials_by_mode[mode].append(trial)


# ── Analysis ──


def _analyze_matroid(aug: np.ndarray, rank: int) -> tuple[bool | None, bool | None, int, bool]:
    """Analyze matroid properties. Returns (is_pos, is_unif, n_nonbases, supp_interval)."""
    from math import comb

    n = aug.shape[0]
    k = rank

    if k <= 0 or k > n:
        return None, None, 0, True

    n_subsets = comb(n, k)
    if n_subsets > MAX_EXACT_SUBSETS:
        return None, None, 0, True

    try:
        matroid = linear_matroid_from_vectors(aug)
    except ValueError:
        return None, None, 0, True

    is_unif = matroid.is_uniform()
    if is_unif:
        return True, True, 0, True

    is_pos = is_positroid(matroid)
    n_nonbases = n_subsets - len(matroid.bases)
    supp_interval = support_is_cyclic_interval(matroid)
    return is_pos, is_unif, n_nonbases, supp_interval


def _analyze_layer0(net: ReluNetwork) -> tuple[bool | None, bool | None]:
    """Analyze layer 0 matroid (existing single-layer analysis)."""
    try:
        arr = net.hyperplane_arrangement(layer_idx=0)
        aug = arr.augmented_matrix()
        aug_normed = _row_normalize(aug)
        rank = int(np.linalg.matrix_rank(aug_normed))
        is_pos, is_unif, _, _ = _analyze_matroid(aug_normed, rank)
        return is_pos, is_unif
    except ValueError, NotImplementedError:
        return None, None


# ── Trial Runner ──


def run_single_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    param_mode: str,
    layer_dims: list[int],
    epochs: int,
    learning_rate: float,
    seed: int,
    max_regions: int = 200,
) -> MultilayerTrialResult:
    """Run a single multi-layer training trial and analyze effective matroids."""
    config = MultiLayerTrainConfig(
        layer_dims=layer_dims,
        learning_rate=learning_rate,
        epochs=epochs,
        param_mode=param_mode,
        seed=seed,
    )
    net, history = train_multilayer(x, y, config)

    # Layer 0 matroid
    l0_pos, l0_unif = _analyze_layer0(net)

    # Sample activation regions
    regions = sample_activation_regions(net, x, max_regions=max_regions)

    effective_results: list[EffectiveMatroidResult] = []
    for pattern_key, sample_indices in regions.items():
        pattern_arr = np.array(pattern_key)
        n_active = int(np.sum(pattern_arr))
        contiguous = is_activation_contiguous(pattern_key)

        # Compute effective matrix
        aug_eff = compute_effective_matrix(net, pattern_arr)
        eff_rank = int(np.linalg.matrix_rank(aug_eff))

        is_pos, is_unif, n_nonbases, supp_int = _analyze_matroid(aug_eff, eff_rank)

        effective_results.append(
            EffectiveMatroidResult(
                activation_pattern=pattern_key,
                n_samples_in_region=len(sample_indices),
                pattern_is_contiguous=contiguous,
                n_active=n_active,
                effective_rank=eff_rank,
                is_uniform=is_unif,
                is_positroid=is_pos,
                num_non_bases=n_nonbases,
                support_is_interval=supp_int,
            )
        )

    # Aggregate
    positroid_results = [r for r in effective_results if r.is_positroid is not None]
    all_pos = all(r.is_positroid for r in positroid_results) if positroid_results else True
    any_non_pos = any(r.is_positroid is False for r in effective_results)

    contiguous_results = [
        r for r in effective_results if r.pattern_is_contiguous and r.is_positroid is not None
    ]
    contig_all_pos = all(r.is_positroid for r in contiguous_results) if contiguous_results else True

    # Activation pattern matroid
    pat_mat = activation_pattern_matroid(regions)
    pat_pos: bool | None = None
    pat_rank = 0
    pat_nb = 0
    if pat_mat is not None:
        pat_rank = pat_mat.rank
        n_elements = len(pat_mat.ground_set)
        from math import comb

        total_subsets = comb(n_elements, pat_rank)
        pat_nb = total_subsets - len(pat_mat.bases)
        if total_subsets <= MAX_EXACT_SUBSETS:
            pat_pos = True if pat_mat.is_uniform() else is_positroid(pat_mat)

    # Region adjacency graph
    _, n_adj = region_adjacency_graph(regions)

    return MultilayerTrialResult(
        dataset_name=dataset_name,
        layer_dims=layer_dims,
        param_mode=param_mode,
        seed=seed,
        final_loss=history.losses[-1] if history.losses else float("nan"),
        final_accuracy=history.accuracies[-1] if history.accuracies else 0.0,
        layer0_is_positroid=l0_pos,
        layer0_is_uniform=l0_unif,
        n_activation_regions=len(regions),
        effective_results=effective_results,
        all_effective_positroid=all_pos,
        any_effective_non_positroid=any_non_pos,
        contiguous_patterns_all_positroid=contig_all_pos,
        pattern_matroid_is_positroid=pat_pos,
        pattern_matroid_rank=pat_rank,
        pattern_matroid_num_non_bases=pat_nb,
        n_adjacent_pairs=n_adj,
    )


# ── Experiment Runner ──


def run_multilayer_experiment(
    dataset_name: str = "moons",
    layer_dims: list[int] | None = None,
    num_trials: int = 10,
    param_modes: list[str] | None = None,
    n_samples: int = 200,
    epochs: int = 200,
    learning_rate: float = 0.01,
    max_regions: int = 200,
    seed: int = 42,
) -> MultilayerExperimentResult:
    """Run the full multi-layer positroid experiment."""
    if layer_dims is None:
        layer_dims = [8, 6]
    if param_modes is None:
        param_modes = ["tp_exponential", "unconstrained"]

    dataset_fn = DATASETS[dataset_name]
    result = MultilayerExperimentResult()

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=rng)

        for mode in param_modes:
            mode_seed = trial_seed + _MODE_SEED_OFFSET.get(mode, 0)
            trial_result = run_single_trial(
                dataset_name=dataset_name,
                x=x,
                y=y,
                param_mode=mode,
                layer_dims=layer_dims,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=mode_seed,
                max_regions=max_regions,
            )
            result.add_trial(mode, trial_result)

    return result


# ── Output ──


def print_results(result: MultilayerExperimentResult, param_modes: list[str]) -> None:
    """Print summary table."""
    print()
    print("=" * 110)
    print("Multi-Layer Positroid Experiment Results")
    print("=" * 110)
    print(
        f"{'Mode':<20s} {'AllPos%':>8s} {'ContigPos%':>11s} {'AnyNon%':>8s}"
        f" {'#Regions':>9s} {'L0Pos%':>7s} {'Acc%':>6s}"
        f" {'PatPos%':>8s} {'AvgAdj':>7s}"
    )
    print("-" * 110)

    for mode in param_modes:
        trials = result.trials_by_mode.get(mode, [])
        if not trials:
            continue

        n = len(trials)
        all_pos_pct = 100.0 * sum(t.all_effective_positroid for t in trials) / n
        contig_pos_pct = 100.0 * sum(t.contiguous_patterns_all_positroid for t in trials) / n
        any_non_pct = 100.0 * sum(t.any_effective_non_positroid for t in trials) / n
        avg_regions = sum(t.n_activation_regions for t in trials) / n
        l0_pos_count = sum(1 for t in trials if t.layer0_is_positroid is True)
        l0_total = sum(1 for t in trials if t.layer0_is_positroid is not None)
        l0_pct = 100.0 * l0_pos_count / l0_total if l0_total > 0 else float("nan")
        avg_acc = 100.0 * sum(t.final_accuracy for t in trials) / n
        pat_pos_count = sum(1 for t in trials if t.pattern_matroid_is_positroid is True)
        pat_total = sum(1 for t in trials if t.pattern_matroid_is_positroid is not None)
        pat_pct = 100.0 * pat_pos_count / pat_total if pat_total > 0 else float("nan")
        avg_adj = sum(t.n_adjacent_pairs for t in trials) / n

        print(
            f"{mode:<20s} {all_pos_pct:>7.1f}% {contig_pos_pct:>10.1f}%"
            f" {any_non_pct:>7.1f}% {avg_regions:>9.1f} {l0_pct:>6.1f}% {avg_acc:>5.1f}%"
            f" {pat_pct:>7.1f}% {avg_adj:>7.1f}"
        )

    print("=" * 110)


def print_detailed_results(result: MultilayerExperimentResult, param_modes: list[str]) -> None:
    """Print per-trial and per-region details."""
    for mode in param_modes:
        trials = result.trials_by_mode.get(mode, [])
        if not trials:
            continue

        print(f"\n{'=' * 80}")
        print(f"Mode: {mode} ({len(trials)} trials)")
        print(f"{'=' * 80}")

        for i, trial in enumerate(trials):
            print(f"\n  Trial {i} (seed={trial.seed}, acc={trial.final_accuracy:.1%})")
            print(f"    Layer0: pos={trial.layer0_is_positroid}, unif={trial.layer0_is_uniform}")
            n_reg = trial.n_activation_regions
            print(f"    Regions: {n_reg}, AllPos={trial.all_effective_positroid}")
            print(
                f"    Pattern matroid: rank={trial.pattern_matroid_rank},"
                f" positroid={trial.pattern_matroid_is_positroid},"
                f" non-bases={trial.pattern_matroid_num_non_bases},"
                f" adj_pairs={trial.n_adjacent_pairs}"
            )

            for j, er in enumerate(trial.effective_results):
                pos_str = "POS" if er.is_positroid else ("NON" if er.is_positroid is False else "?")
                unif_str = "U" if er.is_uniform else "NU"
                contig_str = "C" if er.pattern_is_contiguous else "NC"
                print(
                    f"    Region {j:3d}: act={er.n_active:2d}/{len(er.activation_pattern):2d}"
                    f" {contig_str:>2s} rank={er.effective_rank}"
                    f" {unif_str:>2s} {pos_str:>3s}"
                    f" nb={er.num_non_bases} si={er.support_is_interval}"
                    f" ({er.n_samples_in_region} pts)"
                )


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-layer positroid experiment",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["moons"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--layer-dims",
        type=int,
        nargs="+",
        default=[8, 6],
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument(
        "--param-modes",
        nargs="+",
        default=["tp_exponential", "unconstrained"],
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-regions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        print(f"\nDataset: {dataset_name}")
        print(f"Layer dims: {args.layer_dims}")

        result = run_multilayer_experiment(
            dataset_name=dataset_name,
            layer_dims=args.layer_dims,
            num_trials=args.num_trials,
            param_modes=args.param_modes,
            n_samples=args.n_samples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_regions=args.max_regions,
            seed=args.seed,
        )

        print_results(result, args.param_modes)

        if args.detailed:
            print_detailed_results(result, args.param_modes)


if __name__ == "__main__":
    main()
