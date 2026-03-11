"""Scale Evolution Experiment.

Tests whether the "initialization determines everything" finding from Finding 004
survives at scale. At d=2 (rank k=3), C(H,3) grows cubically — C(200,3)=1.3M —
so exact matroid computation is feasible even at H=200. For larger d, proxy metrics
based on contiguous window ranks and random sampling are used.

Key hypothesis: Created% should be 0% for tp_exponential — matroid structure is a
property of the TP weight manifold, not the learning algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np

from positroid.datasets import DATASETS
from positroid.matroid.linear_matroid import linear_matroid_from_vectors
from positroid.matroid.positroid import is_cyclic_interval, is_positroid
from positroid.network.relu_network import ReluNetwork
from positroid.network.train import TrainConfig, train

# The real bottleneck for exact matroid computation is the exchange axiom
# validation O(|B|^2). With |B| ~ C(n,k), we need C(n,k)^2 to be tractable.
# 50K subsets → ~2.5B validation checks → ~30s. Safe for H≤50 at k=3.
MAX_EXACT_SUBSETS = 50_000

_MODE_SEED_OFFSET: dict[str, int] = {
    "tp_exponential": 0,
    "tp_cauchy": 1000,
    "negated_bidiagonal": 6000,
}


# ---------------------------------------------------------------------------
# Proxy metric functions
# ---------------------------------------------------------------------------


def get_augmented_matrix(net: ReluNetwork) -> tuple[np.ndarray, int]:
    """Return (row-normalized augmented matrix, rank) from layer-0 arrangement.

    The augmented matrix is row-normalized (each row divided by its L2 norm)
    to eliminate the extreme condition numbers produced by the exponential
    kernel at large H (condition numbers of 1e100+). Row normalization
    preserves the row matroid (same linear dependencies between rows)
    while making numerical rank determination reliable.
    """
    arr = net.hyperplane_arrangement(layer_idx=0)
    aug = arr.augmented_matrix()
    # Two-pass normalization to avoid overflow when squaring large entries:
    # 1. Scale each row by its max absolute value (overflow-safe)
    # 2. Then normalize to unit length
    row_max = np.abs(aug).max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    scaled = aug / row_max
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    aug_normed = scaled / norms
    k = int(np.linalg.matrix_rank(aug_normed))
    return aug_normed, k


def contiguous_window_ranks(
    aug: np.ndarray,
    k: int,
) -> list[tuple[int, int]]:
    """Compute rank of each contiguous k-window of rows.

    Returns list of (start_index, rank) for windows [i, i+k).
    """
    n = aug.shape[0]
    results = []
    for i in range(n - k + 1):
        rank = int(np.linalg.matrix_rank(aug[i : i + k]))
        results.append((i, rank))
    return results


def random_subset_rank_test(
    aug: np.ndarray,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample random k-subsets and count rank-deficient ones.

    Returns (n_deficient, n_tested).
    """
    n = aug.shape[0]
    n_deficient = 0
    for _ in range(n_samples):
        indices = rng.choice(n, size=k, replace=False)
        rank = int(np.linalg.matrix_rank(aug[indices]))
        if rank < k:
            n_deficient += 1
    return n_deficient, n_samples


def rank_deficiency_support(aug: np.ndarray, k: int) -> frozenset[int]:
    """Union of all indices in any rank-deficient contiguous k-window."""
    support: set[int] = set()
    for start, rank in contiguous_window_ranks(aug, k):
        if rank < k:
            support.update(range(start, start + k))
    return frozenset(support)


def min_window_singular_value(aug: np.ndarray, k: int) -> float:
    """Smallest singular value across all contiguous k-windows."""
    n = aug.shape[0]
    min_sv = float("inf")
    for i in range(n - k + 1):
        svs = np.linalg.svd(aug[i : i + k], compute_uv=False)
        min_sv = min(min_sv, float(svs[-1]))
    return min_sv


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScaleSnapshot:
    """Per-epoch metrics combining exact and proxy analysis."""

    epoch: int
    loss: float
    accuracy: float
    ground_set_size: int
    rank: int

    # Exact (None if C(n,k) > threshold)
    exact_is_uniform: bool | None = None
    exact_is_positroid: bool | None = None
    exact_num_non_bases: int | None = None
    exact_support: tuple[int, ...] | None = None

    # Proxy (always computed)
    num_rank_deficient_windows: int = 0
    proxy_support: tuple[int, ...] = ()
    proxy_support_size: int = 0
    proxy_support_is_interval: bool = True
    min_singular_value: float = 0.0

    # Random sampling
    random_samples_tested: int = 0
    random_deficient: int = 0
    random_noncontiguous_deficient: int = 0

    @property
    def is_nonuniform(self) -> bool:
        """Determine non-uniformity from exact or proxy."""
        if self.exact_is_uniform is not None:
            return not self.exact_is_uniform
        return self.num_rank_deficient_windows > 0


@dataclass
class ScaleTrialResult:
    """One trial's full timeline at scale."""

    dataset_name: str
    hidden_dim: int
    input_dim: int
    param_mode: str
    seed: int
    exact_mode: bool

    final_loss: float
    final_accuracy: float
    snapshots: list[ScaleSnapshot]

    # Init-vs-final summary
    init_nonuniform: bool = False
    final_nonuniform: bool = False
    training_created_nonuniformity: bool = False
    training_eliminated_nonuniformity: bool = False


@dataclass
class ScaleExperimentResult:
    """Aggregated results across parameter modes."""

    trials_by_mode: dict[str, list[ScaleTrialResult]] = field(default_factory=dict)

    def _trials(self, mode: str) -> list[ScaleTrialResult]:
        return self.trials_by_mode.get(mode, [])

    def init_nonuniform_rate(self, mode: str) -> float:
        trials = self._trials(mode)
        if not trials:
            return 0.0
        return sum(t.init_nonuniform for t in trials) / len(trials)

    def final_nonuniform_rate(self, mode: str) -> float:
        trials = self._trials(mode)
        if not trials:
            return 0.0
        return sum(t.final_nonuniform for t in trials) / len(trials)

    def training_created_rate(self, mode: str) -> float:
        trials = self._trials(mode)
        if not trials:
            return 0.0
        return sum(t.training_created_nonuniformity for t in trials) / len(trials)

    def training_eliminated_rate(self, mode: str) -> float:
        trials = self._trials(mode)
        if not trials:
            return 0.0
        return sum(t.training_eliminated_nonuniformity for t in trials) / len(trials)

    def always_contiguous_rate(self, mode: str) -> float:
        trials = self._trials(mode)
        if not trials:
            return 0.0
        return sum(all(s.proxy_support_is_interval for s in t.snapshots) for t in trials) / len(
            trials
        )

    def always_positroid_rate(self, mode: str) -> float:
        """Only meaningful when all trials ran in exact mode."""
        trials = self._trials(mode)
        if not trials:
            return 0.0
        exact_trials = [t for t in trials if t.exact_mode]
        if not exact_trials:
            return float("nan")
        return sum(
            all(s.exact_is_positroid is True for s in t.snapshots) for t in exact_trials
        ) / len(exact_trials)

    def mean_accuracy(self, mode: str) -> float:
        trials = self._trials(mode)
        if not trials:
            return 0.0
        return float(np.mean([t.final_accuracy for t in trials]))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_scale_snapshot(
    net: ReluNetwork,
    epoch: int,
    loss: float,
    accuracy: float,
    n_random_samples: int,
    rng: np.random.Generator,
    max_exact_subsets: int = MAX_EXACT_SUBSETS,
) -> ScaleSnapshot | None:
    """Analyze a network snapshot with both proxy and (optionally) exact metrics.

    Returns None if augmented matrix extraction fails.
    """
    try:
        aug, k = get_augmented_matrix(net)
    except ValueError, np.linalg.LinAlgError:
        return None

    n = aug.shape[0]

    # Proxy metrics (always computed)
    window_ranks = contiguous_window_ranks(aug, k)
    num_deficient = sum(1 for _, r in window_ranks if r < k)
    # Derive support directly from already-computed window_ranks
    proxy_sup: set[int] = set()
    for start, rank in window_ranks:
        if rank < k:
            proxy_sup.update(range(start, start + k))
    proxy_sup_frozen = frozenset(proxy_sup)
    proxy_sup_tuple = tuple(sorted(proxy_sup_frozen))
    proxy_is_interval = True if not proxy_sup_frozen else is_cyclic_interval(proxy_sup_frozen, n)
    min_sv = min_window_singular_value(aug, k)

    # Random sampling — single loop for both deficient and non-contiguous counts
    random_def = 0
    random_tested = 0
    random_noncont_def = 0
    if n_random_samples > 0:
        random_tested = n_random_samples
        for _ in range(n_random_samples):
            indices = rng.choice(n, size=k, replace=False)
            rank = int(np.linalg.matrix_rank(aug[indices]))
            if rank < k:
                random_def += 1
                idx_set = frozenset(int(i) for i in indices)
                if not is_cyclic_interval(idx_set, n):
                    random_noncont_def += 1

    # Exact metrics (if feasible)
    exact_is_uniform: bool | None = None
    exact_is_positroid: bool | None = None
    exact_num_non_bases: int | None = None
    exact_support: tuple[int, ...] | None = None

    if comb(n, k) <= max_exact_subsets:
        try:
            # Use the row-normalized augmented matrix (same as proxy) for
            # exact matroid construction. This ensures proxy and exact use
            # identical numerical conditioning.
            aff_mat = linear_matroid_from_vectors(aug)

            all_k_subsets = frozenset(frozenset(s) for s in combinations(range(n), k))
            non_bases = all_k_subsets - aff_mat.bases

            exact_is_uniform = aff_mat.is_uniform()
            exact_is_positroid = is_positroid(aff_mat)
            exact_num_non_bases = len(non_bases)

            ex_support = frozenset().union(*non_bases) if non_bases else frozenset()
            exact_support = tuple(sorted(ex_support))

            # Validation: proxy support should be subset of exact support.
            # Both use the same normalized matrix, so this should hold.
            # Warn (don't assert) in case of borderline numerical disagreement.
            if not proxy_sup_frozen <= ex_support:
                import warnings

                warnings.warn(
                    f"Proxy support {sorted(proxy_sup_frozen)} not subset of "
                    f"exact support {sorted(ex_support)} at epoch {epoch}",
                    stacklevel=2,
                )
        except ValueError:
            pass

    return ScaleSnapshot(
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        ground_set_size=n,
        rank=k,
        exact_is_uniform=exact_is_uniform,
        exact_is_positroid=exact_is_positroid,
        exact_num_non_bases=exact_num_non_bases,
        exact_support=exact_support,
        num_rank_deficient_windows=num_deficient,
        proxy_support=proxy_sup_tuple,
        proxy_support_size=len(proxy_sup),
        proxy_support_is_interval=proxy_is_interval,
        min_singular_value=min_sv,
        random_samples_tested=random_tested,
        random_deficient=random_def,
        random_noncontiguous_deficient=random_noncont_def,
    )


def run_scale_trial(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    param_mode: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    snapshot_epochs: list[int],
    seed: int,
    n_random_samples: int = 10000,
    max_exact_subsets: int = MAX_EXACT_SUBSETS,
) -> ScaleTrialResult:
    """Train a network and analyze matroid evolution at scale."""
    config = TrainConfig(
        hidden_dim=hidden_dim,
        learning_rate=lr,
        epochs=epochs,
        param_mode=param_mode,
        seed=seed,
    )
    _, history = train(x, y, config, snapshot_epochs=snapshot_epochs)

    rng = np.random.default_rng(seed + 999)
    input_dim = x.shape[1]

    snapshots: list[ScaleSnapshot] = []
    for ep in sorted(history.snapshots.keys()):
        net = history.snapshots[ep]
        loss = history.losses[ep] if ep < len(history.losses) else float("inf")
        acc = history.accuracies[ep] if ep < len(history.accuracies) else 0.0
        snap = analyze_scale_snapshot(
            net,
            ep,
            loss,
            acc,
            n_random_samples,
            rng,
            max_exact_subsets,
        )
        if snap is not None:
            snapshots.append(snap)

    # Determine exact mode: all snapshots have exact data
    exact_mode = all(s.exact_is_uniform is not None for s in snapshots) and len(snapshots) > 0

    final_loss = history.losses[-1] if history.losses else float("inf")
    final_acc = history.accuracies[-1] if history.accuracies else 0.0

    # Init-vs-final summary
    init_nu = snapshots[0].is_nonuniform if snapshots else False
    final_nu = snapshots[-1].is_nonuniform if snapshots else False

    return ScaleTrialResult(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        param_mode=param_mode,
        seed=seed,
        exact_mode=exact_mode,
        final_loss=final_loss,
        final_accuracy=final_acc,
        snapshots=snapshots,
        init_nonuniform=init_nu,
        final_nonuniform=final_nu,
        training_created_nonuniformity=not init_nu and final_nu,
        training_eliminated_nonuniformity=init_nu and not final_nu,
    )


def run_scale_experiment(
    dataset_name: str,
    hidden_dim: int,
    num_trials: int,
    param_modes: list[str],
    n_samples: int = 200,
    epochs: int = 100,
    learning_rate: float = 0.01,
    snapshot_epochs: list[int] | None = None,
    seed: int = 42,
    n_random_samples: int = 10000,
    max_exact_subsets: int = MAX_EXACT_SUBSETS,
) -> ScaleExperimentResult:
    """Run the scale evolution experiment for one configuration."""
    result = ScaleExperimentResult()
    for mode in param_modes:
        result.trials_by_mode[mode] = []

    if snapshot_epochs is None:
        mid = epochs // 2
        snapshot_epochs = sorted({0, mid, epochs - 1})

    dataset_fn = DATASETS[dataset_name]

    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        data_rng = np.random.default_rng(trial_seed)
        x, y = dataset_fn(n_samples=n_samples, rng=data_rng)

        for mode in param_modes:
            mode_seed = trial_seed + _MODE_SEED_OFFSET.get(mode, 0)
            trial_result = run_scale_trial(
                dataset_name,
                x,
                y,
                mode,
                hidden_dim,
                epochs,
                learning_rate,
                snapshot_epochs,
                mode_seed,
                n_random_samples,
                max_exact_subsets,
            )
            result.trials_by_mode[mode].append(trial_result)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_results(results: list[ScaleExperimentResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 120)
    print("SCALE EVOLUTION EXPERIMENT RESULTS")
    print("=" * 120)
    print()
    print(
        f"{'Dataset':>8}  {'d':>2}  {'H':>4}  {'Mode':>22}  "
        f"{'InitNU%':>8}  {'FinalNU%':>9}  {'Created%':>9}  "
        f"{'Elim%':>6}  {'AlwCont%':>9}  {'AlwPos%':>8}  {'Acc%':>6}"
    )
    print("-" * 120)

    for r in results:
        for mode in sorted(r.trials_by_mode.keys()):
            trials = r.trials_by_mode[mode]
            if not trials:
                continue
            ds = trials[0].dataset_name
            d = trials[0].input_dim
            h = trials[0].hidden_dim

            alw_pos = r.always_positroid_rate(mode)
            alw_pos_str = f"{alw_pos:>8.1%}" if not np.isnan(alw_pos) else "     ---"

            print(
                f"{ds:>8}  {d:>2}  {h:>4}  {mode:>22}  "
                f"{r.init_nonuniform_rate(mode):>8.1%}  "
                f"{r.final_nonuniform_rate(mode):>9.1%}  "
                f"{r.training_created_rate(mode):>9.1%}  "
                f"{r.training_eliminated_rate(mode):>6.1%}  "
                f"{r.always_contiguous_rate(mode):>9.1%}  "
                f"{alw_pos_str}  "
                f"{r.mean_accuracy(mode):>6.1%}"
            )

    print()
    print("Legend:")
    print("  InitNU%   = fraction of trials non-uniform at initialization")
    print("  FinalNU%  = fraction of trials non-uniform at final epoch")
    print("  Created%  = training CREATED non-uniformity (key column: should be 0%)")
    print("  Elim%     = training eliminated non-uniformity")
    print("  AlwCont%  = proxy support always contiguous interval")
    print("  AlwPos%   = always positroid (exact mode only, '---' if proxy)")
    print("  Acc%      = mean final training accuracy")
    print()


def print_detailed_results(results: list[ScaleExperimentResult]) -> None:
    """Print per-trial snapshots for non-uniform trials."""
    print("\n" + "=" * 120)
    print("DETAILED SCALE EVOLUTION (non-uniform trials)")
    print("=" * 120)

    any_printed = False
    for r in results:
        for mode, trials in r.trials_by_mode.items():
            for i, t in enumerate(trials):
                if not any(s.is_nonuniform for s in t.snapshots):
                    continue
                any_printed = True

                flags = []
                if t.training_created_nonuniformity:
                    flags.append("CREATED")
                if t.training_eliminated_nonuniformity:
                    flags.append("eliminated")
                flag_str = " ".join(flags) if flags else "init-only"
                exact_str = "exact" if t.exact_mode else "proxy"

                print(
                    f"\n--- [{flag_str}] {t.dataset_name} d={t.input_dim} "
                    f"H={t.hidden_dim} {mode} trial {i} "
                    f"(seed={t.seed}, {exact_str}) ---"
                )

                print(
                    f"  {'epoch':>5}  {'loss':>8}  {'acc':>6}  "
                    f"{'#defW':>5}  {'proxySupp':>20}  {'intv':>5}  "
                    f"{'minSV':>10}  {'rndDef':>6}  "
                    f"{'exact':>8}"
                )
                print("  " + "-" * 100)
                for s in t.snapshots:
                    support_str = (
                        "{}"
                        if not s.proxy_support
                        else "{" + ",".join(str(e) for e in s.proxy_support) + "}"
                    )
                    exact_str_snap = ""
                    if s.exact_is_uniform is not None:
                        if s.exact_is_uniform:
                            exact_str_snap = "U"
                        elif s.exact_is_positroid:
                            exact_str_snap = "pos"
                        else:
                            exact_str_snap = "NON-POS"

                    rnd_str = (
                        f"{s.random_deficient}/{s.random_samples_tested}"
                        if s.random_samples_tested > 0
                        else "---"
                    )

                    print(
                        f"  {s.epoch:>5}  {s.loss:>8.4f}  {s.accuracy:>6.1%}  "
                        f"{s.num_rank_deficient_windows:>5}  {support_str:>20}  "
                        f"{'Y' if s.proxy_support_is_interval else 'N':>5}  "
                        f"{s.min_singular_value:>10.2e}  {rnd_str:>6}  "
                        f"{exact_str_snap:>8}"
                    )

    if not any_printed:
        print("\n  All matroids were uniform at every snapshot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale Evolution Experiment: test init-determines-everything at scale",
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
        "--snapshot-epochs",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--param-modes",
        nargs="+",
        default=["tp_exponential", "negated_bidiagonal"],
    )
    parser.add_argument("--n-random-samples", type=int, default=10000)
    parser.add_argument("--max-exact-subsets", type=int, default=MAX_EXACT_SUBSETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    all_results: list[ScaleExperimentResult] = []

    for dataset_name in args.datasets:
        for h in args.hidden_dims:
            print(f"Running {dataset_name}, H={h}...", flush=True)
            result = run_scale_experiment(
                dataset_name=dataset_name,
                hidden_dim=h,
                num_trials=args.num_trials,
                param_modes=args.param_modes,
                n_samples=args.n_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                snapshot_epochs=args.snapshot_epochs,
                seed=args.seed,
                n_random_samples=args.n_random_samples,
                max_exact_subsets=args.max_exact_subsets,
            )
            for mode in args.param_modes:
                cr = result.training_created_rate(mode)
                inu = result.init_nonuniform_rate(mode)
                print(f"  {mode}: init-NU={inu:.1%}, created={cr:.1%}")
            all_results.append(result)

    print_results(all_results)

    if args.detailed:
        print_detailed_results(all_results)


if __name__ == "__main__":
    main()
