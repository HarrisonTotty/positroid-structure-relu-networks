"""Diagnose the three CREATED cases from the scale evolution experiment.

These three trials showed training apparently creating non-uniformity from a
uniform initialization. This script reproduces each case with dense per-epoch
snapshots to determine whether the transitions are genuine structural changes
or numerical near-boundary crossings where the matrix was always on the edge.

Cases:
  1. moons H=50  seed=44 tp_exponential, 100 epochs
  2. moons H=200 seed=45 tp_exponential, 100 epochs
  3. digits_0v1_pca10 H=12 seed=44 tp_exponential, 100 epochs
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from positroid.datasets import DATASETS
from positroid.experiments.scale_evolution import (
    contiguous_window_ranks,
    get_augmented_matrix,
    min_window_singular_value,
)
from positroid.matroid.positroid import is_positroid
from positroid.network.relu_network import ReluNetwork
from positroid.network.train import TrainConfig, train

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------


@dataclass
class CaseDef:
    dataset: str
    hidden_dim: int
    seed: int
    label: str


CASES = [
    CaseDef("moons", 50, 44, "moons H=50 seed=44"),
    CaseDef("moons", 200, 45, "moons H=200 seed=45"),
    CaseDef("digits_0v1_pca10", 12, 44, "digits_0v1_pca10 H=12 seed=44"),
]

N_SAMPLES = 200
EPOCHS = 100
LR = 0.01
PARAM_MODE = "tp_exponential"


# ---------------------------------------------------------------------------
# Per-window singular value analysis
# ---------------------------------------------------------------------------


def window_singular_values(aug: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (start, min_sv) for each contiguous k-window."""
    n = aug.shape[0]
    results = []
    for i in range(n - k + 1):
        svs = np.linalg.svd(aug[i : i + k], compute_uv=False)
        results.append((i, float(svs[-1])))
    return results


def find_weakest_window(aug: np.ndarray, k: int) -> tuple[int, float, float]:
    """Return (start, min_sv, condition_number) of the weakest k-window."""
    n = aug.shape[0]
    best_start = 0
    best_min_sv = float("inf")
    best_cond = 1.0
    for i in range(n - k + 1):
        svs = np.linalg.svd(aug[i : i + k], compute_uv=False)
        if svs[-1] < best_min_sv:
            best_min_sv = float(svs[-1])
            best_start = i
            best_cond = float(svs[0] / svs[-1]) if svs[-1] > 0 else float("inf")
    return best_start, best_min_sv, best_cond


def deficient_windows(aug: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Return list of (start, rank) for rank-deficient windows."""
    return [(s, r) for s, r in contiguous_window_ranks(aug, k) if r < k]


# ---------------------------------------------------------------------------
# Exact matroid check (feasibility-gated)
# ---------------------------------------------------------------------------

MAX_EXACT = 5_000_000


def exact_check(net: ReluNetwork, n: int, k: int) -> str:
    """Return 'U' (uniform), 'pos', 'NON-POS', or '---' (infeasible)."""
    if comb(n, k) > MAX_EXACT:
        return "---"
    try:
        arr = net.hyperplane_arrangement(layer_idx=0)
        mat = arr.affine_matroid()
        if mat.is_uniform():
            return "U"
        return "pos" if is_positroid(mat) else "NON-POS"
    except ValueError, np.linalg.LinAlgError:
        return "err"


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def diagnose_case(case: CaseDef) -> None:
    dataset_name = case.dataset
    hidden_dim = case.hidden_dim
    seed = case.seed
    label = case.label

    print(f"\n{'=' * 100}")
    print(f"CASE: {label}  (param_mode={PARAM_MODE})")
    print(f"{'=' * 100}")

    # Generate data (same seed as scale experiment)
    data_rng = np.random.default_rng(seed)
    dataset_fn = DATASETS[dataset_name]
    x, y = dataset_fn(n_samples=N_SAMPLES, rng=data_rng)
    print(f"Dataset: {dataset_name}, shape={x.shape}, classes={np.unique(y)}")

    # Train with dense snapshots (every epoch for small H, sparser for large H)
    if hidden_dim <= 50:
        all_epochs = list(range(EPOCHS))
    else:
        # For large H: every 5th epoch + first/last 10
        all_epochs = sorted(
            set(
                list(range(10))
                + list(range(0, EPOCHS, 5))
                + list(range(max(0, EPOCHS - 10), EPOCHS))
            )
        )
    config = TrainConfig(
        hidden_dim=hidden_dim,
        learning_rate=LR,
        epochs=EPOCHS,
        param_mode=PARAM_MODE,
        seed=seed,
    )
    _, history = train(x, y, config, snapshot_epochs=all_epochs)

    # ----- Phase 1: Find transition epoch -----
    print("\n--- Epoch-by-epoch min singular value trajectory ---")
    header = (
        f"{'epoch':>5}  {'minSV':>12}  {'#def':>5}  {'weakest_win':>12}  {'cond':>12}  {'exact':>8}"
    )
    print(header)
    print("-" * 70)

    transition_epoch = None

    for ep in sorted(history.snapshots.keys()):
        net = history.snapshots[ep]
        try:
            aug, k = get_augmented_matrix(net)
        except ValueError, np.linalg.LinAlgError:
            print(f"  {ep:>5}  {'ERR':>12}")
            continue

        n = aug.shape[0]
        min_sv = min_window_singular_value(aug, k)
        dw = deficient_windows(aug, k)
        n_def = len(dw)
        weak_start, weak_sv, weak_cond = find_weakest_window(aug, k)

        # Only do exact check sparingly (expensive for large n)
        do_exact = (
            ep == 0
            or ep == EPOCHS - 1
            or (transition_epoch is not None and abs(ep - transition_epoch) <= 3)
            or n_def > 0
        )
        exact_str = exact_check(net, n, k) if do_exact else ""

        # Detect transition
        marker = ""
        if n_def > 0 and transition_epoch is None:
            transition_epoch = ep
            marker = " <-- TRANSITION"
        if n_def > 0:
            marker = marker or " *"

        # Only print: near transition, first/last 5, every 10th, or deficient
        show = (
            ep < 5
            or ep >= EPOCHS - 5
            or ep % 10 == 0
            or n_def > 0
            or (transition_epoch is not None and abs(ep - transition_epoch) <= 5)
        )
        if show:
            print(
                f"  {ep:>5}  {min_sv:>12.6e}  {n_def:>5}  "
                f"win@{weak_start:>4}  {weak_cond:>12.2e}  {exact_str:>8}{marker}"
            )

    # ----- Phase 2: Zoom in on transition -----
    if transition_epoch is not None:
        print(f"\n--- Zoom: transition at epoch {transition_epoch} ---")
        zoom_start = max(0, transition_epoch - 5)
        zoom_end = min(EPOCHS, transition_epoch + 6)

        print(f"{'epoch':>5}  {'minSV':>12}  {'#def':>5}  {'deficient_windows':>40}  {'exact':>8}")
        print("-" * 85)

        for ep in range(zoom_start, zoom_end):
            if ep not in history.snapshots:
                continue
            net = history.snapshots[ep]
            try:
                aug, k = get_augmented_matrix(net)
            except ValueError, np.linalg.LinAlgError:
                continue

            n = aug.shape[0]
            min_sv = min_window_singular_value(aug, k)
            dw = deficient_windows(aug, k)
            n_def = len(dw)

            # Show which windows are deficient
            dw_str = ", ".join(f"[{s}..{s + k - 1}]r={r}" for s, r in dw) if dw else "none"

            exact_str = exact_check(net, n, k)

            print(f"  {ep:>5}  {min_sv:>12.6e}  {n_def:>5}  {dw_str:>40}  {exact_str:>8}")

        # ----- Phase 3: Singular value trajectory of the weakest window -----
        # Identify which window is deficient at transition, track it backward
        net_trans = history.snapshots[transition_epoch]
        aug_trans, k_trans = get_augmented_matrix(net_trans)
        dw_trans = deficient_windows(aug_trans, k_trans)
        if dw_trans:
            target_start = dw_trans[0][0]
            print(
                f"\n--- SV trajectory of window [{target_start}..{target_start + k_trans - 1}] ---"
            )
            print(f"{'epoch':>5}  {'sv_min':>12}  {'sv_2nd':>12}  {'rank':>5}")
            print("-" * 50)

            track_start = max(0, transition_epoch - 20)
            track_end = min(EPOCHS, transition_epoch + 11)
            for ep in range(track_start, track_end):
                if ep not in history.snapshots:
                    continue
                net_ep = history.snapshots[ep]
                try:
                    aug_ep, k_ep = get_augmented_matrix(net_ep)
                except ValueError, np.linalg.LinAlgError:
                    continue
                n_ep = aug_ep.shape[0]
                if target_start + k_ep > n_ep:
                    continue
                window = aug_ep[target_start : target_start + k_ep]
                svs = np.linalg.svd(window, compute_uv=False)
                rank_w = int(np.linalg.matrix_rank(window))
                sv_min = float(svs[-1])
                sv_2nd = float(svs[-2]) if len(svs) > 1 else float("nan")
                marker = " <--" if ep == transition_epoch else ""
                print(f"  {ep:>5}  {sv_min:>12.6e}  {sv_2nd:>12.6e}  {rank_w:>5}{marker}")
    else:
        print("\n  No transition found -- matroid stayed uniform throughout!")
        print("  (This case may not reproduce with these exact parameters.)")

    # ----- Phase 4: Init condition analysis -----
    print("\n--- Initialization analysis (epoch 0) ---")
    net0 = history.snapshots[0]
    try:
        aug0, k0 = get_augmented_matrix(net0)
        n0 = aug0.shape[0]
        print(f"  n={n0}, k={k0}, C(n,k)={comb(n0, k0):,}")

        weak_start, weak_sv, weak_cond = find_weakest_window(aug0, k0)
        print(f"  Weakest window: [{weak_start}..{weak_start + k0 - 1}]")
        print(f"    min SV = {weak_sv:.6e}")
        print(f"    condition number = {weak_cond:.2e}")

        # Show all window SVs at init
        wsv = window_singular_values(aug0, k0)
        sorted_wsv = sorted(wsv, key=lambda t: t[1])
        print("  Bottom 5 windows by min SV:")
        for start, sv in sorted_wsv[:5]:
            print(f"    [{start:>3}..{start + k0 - 1:>3}]  sv={sv:.6e}")

        # numpy rank threshold for reference
        svs_full = np.linalg.svd(aug0, compute_uv=False)
        rank_tol = max(n0, k0) * svs_full[0] * np.finfo(aug0.dtype).eps
        print(f"  numpy rank threshold (for reference): {rank_tol:.6e}")
        print(f"  Ratio weakest_sv / rank_threshold: {weak_sv / rank_tol:.2f}")
    except ValueError, np.linalg.LinAlgError:
        print("  Failed to extract augmented matrix at epoch 0")

    # ----- Phase 5: Final summary -----
    print(f"\n--- Final epoch ({EPOCHS - 1}) ---")
    net_final = history.snapshots[EPOCHS - 1]
    try:
        aug_f, k_f = get_augmented_matrix(net_final)
        n_f = aug_f.shape[0]
        dw_f = deficient_windows(aug_f, k_f)
        min_sv_f = min_window_singular_value(aug_f, k_f)
        exact_f = exact_check(net_final, n_f, k_f)
        print(f"  n={n_f}, k={k_f}, min_sv={min_sv_f:.6e}, #deficient={len(dw_f)}, exact={exact_f}")
        if dw_f:
            for s, r in dw_f:
                print(f"    deficient: [{s}..{s + k_f - 1}] rank={r}")
    except ValueError, np.linalg.LinAlgError:
        print("  Failed to extract augmented matrix at final epoch")


def main() -> None:
    print("DIAGNOSE CREATED CASES")
    print("=" * 100)
    print("Goal: determine whether 'CREATED' non-uniformity is genuine")
    print("structural change or numerical near-boundary crossing.")
    print()
    print("Key indicators of NUMERICAL ARTIFACT:")
    print("  - min SV decreasing monotonically toward zero over many epochs")
    print("  - Condition number at init already very large (>1e8)")
    print("  - Weakest SV at init only ~10-100x above rank threshold")
    print()
    print("Key indicators of GENUINE structural change:")
    print("  - min SV stable then drops suddenly in 1-2 epochs")
    print("  - Condition number at init is moderate (<1e6)")
    print("  - Weakest SV at init is >1000x above rank threshold")

    import sys

    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        cases = [CASES[i] for i in indices]
    else:
        cases = CASES

    for case in cases:
        diagnose_case(case)

    print("\n" + "=" * 100)
    print("DIAGNOSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
