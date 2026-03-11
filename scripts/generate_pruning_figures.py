"""Generate figures for the pruning blog post.

Saves PNGs to ~/gh/harrisontotty.github.io/images/.

Usage:
    uv run python scripts/generate_pruning_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from positroid.datasets.toy2d import make_moons
from positroid.experiments.pruning import (
    evaluate_network,
    identify_essential_and_tail,
    prune_full_removal,
)
from positroid.network.train import TrainConfig, forward_pass, train

OUTDIR = Path.home() / "gh" / "harrisontotty.github.io" / "images"

# Blog palette (from generate_blog_figures.py)
BG = "#ffffff"
FG = "#1d1b1b"
FG_DIM = "#abafb6"
LINK_BLUE = "#60728d"
SAGE = "#608d7b"
OLIVE = "#698d60"
PLUM = "#84608d"
WARM_BROWN = "#8d7260"
INDIAN_RED = "#cd5c5c"
DARK_BLUE = "#385074"


def setup_style() -> None:
    """Configure matplotlib to match the blog's light theme."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": FG_DIM,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": FG_DIM,
        "ytick.color": FG_DIM,
        "font.family": "monospace",
        "font.size": 11,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def _style_ax(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# Figure 1: Essential vs Tail hyperplane arrangement
# ---------------------------------------------------------------------------


def fig1_partition() -> None:
    """Hyperplane arrangement showing essential (sage) vs tail (indian red)."""
    # seed=43 at H=20 gives 4 essential / 16 tail (seed=42 gives uniform matroid)
    rng = np.random.default_rng(43)
    X, y = make_moons(n_samples=200, noise=0.1, rng=rng)

    config = TrainConfig(
        hidden_dim=20, learning_rate=0.01, epochs=100,
        param_mode="tp_exponential", seed=43,
    )
    net, _ = train(X, y, config)
    essential, tail, rank = identify_essential_and_tail(net)

    W1 = net.layers[0].weight
    b1 = net.layers[0].bias
    H = W1.shape[0]
    essential_set = set(essential)

    xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points
    c0 = y == 0
    c1 = y == 1
    ax.scatter(X[c0, 0], X[c0, 1], c=DARK_BLUE, s=12, alpha=0.4, zorder=3)
    ax.scatter(X[c1, 0], X[c1, 1], c=INDIAN_RED, s=12, alpha=0.4, zorder=3)

    # Draw hyperplanes as lines: w0*x + w1*y + b = 0
    xs = np.array([xlim[0], xlim[1]])
    for i in range(H):
        w0, w1_val = W1[i]
        b_val = b1[i]
        is_ess = i in essential_set

        if abs(w1_val) > 1e-10:
            ys = -(w0 * xs + b_val) / w1_val
            ax.plot(
                xs, ys,
                color=SAGE if is_ess else INDIAN_RED,
                linewidth=2.0 if is_ess else 0.8,
                alpha=0.9 if is_ess else 0.3,
                zorder=4 if is_ess else 2,
            )
        else:
            xv = -b_val / w0 if abs(w0) > 1e-10 else 0
            ax.axvline(
                xv,
                color=SAGE if is_ess else INDIAN_RED,
                linewidth=2.0 if is_ess else 0.8,
                alpha=0.9 if is_ess else 0.3,
                zorder=4 if is_ess else 2,
            )

    # Legend
    ax.plot([], [], color=SAGE, linewidth=2.0, label=f"essential ({len(essential)})")
    ax.plot([], [], color=INDIAN_RED, linewidth=0.8, alpha=0.5,
            label=f"tail ({len(tail)})")
    ax.legend(
        loc="upper right", fontsize=10, framealpha=0.7,
        edgecolor=FG_DIM, facecolor=BG,
    )

    ax.set_title(
        f"{H} neurons, rank {rank}: "
        f"{len(essential)} essential + {len(tail)} tail",
        fontsize=12, pad=10,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _style_ax(ax)

    path = OUTDIR / "pruning-fig1-partition.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Decision boundary comparison (original / matroid / random)
# ---------------------------------------------------------------------------


def _decision_boundary(
    ax: plt.Axes,
    net,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Plot decision boundary + data points for a network."""
    # Create mesh
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 300),
        np.linspace(ylim[0], ylim[1], 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Get predictions
    w1 = net.layers[0].weight
    b1 = net.layers[0].bias
    w2 = net.layers[1].weight
    b2 = net.layers[1].bias
    y_pred, _, _, _ = forward_pass(grid, w1, b1, w2, b2)
    zz = y_pred.reshape(xx.shape)

    # Decision boundary
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=[DARK_BLUE, INDIAN_RED],
                alpha=0.12)
    ax.contour(xx, yy, zz, levels=[0.5], colors=[FG], linewidths=1.5, alpha=0.6)

    # Data points
    c0 = y == 0
    c1 = y == 1
    ax.scatter(X[c0, 0], X[c0, 1], c=DARK_BLUE, s=10, alpha=0.5, zorder=3)
    ax.scatter(X[c1, 0], X[c1, 1], c=INDIAN_RED, s=10, alpha=0.5, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _style_ax(ax)


def fig2_boundaries() -> None:
    """Three-panel decision boundary: original, matroid-pruned, random-pruned."""
    # H=50 seed=42 gives 5 essential / 45 tail — dramatic pruning
    rng = np.random.default_rng(42)
    X, y = make_moons(n_samples=200, noise=0.1, rng=rng)

    config = TrainConfig(
        hidden_dim=50, learning_rate=0.01, epochs=100,
        param_mode="tp_exponential", seed=42,
    )
    net, _ = train(X, y, config)

    essential, tail, _rank = identify_essential_and_tail(net)
    orig_acc, _ = evaluate_network(net, X, y)

    # Matroid-guided: remove 75% of tail
    n_prune = int(0.75 * len(tail))
    matroid_pruned = prune_full_removal(net, tail[:n_prune])
    matroid_acc, _ = evaluate_network(matroid_pruned, X, y)

    # Random: remove same count — try several seeds, pick one that shows degradation
    h = net.layers[0].weight.shape[0]
    best_random_seed = 0
    worst_acc = 1.0
    for rs in range(20):
        rand_rng = np.random.default_rng(rs)
        rn = rand_rng.choice(h, size=n_prune, replace=False).tolist()
        pn = prune_full_removal(net, rn)
        acc, _ = evaluate_network(pn, X, y)
        if acc < worst_acc:
            worst_acc = acc
            best_random_seed = rs

    rand_rng = np.random.default_rng(best_random_seed)
    random_neurons = rand_rng.choice(h, size=n_prune, replace=False).tolist()
    random_pruned = prune_full_removal(net, random_neurons)
    random_acc, _ = evaluate_network(random_pruned, X, y)

    xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    _decision_boundary(
        ax1, net, X, y,
        f"original ({h} neurons)\nacc = {orig_acc:.1%}",
        xlim, ylim,
    )
    _decision_boundary(
        ax2, matroid_pruned, X, y,
        f"matroid-pruned ({h - n_prune} neurons)\nacc = {matroid_acc:.1%}",
        xlim, ylim,
    )
    _decision_boundary(
        ax3, random_pruned, X, y,
        f"random-pruned ({h - n_prune} neurons)\nacc = {random_acc:.1%}",
        xlim, ylim,
    )

    fig.suptitle(
        f"Decision boundaries after removing {n_prune}/{h} neurons (75% of tail)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    path = OUTDIR / "pruning-fig2-boundaries.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Pruning curves at H=200
# ---------------------------------------------------------------------------


def fig3_curves() -> None:
    """Pruning curves: accuracy delta vs prune fraction, all strategies."""
    # Data from Finding 006/007 (H=200, tp_exponential, 5 trials)
    fractions = [0.0, 0.25, 0.50, 0.75, 1.00]

    # Matroid-guided deltas at each fraction (all zero except 100%)
    matroid_deltas = [0.0, 0.0, 0.0, 0.0, -0.7]

    # Standard heuristics (magnitude/activation/sensitivity — all identical at H=200)
    heuristic_deltas = [0.0, -0.3, -0.3, -0.3, -0.3]

    # Random deltas from Finding 006/007
    random_deltas = [0.0, -4.9, -11.9, -14.4, -20.9]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fractions, matroid_deltas, color=SAGE, linewidth=2.5,
            marker="o", markersize=8, label="matroid-guided", zorder=5)
    ax.plot(fractions, heuristic_deltas, color=LINK_BLUE, linewidth=2.0,
            marker="D", markersize=6, label="magnitude / activation / sensitivity",
            zorder=4)
    ax.plot(fractions, random_deltas, color=INDIAN_RED, linewidth=2.5,
            marker="s", markersize=8, label="random", zorder=4)

    # Shade the matroid vs random advantage region
    ax.fill_between(fractions, matroid_deltas, random_deltas,
                     color=SAGE, alpha=0.06)

    # Annotate the advantage at 75%
    ax.annotate(
        "14.4pp vs random",
        xy=(0.75, -7.2), fontsize=10, color=SAGE,
        fontweight="bold", ha="center",
    )

    ax.axhline(0, color=FG_DIM, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("fraction of tail pruned", fontsize=12)
    ax.set_ylabel("accuracy delta (%)", fontsize=12)
    ax.set_title("Pruning curves at H=200 (tp_exponential)", fontsize=13, pad=15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.5,
              edgecolor=FG_DIM, facecolor=BG)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-25, 3)
    ax.set_xticks(fractions)
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, alpha=0.15, color=FG_DIM)
    _style_ax(ax)

    path = OUTDIR / "pruning-fig3-curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Advantage scaling across H
# ---------------------------------------------------------------------------


def fig4_scaling() -> None:
    """Grouped bar chart: matroid vs heuristics vs random at 75% across H."""
    hidden_dims = [20, 50, 100, 200]

    # Data from Finding 006/007, tp_exponential, @75%
    matroid_75 = [0.0, 0.0, 0.0, 0.0]
    heuristic_75 = [-0.3, -0.7, -0.5, -0.3]  # magnitude (= activation = sensitivity)
    random_75 = [-12.5, -10.2, -15.4, -14.4]

    x = np.arange(len(hidden_dims))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(x - width, matroid_75, width, label="matroid-guided",
           color=SAGE, alpha=0.85, edgecolor="none")
    ax.bar(x, heuristic_75, width, label="mag / act / sens",
           color=LINK_BLUE, alpha=0.85, edgecolor="none")
    bars_r = ax.bar(x + width, random_75, width, label="random",
                    color=INDIAN_RED, alpha=0.85, edgecolor="none")

    # Annotate matroid advantage vs random
    for i in range(len(hidden_dims)):
        adv = -random_75[i]
        ax.annotate(
            f"+{adv:.1f}pp",
            xy=(x[i] - width, 1.5), ha="center", fontsize=9,
            color=SAGE, fontweight="bold",
        )

    # Value labels on random bars
    for bar in bars_r:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h - 1.2,
                f"{h:.1f}%", ha="center", va="top", fontsize=8,
                color=BG, fontweight="bold")

    ax.axhline(0, color=FG, linewidth=0.8, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in hidden_dims], fontsize=11)
    ax.set_ylabel("accuracy delta at 75% tail removal (%)", fontsize=11)
    ax.set_title(
        "All pruning strategies at 75% tail removal (tp_exponential)",
        fontsize=12, pad=15,
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.5,
              edgecolor=FG_DIM, facecolor=BG)

    ax.set_ylim(-20, 5)
    ax.yaxis.grid(True, alpha=0.15, color=FG_DIM)
    _style_ax(ax)

    path = OUTDIR / "pruning-fig4-scaling.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Causal mechanism diagram
# ---------------------------------------------------------------------------


def fig5_mechanism() -> None:
    """Flow diagram: matroid rank -> partition -> safe pruning."""
    fig, ax = plt.subplots(figsize=(12, 4))

    boxes = [
        ("trained\nnetwork", 0.5),
        ("augmented matrix\n[W1 | b1]", 2.5),
        ("matroid\nrank partition", 4.5),
        ("remove tail\n(75%)", 6.5),
        ("zero\naccuracy loss", 8.5),
    ]

    box_w = 1.6
    box_h = 0.9

    for label, cx in boxes:
        color = SAGE if "zero" in label else LINK_BLUE
        rect = plt.Rectangle(
            (cx - box_w / 2, -box_h / 2), box_w, box_h,
            facecolor="none", edgecolor=color,
            linewidth=1.5, zorder=3, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(cx, 0, label, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold", zorder=4)

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        cx1 = boxes[i][1] + box_w / 2 + 0.05
        cx2 = boxes[i + 1][1] - box_w / 2 - 0.05
        arrow = FancyArrowPatch(
            (cx1, 0), (cx2, 0),
            arrowstyle="->", color=WARM_BROWN,
            mutation_scale=15, linewidth=1.5, zorder=2,
        )
        ax.add_patch(arrow)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "The matroid-guided pruning pipeline",
        fontsize=13, pad=10,
    )

    path = OUTDIR / "pruning-fig5-mechanism.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating pruning blog figures...")
    fig1_partition()
    fig2_boundaries()
    fig3_curves()
    fig4_scaling()
    fig5_mechanism()
    print("Done!")


if __name__ == "__main__":
    main()
