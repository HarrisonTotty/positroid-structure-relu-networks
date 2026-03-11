"""Generate figures for the positroid structure paper.

Outputs PDFs to papers/1-positroid-structure/figures/.

Usage:
    uv run python scripts/generate_paper_figures.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

from positroid.datasets.toy2d import make_moons
from positroid.experiments.pruning import (
    identify_essential_and_tail,
)
from positroid.matroid.positroid import is_positroid
from positroid.network.train import TrainConfig, train

OUTDIR = Path(__file__).resolve().parent.parent / "papers" / "1-positroid-structure" / "figures"

# Academic-friendly palette (grayscale-safe with selective color)
BLACK = "#000000"
DARK_GRAY = "#333333"
MID_GRAY = "#888888"
LIGHT_GRAY = "#cccccc"
ACCENT_GREEN = "#2d7d46"   # positroid / essential / good
ACCENT_RED = "#c0392b"     # non-positroid / tail / bad
ACCENT_BLUE = "#2c3e80"    # data class 0
ACCENT_ORANGE = "#d4760a"  # data class 1


def setup_style() -> None:
    """Configure matplotlib for academic paper figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MID_GRAY,
        "axes.labelcolor": BLACK,
        "text.color": BLACK,
        "xtick.color": DARK_GRAY,
        "ytick.color": DARK_GRAY,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
    })


def _style_ax(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(MID_GRAY)
    ax.tick_params(width=0.5, length=3)


# ── Figure 1: Hyperplane arrangement on two-moons ──────────────────────────


def fig1_hyperplanes() -> None:
    """Train a TP network on moons and plot hyperplanes over data."""
    # seed=24 produces a non-trivial positroid (10 non-bases at H=8)
    rng = np.random.default_rng(24)
    X, y = make_moons(n_samples=300, noise=0.1, rng=rng)

    config = TrainConfig(
        hidden_dim=8,
        learning_rate=0.05,
        epochs=300,
        batch_size=32,
        tp_constrained=True,
        tp_kernel="exponential",
        seed=24,
    )
    network, _ = train(X, y, config)

    arrangement = network.hyperplane_arrangement(0)
    matroid = arrangement.affine_matroid()
    n_bases = len(matroid.bases)
    total = len(list(combinations(range(matroid.size), matroid.rank)))
    n_nonbases = total - n_bases

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Data points
    c0 = y == 0
    c1 = y == 1
    ax.scatter(X[c0, 0], X[c0, 1], c=ACCENT_BLUE, s=8, alpha=0.4,
               zorder=3, label="class 0")
    ax.scatter(X[c1, 0], X[c1, 1], c=ACCENT_ORANGE, s=8, alpha=0.4,
               zorder=3, label="class 1")

    # Hyperplane lines
    xlims = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    xs = np.linspace(xlims[0], xlims[1], 200)

    for hp in arrangement.hyperplanes:
        w1, w2 = hp.normal
        b = hp.bias
        if abs(w2) > 1e-10:
            ys = -(w1 * xs + b) / w2
            mask = (ys > X[:, 1].min() - 1) & (ys < X[:, 1].max() + 1)
            ax.plot(xs[mask], ys[mask], color=MID_GRAY, alpha=0.6,
                    linewidth=0.8, zorder=2)
        else:
            xv = -b / w1
            ax.axvline(xv, color=MID_GRAY, alpha=0.6, linewidth=0.8, zorder=2)

    ax.set_xlim(xlims)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7,
              edgecolor=LIGHT_GRAY)
    _style_ax(ax)

    path = OUTDIR / "fig-hyperplanes.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: Cyclic interval vs non-interval ──────────────────────────────


def _draw_circle_nodes(
    ax: plt.Axes,
    n: int,
    radius: float = 1.0,
    highlight: set[int] | None = None,
    highlight_color: str = ACCENT_GREEN,
    draw_arc: bool = False,
    title: str = "",
) -> None:
    """Draw n nodes on a circle, highlighting a subset."""
    positions = []
    for i in range(n):
        angle = np.pi / 2 - 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append((x, y))

    # Circle outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            color=LIGHT_GRAY, linewidth=0.5)

    # Highlighted arc if contiguous
    if draw_arc and highlight:
        elems = sorted(highlight)
        gaps = [(elems[(i + 1) % len(elems)] - elems[i]) % n
                for i in range(len(elems))]
        if all(g == 1 for g in gaps[:-1]):
            start_angle = 90 - elems[0] * (360 / n)
            end_angle = 90 - elems[-1] * (360 / n)
            arc = Arc(
                (0, 0), 2 * radius, 2 * radius,
                angle=0, theta1=end_angle, theta2=start_angle,
                color=highlight_color, linewidth=5, alpha=0.25,
            )
            ax.add_patch(arc)

    # Dashed lines between spread elements
    if highlight and not draw_arc:
        elems = sorted(highlight)
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                x1, y1 = positions[elems[i]]
                x2, y2 = positions[elems[j]]
                ax.plot([x1, x2], [y1, y2], color=highlight_color,
                        linewidth=1.0, linestyle="--", alpha=0.4)

    # Nodes
    for i in range(n):
        x, y = positions[i]
        color = highlight_color if (highlight and i in highlight) else LIGHT_GRAY
        size = 160 if (highlight and i in highlight) else 80
        ax.scatter(x, y, c=color, s=size, zorder=5, edgecolors="none")
        lx = 1.25 * np.cos(np.pi / 2 - 2 * np.pi * i / n)
        ly = 1.25 * np.sin(np.pi / 2 - 2 * np.pi * i / n)
        ax.text(lx, ly, str(i), ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")


def fig2_intervals() -> None:
    """Two circle diagrams: cyclic interval vs spread non-basis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0))

    _draw_circle_nodes(
        ax1, n=6,
        highlight={2, 3, 4},
        highlight_color=ACCENT_GREEN,
        draw_arc=True,
        title=r"$\{2,3,4\}$: cyclic interval",
    )

    _draw_circle_nodes(
        ax2, n=6,
        highlight={0, 2, 4},
        highlight_color=ACCENT_RED,
        draw_arc=False,
        title=r"$\{0,2,4\}$: non-interval",
    )

    fig.tight_layout()

    path = OUTDIR / "fig-intervals.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: Counterexample dichotomy ─────────────────────────────────────


def fig3_dichotomy() -> None:
    """Bar chart of the 433/0/58/747 counterexample dichotomy."""
    categories = [
        "All non-bases are\ncyclic intervals",
        "Has $\\geq 1$ non-interval\nnon-basis",
    ]
    positroid_counts = [433, 58]
    non_positroid_counts = [0, 747]

    x = np.arange(len(categories))
    width = 0.30

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    bars_pos = ax.bar(x - width / 2, positroid_counts, width,
                      label="positroid", color=ACCENT_GREEN, alpha=0.85,
                      edgecolor="none")
    bars_neg = ax.bar(x + width / 2, non_positroid_counts, width,
                      label="non-positroid", color=ACCENT_RED, alpha=0.85,
                      edgecolor="none")

    # Value labels
    for bar in bars_pos:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 12,
                str(int(h)), ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=ACCENT_GREEN)
    for bar in bars_neg:
        h = bar.get_height()
        if h == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 12,
                    "0", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=BLACK)
            ax.annotate(
                "CIP theorem",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(bar.get_x() + bar.get_width() / 2 + 0.25, 200),
                fontsize=8, color=DARK_GRAY, fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=DARK_GRAY, lw=1.0),
            )
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 12,
                    str(int(h)), ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=ACCENT_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("number of matroids")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7, edgecolor=LIGHT_GRAY)
    ax.set_ylim(0, 870)
    ax.yaxis.grid(True, alpha=0.15, color=LIGHT_GRAY)
    _style_ax(ax)

    path = OUTDIR / "fig-dichotomy.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 4: Essential/tail partition + pruning curves ────────────────────


def fig4_pruning() -> None:
    """Two-panel figure: (a) essential vs tail hyperplanes, (b) pruning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # ── Panel (a): Essential vs tail hyperplane partition ──
    rng = np.random.default_rng(43)
    X, y = make_moons(n_samples=200, noise=0.1, rng=rng)

    config = TrainConfig(
        hidden_dim=20, learning_rate=0.01, epochs=100,
        param_mode="tp_exponential", seed=43,
    )
    net, _ = train(X, y, config)
    essential, tail, _rank = identify_essential_and_tail(net)
    essential_set = set(essential)

    W1 = net.layers[0].weight
    b1 = net.layers[0].bias
    H = W1.shape[0]

    xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    # Data points
    c0 = y == 0
    c1 = y == 1
    ax1.scatter(X[c0, 0], X[c0, 1], c=ACCENT_BLUE, s=6, alpha=0.3, zorder=3)
    ax1.scatter(X[c1, 0], X[c1, 1], c=ACCENT_ORANGE, s=6, alpha=0.3, zorder=3)

    # Hyperplanes colored by essential/tail
    xs = np.array([xlim[0], xlim[1]])
    for i in range(H):
        w0, w1_val = W1[i]
        b_val = b1[i]
        is_ess = i in essential_set

        if abs(w1_val) > 1e-10:
            ys = -(w0 * xs + b_val) / w1_val
            ax1.plot(
                xs, ys,
                color=ACCENT_GREEN if is_ess else ACCENT_RED,
                linewidth=1.5 if is_ess else 0.6,
                alpha=0.9 if is_ess else 0.25,
                zorder=4 if is_ess else 2,
            )
        else:
            xv = -b_val / w0 if abs(w0) > 1e-10 else 0
            ax1.axvline(
                xv,
                color=ACCENT_GREEN if is_ess else ACCENT_RED,
                linewidth=1.5 if is_ess else 0.6,
                alpha=0.9 if is_ess else 0.25,
                zorder=4 if is_ess else 2,
            )

    ax1.plot([], [], color=ACCENT_GREEN, linewidth=1.5,
             label=f"essential ({len(essential)})")
    ax1.plot([], [], color=ACCENT_RED, linewidth=0.6, alpha=0.4,
             label=f"tail ({len(tail)})")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.7,
               edgecolor=LIGHT_GRAY)
    ax1.set_title(f"(a) $H={H}$: {len(essential)} essential + {len(tail)} tail",
                  fontsize=9)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("$x_1$", fontsize=9)
    ax1.set_ylabel("$x_2$", fontsize=9)
    _style_ax(ax1)

    # ── Panel (b): Pruning curves at H=200 ──
    fractions = [0.0, 0.25, 0.50, 0.75, 1.00]
    matroid_deltas = [0.0, 0.0, 0.0, 0.0, -0.7]
    heuristic_deltas = [0.0, -0.3, -0.3, -0.3, -0.3]
    random_deltas = [0.0, -4.9, -11.9, -14.4, -20.9]

    ax2.plot(fractions, matroid_deltas, color=ACCENT_GREEN, linewidth=1.8,
             marker="o", markersize=5, label="matroid-guided", zorder=5)
    ax2.plot(fractions, heuristic_deltas, color=ACCENT_BLUE, linewidth=1.5,
             marker="D", markersize=4, label="mag/act/sens", zorder=4)
    ax2.plot(fractions, random_deltas, color=ACCENT_RED, linewidth=1.8,
             marker="s", markersize=5, label="random", zorder=4)

    ax2.fill_between(fractions, matroid_deltas, random_deltas,
                     color=ACCENT_GREEN, alpha=0.06)

    ax2.annotate(
        "14.4pp", xy=(0.75, -7.2), fontsize=8, color=ACCENT_GREEN,
        fontweight="bold", ha="center",
    )

    ax2.axhline(0, color=LIGHT_GRAY, linewidth=0.6, linestyle="--")
    ax2.set_xlabel("fraction of tail pruned", fontsize=9)
    ax2.set_ylabel("accuracy delta (%)", fontsize=9)
    ax2.set_title("(b) pruning curves at $H=200$", fontsize=9)
    ax2.legend(loc="lower left", fontsize=7, framealpha=0.7,
               edgecolor=LIGHT_GRAY)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-25, 3)
    ax2.set_xticks(fractions)
    ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax2.yaxis.grid(True, alpha=0.1, color=LIGHT_GRAY)
    _style_ax(ax2)

    fig.tight_layout(w_pad=2.0)

    path = OUTDIR / "fig-pruning.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")
    fig1_hyperplanes()
    fig2_intervals()
    fig3_dichotomy()
    fig4_pruning()
    print("Done!")


if __name__ == "__main__":
    main()
