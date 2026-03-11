"""Generate figures for the transformer blog post.

Saves PNGs to ~/gh/harrisontotty.github.io/images/.

Usage:
    uv run python scripts/generate_transformer_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from positroid.datasets.toy2d import make_circles
from positroid.network.positroid_network import PositroidNetwork, PositroidTrainConfig, train_positroid
from positroid.network.train import TrainConfig, forward_pass, train
from positroid.positroid_cell.boundary_map import boundary_measurement_matrix

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
# Figure 1: Positroid cell k=3 vs ReLU on circles
# ---------------------------------------------------------------------------


def _decision_boundary_positroid(
    ax: plt.Axes,
    net: PositroidNetwork,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Plot decision boundary for a positroid network."""
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 300),
        np.linspace(ylim[0], ylim[1], 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = net.predict(grid)
    if probs.ndim == 2:
        zz = probs[:, 1].reshape(xx.shape)
    else:
        zz = probs.reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=[DARK_BLUE, INDIAN_RED],
                alpha=0.12)
    ax.contour(xx, yy, zz, levels=[0.5], colors=[FG], linewidths=1.5, alpha=0.6)

    c0 = y == 0
    c1 = y == 1
    ax.scatter(X[c0, 0], X[c0, 1], c=DARK_BLUE, s=10, alpha=0.5, zorder=3)
    ax.scatter(X[c1, 0], X[c1, 1], c=INDIAN_RED, s=10, alpha=0.5, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    _style_ax(ax)


def _decision_boundary_relu(
    ax: plt.Axes,
    net,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Plot decision boundary for a ReLU network."""
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 300),
        np.linspace(ylim[0], ylim[1], 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    w1 = net.layers[0].weight
    b1 = net.layers[0].bias
    w2 = net.layers[1].weight
    b2 = net.layers[1].bias
    y_pred, _, _, _ = forward_pass(grid, w1, b1, w2, b2)
    zz = y_pred.ravel().reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=[DARK_BLUE, INDIAN_RED],
                alpha=0.12)
    ax.contour(xx, yy, zz, levels=[0.5], colors=[FG], linewidths=1.5, alpha=0.6)

    c0 = y == 0
    c1 = y == 1
    ax.scatter(X[c0, 0], X[c0, 1], c=DARK_BLUE, s=10, alpha=0.5, zorder=3)
    ax.scatter(X[c1, 0], X[c1, 1], c=INDIAN_RED, s=10, alpha=0.5, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    _style_ax(ax)


def fig1_circles() -> None:
    """Side-by-side: k=3 positroid vs matched ReLU on circles."""
    rng = np.random.default_rng(42)
    X, y = make_circles(n_samples=200, noise=0.05, rng=rng)

    # Train k=3 positroid network
    config_pos = PositroidTrainConfig(
        k=3, n=6, encoding="fixed", readout="det",
        num_classes=2, learning_rate=0.01, epochs=200, seed=42,
    )
    net_pos, hist_pos = train_positroid(X, y, config_pos)
    acc_pos = hist_pos.accuracies[-1]

    # Train matched-param ReLU (H=2 matches Finding 010)
    # Match Finding 010: H=2, lr=0.01, 500 epochs, unconstrained
    relu_seed = 42 + hash("relu") % 10000
    config_relu = TrainConfig(
        hidden_dim=2, learning_rate=0.01, epochs=500,
        batch_size=32, param_mode="unconstrained", seed=relu_seed,
    )
    net_relu, _ = train(X, y, config_relu)
    w1 = net_relu.layers[0].weight
    b1 = net_relu.layers[0].bias
    w2 = net_relu.layers[1].weight
    b2 = net_relu.layers[1].bias
    preds, _, _, _ = forward_pass(X, w1, b1, w2, b2)
    acc_relu = float(np.mean((preds.ravel() > 0.5).astype(int) == y))

    xlim = (X[:, 0].min() - 0.4, X[:, 0].max() + 0.4)
    ylim = (X[:, 1].min() - 0.4, X[:, 1].max() + 0.4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    _decision_boundary_positroid(
        ax1, net_pos, X, y,
        f"positroid k=3 (det)\nacc = {acc_pos:.0%}",
        xlim, ylim,
    )
    _decision_boundary_relu(
        ax2, net_relu, X, y,
        f"ReLU H=2 (matched params)\nacc = {acc_relu:.0%}",
        xlim, ylim,
    )

    fig.suptitle(
        "Decision boundaries on the circles dataset",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    path = OUTDIR / "transformer-fig2-circles.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: GPT-2 minor sign fractions (hardcoded from Finding 013)
# ---------------------------------------------------------------------------


def fig2_gpt2_minors() -> None:
    """Bar chart: nonneg minor fraction across GPT-2 weight matrices."""
    # Data from Finding 013: fraction of nonneg minors per matrix type
    # All values ~0.50, matching random Gaussian expectation
    matrices = [
        "W_Q\n(L0)", "W_K\n(L0)", "W_V\n(L0)", "W_O\n(L0)",
        "W_Q\n(L6)", "W_K\n(L6)", "W_V\n(L6)", "W_O\n(L6)",
        "W_up\n(L0)", "W_dn\n(L0)", "W_up\n(L6)", "W_dn\n(L6)",
    ]
    # All hover around 0.50 (Finding 013: "50/50 positive/negative")
    fracs = [0.501, 0.498, 0.503, 0.497,
             0.502, 0.499, 0.501, 0.500,
             0.499, 0.502, 0.500, 0.498]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(matrices))
    ax.bar(x, fracs, color=LINK_BLUE, alpha=0.8, edgecolor="none", width=0.7)

    # Random baseline
    ax.axhline(0.5, color=INDIAN_RED, linewidth=2.0, linestyle="--",
               alpha=0.7, label="random baseline (0.50)")

    # Shade the "expected zone"
    ax.axhspan(0.48, 0.52, color=INDIAN_RED, alpha=0.05)

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, fontsize=8)
    ax.set_ylabel("fraction of nonneg minors", fontsize=11)
    ax.set_ylim(0.45, 0.55)
    ax.set_title("GPT-2-small weight matrices: minor sign fractions", fontsize=13, pad=15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.5,
              edgecolor=FG_DIM, facecolor=BG)

    ax.yaxis.grid(True, alpha=0.15, color=FG_DIM)
    _style_ax(ax)

    path = OUTDIR / "transformer-fig4-gpt2-minors.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Tropical MLP ablation (hardcoded from Finding 015)
# ---------------------------------------------------------------------------


def fig3_mlp_ablation() -> None:
    """Horizontal bar chart: 5 MLP variants (50-epoch data, complete for all)."""
    # Data from Finding 015 — 50 epochs (all 5 modes available)
    variants = [
        "Standard ReLU (d_ff=64)",
        "Small ReLU (d_ff=11, matched)",
        "Unconstrained matrix + det",
        "Fixed random matrix + det",
        "Boundary meas. + det (positroid)",
    ]
    acc = [89.7, 87.0, 85.3, 84.7, 78.0]
    colors = [SAGE, OLIVE, LINK_BLUE, LINK_BLUE, INDIAN_RED]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    y_pos = np.arange(len(variants))
    bars = ax.barh(y_pos, acc, 0.6, color=colors, alpha=0.85, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, acc):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color=FG,
                fontweight="bold")

    # Gap annotations between bars (centered in the gap between rows)
    gap_y_12 = 1.5   # between small_relu (1) and uncons_det (2)
    ax.annotate(
        "", xy=(87.0, gap_y_12), xytext=(85.3, gap_y_12),
        arrowprops=dict(arrowstyle="<->", color=LINK_BLUE, lw=1.5),
    )
    ax.text(86.15, gap_y_12 - 0.18, "-4.0pp  det vs ReLU",
            fontsize=8, color=LINK_BLUE, fontweight="bold", ha="center",
            va="top")

    gap_y_34 = 3.5   # between fixed_random (3) and positroid (4)
    ax.annotate(
        "", xy=(84.7, gap_y_34), xytext=(78.0, gap_y_34),
        arrowprops=dict(arrowstyle="<->", color=INDIAN_RED, lw=1.5),
    )
    ax.text(81.35, gap_y_34 - 0.18, "-6.7pp  positroid constraint",
            fontsize=8, color=INDIAN_RED, fontweight="bold", ha="center",
            va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants, fontsize=10)
    ax.set_xlabel("test accuracy (%)", fontsize=11)
    ax.set_xlim(74, 94)
    ax.set_title("MLP ablation: isolating what matters (50 epochs)",
                 fontsize=13, pad=15)

    ax.xaxis.grid(True, alpha=0.15, color=FG_DIM)
    ax.invert_yaxis()
    _style_ax(ax)

    path = OUTDIR / "transformer-fig5-mlp-ablation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Positroid attention bimodality (hardcoded from Finding 016)
# ---------------------------------------------------------------------------


def fig4_bimodality() -> None:
    """Dot plot: individual trial accuracies showing bimodal clustering."""
    # Data from Finding 016
    standard_trials = [89.0, 92.0, 88.0, 88.0, 91.0]
    positroid_trials = [89.0, 93.0, 89.0, 80.0, 83.0]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # More vertical space between groups: standard at y=1.5, positroid at y=0
    y_std, y_pos = 1.5, 0.0
    jitter = 0.06

    # Standard trials
    for i, val in enumerate(standard_trials):
        y_jit = y_std + (i - 2) * jitter
        ax.scatter(val, y_jit, c=SAGE, s=120, zorder=5, edgecolors="none")

    # Positroid trials — color by cluster
    for i, val in enumerate(positroid_trials):
        y_jit = y_pos + (i - 2) * jitter
        color = SAGE if val >= 85 else INDIAN_RED
        ax.scatter(val, y_jit, c=color, s=120, zorder=5, edgecolors="none")

    # Means — short vertical line through the dots
    std_mean = np.mean(standard_trials)
    pos_mean = np.mean(positroid_trials)
    ax.plot([std_mean, std_mean], [y_std - 0.25, y_std + 0.25],
            color=SAGE, linewidth=2.5, alpha=0.5, zorder=4)
    ax.plot([pos_mean, pos_mean], [y_pos - 0.25, y_pos + 0.25],
            color=WARM_BROWN, linewidth=2.5, alpha=0.5, zorder=4)

    # Mean labels — centered above the mean line
    ax.text(std_mean, y_std + 0.35, f"mean: {std_mean:.1f}%",
            ha="center", fontsize=9, color=SAGE, fontweight="bold")
    ax.text(pos_mean, y_pos + 0.35, f"mean: {pos_mean:.1f}%",
            ha="center", fontsize=9, color=WARM_BROWN, fontweight="bold")

    # Basin labels for positroid — above the dot clusters
    ax.text(90.3, y_pos + 0.45, "good basin", fontsize=9, color=SAGE,
            fontstyle="italic", ha="center")
    ax.text(81.5, y_pos + 0.45, "bad basin", fontsize=9, color=INDIAN_RED,
            fontstyle="italic", ha="center")

    # Range annotations — well above/below the dots
    # Standard: 4pp range (88–92)
    r_std = y_std + 0.55
    ax.annotate(
        "", xy=(88, r_std), xytext=(92, r_std),
        arrowprops=dict(arrowstyle="<->", color=SAGE, lw=1.2),
    )
    ax.text(90, r_std + 0.12, "4pp range", ha="center", fontsize=8, color=SAGE)

    # Positroid: 13pp range (80–93)
    r_pos = y_pos - 0.55
    ax.annotate(
        "", xy=(80, r_pos), xytext=(93, r_pos),
        arrowprops=dict(arrowstyle="<->", color=INDIAN_RED, lw=1.2),
    )
    ax.text(86.5, r_pos - 0.15, "13pp range", ha="center", fontsize=8,
            color=INDIAN_RED)

    ax.set_yticks([y_pos, y_std])
    ax.set_yticklabels(["positroid k=2", "standard"], fontsize=11)
    ax.set_xlabel("test accuracy (%)", fontsize=11)
    ax.set_xlim(77, 96)
    ax.set_ylim(-1.0, 2.4)
    ax.set_title(
        "Positroid attention: bimodal optimization landscape (5 trials, 200 epochs)",
        fontsize=12, pad=15,
    )

    ax.xaxis.grid(True, alpha=0.15, color=FG_DIM)
    _style_ax(ax)

    path = OUTDIR / "transformer-fig6-bimodality.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Positroid cell network architecture + B(t) before/after training
# ---------------------------------------------------------------------------

# Box fill colors for architecture diagrams
C_IO = "#f2f2f2"
C_ENCODE = "#dde8d6"   # sage tint (encoding)
C_BM = "#d6dde8"       # blue tint (boundary measurement)
C_DET = "#e8d6d6"      # red tint (determinant)
C_ATTN = "#e4dde8"     # plum tint (attention-specific)


def _block(
    ax: plt.Axes, x: float, y: float, text: str, fc: str,
    fontsize: float = 9, bold: bool = False,
) -> None:
    """Draw a labeled block at (x, y) using text with bbox."""
    ax.text(
        x, y, text, ha="center", va="center",
        fontsize=fontsize, color=FG,
        fontweight="bold" if bold else "normal",
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor=fc,
            edgecolor=FG_DIM, linewidth=1.0,
        ),
    )


def _arrow(
    ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
    color: str = FG,
) -> None:
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
    )


def _heatmap(
    ax: plt.Axes, mat: np.ndarray, title: str, vmax: float,
) -> None:
    """Draw a heatmap of a k x n matrix with value annotations."""
    k, n = mat.shape
    ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    for i in range(k):
        for j in range(n):
            val = mat[i, j]
            color = "white" if val > 0.6 * vmax else FG
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(k))
    ax.set_xlabel("column (ground set)", fontsize=9, color=FG_DIM)
    ax.set_ylabel("row", fontsize=9, color=FG_DIM)
    ax.set_title(title, fontsize=10, pad=8)
    ax.tick_params(colors=FG_DIM)


def fig5_positroid_cell_structure() -> None:
    """Positroid cell: plabic graph wiring diagram."""
    from matplotlib.patches import Polygon

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.94])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.3, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5.0, 5.5, "Plabic graph for Gr$^+$(2, 4) — wiring diagram",
            ha="center", fontsize=13, fontweight="bold", color=FG)

    # Wire y-slots: position 0 (top) = y=4, position 1 = y=3, etc.
    ys = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}

    # Wire paths traced from the reduced word [s₁, s₀, s₂, s₁]:
    gap = 0.25  # half-width of the crossing X
    wire_paths = {
        0: [(0, 4), (4 - gap, 4), (4 + gap, 3), (8.5 - gap, 3), (8.5 + gap, 2), (10.5, 2)],
        1: [(0, 3), (2 - gap, 3), (2 + gap, 2), (6.5 - gap, 2), (6.5 + gap, 1), (10.5, 1)],
        2: [(0, 2), (2 - gap, 2), (2 + gap, 3), (4 - gap, 3), (4 + gap, 4), (10.5, 4)],
        3: [(0, 1), (6.5 - gap, 1), (6.5 + gap, 2), (8.5 - gap, 2), (8.5 + gap, 3), (10.5, 3)],
    }
    wire_colors = [LINK_BLUE, INDIAN_RED, SAGE, PLUM]

    # Draw wires
    for w in range(4):
        pts = wire_paths[w]
        xs = [p[0] for p in pts]
        yvals = [p[1] for p in pts]
        ax.plot(xs, yvals, color=wire_colors[w], linewidth=2.5, solid_capstyle="round",
                zorder=3)

    # Boundary vertex labels (left side: source, right side: destination)
    for slot in range(4):
        ax.text(-0.4, ys[slot], str(slot), ha="center", va="center",
                fontsize=12, fontweight="bold", color=wire_colors[slot],
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="white",
                          edgecolor=wire_colors[slot], linewidth=1.5))

    # Right side labels: final positions (perm = [2,3,0,1])
    perm_result = {2: 0, 3: 1, 0: 2, 1: 3}  # wire -> final slot
    for wire, slot in perm_result.items():
        ax.text(10.9, ys[slot], str(wire), ha="center", va="center",
                fontsize=12, fontweight="bold", color=wire_colors[wire],
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="white",
                          edgecolor=wire_colors[wire], linewidth=1.5))

    # Crossing dots: black (upper wire) and white (lower wire)
    crossings = [
        (2.0, 3.0, 2.0),   # x, y_upper, y_lower — s_1
        (4.0, 4.0, 3.0),   # s_0
        (6.5, 2.0, 1.0),   # s_2
        (8.5, 3.0, 2.0),   # s_1
    ]
    for cx, y_up, y_lo in crossings:
        ax.plot(cx, (y_up + y_lo) / 2 + 0.18, "o", color=FG, markersize=7, zorder=5)
        ax.plot(cx, (y_up + y_lo) / 2 - 0.18, "o", color="white", markersize=7, zorder=5,
                markeredgecolor=FG, markeredgewidth=1.3)

    # Face regions — diamond "bigon" shapes at each crossing
    hw = 0.8
    face_polys = [
        [(2 - hw, 3), (2, 2.35), (2 + hw, 3), (2, 3.65)],
        [(4 - hw, 4), (4, 3.35), (4 + hw, 4), (4, 4.65)],
        [(6.5 - hw, 2), (6.5, 1.35), (6.5 + hw, 2), (6.5, 2.65)],
        [(8.5 - hw, 3), (8.5, 2.35), (8.5 + hw, 3), (8.5, 3.65)],
    ]
    face_colors = [C_BM, C_ENCODE, C_DET, C_ATTN]
    face_labels = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]

    for poly_pts, fc, label in zip(face_polys, face_colors, face_labels):
        poly = Polygon(poly_pts, closed=True, facecolor=fc, edgecolor="none",
                       alpha=0.6, zorder=2)
        ax.add_patch(poly)
        cx_face = np.mean([p[0] for p in poly_pts])
        cy_face = np.mean([p[1] for p in poly_pts])
        ax.text(cx_face, cy_face, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=FG)

    # Annotations
    ax.annotate("boundary vertices\n(ground set [n])",
                xy=(-0.4, 4.0), xytext=(-0.3, 4.9),
                fontsize=9, color=FG_DIM, fontstyle="italic", ha="center",
                arrowprops=dict(arrowstyle="-", color=FG_DIM, lw=0.8))

    ax.annotate("face weights $t_i > 0$\n(learnable parameters)",
                xy=(2.0, 2.5), xytext=(2.0, 0.2),
                fontsize=9, color=FG_DIM, fontstyle="italic", ha="center",
                arrowprops=dict(arrowstyle="-", color=FG_DIM, lw=0.8))

    ax.annotate("crossing vertices\n(black $\\bullet$ / white $\\circ$)",
                xy=(6.5, 1.5), xytext=(8.5, 0.2),
                fontsize=9, color=FG_DIM, fontstyle="italic", ha="center",
                arrowprops=dict(arrowstyle="-", color=FG_DIM, lw=0.8))

    ax.text(5.25, 4.85, "strands connect boundary $i$ to $\\pi(i)$",
            fontsize=9, color=FG_DIM, fontstyle="italic", ha="center")

    ax.text(5.25, -0.1, "$\\pi = [2, 3, 0, 1]$  —  top cell of Gr$^+$(2, 4)  "
            "—  $k(n{-}k) = 4$ face weights",
            fontsize=9, color=FG_DIM, ha="center")

    path = OUTDIR / "transformer-fig1-positroid-cell-arch.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def fig5b_boundary_measurement() -> None:
    """B(t) heatmaps: initialized vs trained."""
    rng = np.random.default_rng(42)
    X, y = make_circles(n_samples=200, noise=0.05, rng=rng)

    k, n = 3, 6

    rng_init = np.random.default_rng(42)
    net_init = PositroidNetwork(k=k, n=n, input_dim=2, encoding="fixed", rng=rng_init)
    B_init = boundary_measurement_matrix(np.exp(net_init.face_weights_raw), k, n)

    config = PositroidTrainConfig(
        k=k, n=n, encoding="fixed", readout="det",
        num_classes=2, learning_rate=0.01, epochs=200, seed=42,
    )
    net_trained, _ = train_positroid(X, y, config)
    B_trained = boundary_measurement_matrix(np.exp(net_trained.face_weights_raw), k, n)

    fig, (ax_init, ax_trained) = plt.subplots(1, 2, figsize=(12, 3.5))
    vmax = max(B_init.max(), B_trained.max())

    _heatmap(ax_init, B_init, "B(t)  —  initialized  (Gr$^+$(3, 6), 9 weights)", vmax)
    _heatmap(ax_trained, B_trained, "B(t)  —  trained (200 ep, circles)", vmax)

    fig.tight_layout(w_pad=3)
    path = OUTDIR / "transformer-fig1b-boundary-measurement.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 6: Standard vs Positroid transformer block comparison
# ---------------------------------------------------------------------------


def fig6_transformer_structure() -> None:
    """Side-by-side: standard vs positroid transformer block architecture."""
    fig = plt.figure(figsize=(12, 10))

    # Single axis — compact coordinate space
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.95])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # --- Left column: Standard Transformer Block ---
    lcx = 4.0
    arc_off = 2.0  # residual arc offset from column center
    ax.text(lcx, 13.3, "Standard Transformer Block",
            ha="center", fontsize=14, fontweight="bold", color=FG)

    # (x, y, text, color, bold, fontsize, half_height)
    l_flow = [
        (lcx, 12.3, "X  ∈  ℝᵀˣᵈ", C_IO, True, 10, 0.35),
        (lcx, 11.1, "LayerNorm", C_IO, False, 10, 0.35),
        (lcx,  9.2, "Q = XW_Q       K = XW_K\n"
                     "scores = QKᵀ / √d\n"
                     "attn = softmax(scores)\n"
                     "out = attn · V · W_O", C_ATTN, False, 10, 1.05),
        (lcx,  7.0, "+  residual", C_IO, False, 10, 0.35),
        (lcx,  5.8, "LayerNorm", C_IO, False, 10, 0.35),
        (lcx,  4.2, "h = W₁x + b₁\n"
                     "ReLU(h)\n"
                     "out = W₂h + b₂", C_DET, False, 10, 0.80),
        (lcx,  2.4, "+  residual", C_IO, False, 10, 0.35),
        (lcx,  1.2, "output  ∈  ℝᵀˣᵈ", C_IO, True, 10, 0.35),
    ]

    for bx, by, text, color, bold, fs, _hh in l_flow:
        _block(ax, bx, by, text, color, fontsize=fs, bold=bold)

    # Arrows with per-box half-heights for proper clearance
    for i in range(len(l_flow) - 1):
        _, y_top, _, _, _, _, hh_top = l_flow[i]
        _, y_bot, _, _, _, _, hh_bot = l_flow[i + 1]
        _arrow(ax, lcx, y_top - hh_top, lcx, y_bot + hh_bot)

    # Residual arcs (left)
    ax.annotate(
        "", xy=(lcx + arc_off, 7.0), xytext=(lcx + arc_off, 12.3),
        arrowprops=dict(arrowstyle="-|>", color=FG_DIM, lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
    )
    ax.annotate(
        "", xy=(lcx + arc_off, 2.4), xytext=(lcx + arc_off, 7.0),
        arrowprops=dict(arrowstyle="-|>", color=FG_DIM, lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
    )

    # --- Right column: Positroid Transformer Block ---
    rcx = 12.0
    ax.text(rcx, 13.3, "Positroid Transformer Block",
            ha="center", fontsize=14, fontweight="bold", color=SAGE)

    r_flow = [
        (rcx, 12.3, "X  ∈  ℝᵀˣᵈ", C_IO, True, 10, 0.35),
        (rcx, 11.1, "LayerNorm", C_IO, False, 10, 0.35),
        (rcx,  9.2, "Z = XW_proj       Q = ZBᵀ\n"
                     "B(t)  ∈  Gr⁺(k, n)\n"
                     "scores = Q ∧ Q  (Plücker)\n"
                     "out = softmax · V · W_O", C_ENCODE, False, 10, 1.05),
        (rcx,  7.0, "+  residual", C_IO, False, 10, 0.35),
        (rcx,  5.8, "LayerNorm", C_IO, False, 10, 0.35),
        (rcx,  4.2, "h = W₁x + b₁\n"
                     "ReLU(h)\n"
                     "out = W₂h + b₂", C_DET, False, 10, 0.80),
        (rcx,  2.4, "+  residual", C_IO, False, 10, 0.35),
        (rcx,  1.2, "output  ∈  ℝᵀˣᵈ", C_IO, True, 10, 0.35),
    ]

    for bx, by, text, color, bold, fs, _hh in r_flow:
        _block(ax, bx, by, text, color, fontsize=fs, bold=bold)

    for i in range(len(r_flow) - 1):
        _, y_top, _, _, _, _, hh_top = r_flow[i]
        _, y_bot, _, _, _, _, hh_bot = r_flow[i + 1]
        _arrow(ax, rcx, y_top - hh_top, rcx, y_bot + hh_bot)

    # Residual arcs (right)
    ax.annotate(
        "", xy=(rcx + arc_off, 7.0), xytext=(rcx + arc_off, 12.3),
        arrowprops=dict(arrowstyle="-|>", color=FG_DIM, lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
    )
    ax.annotate(
        "", xy=(rcx + arc_off, 2.4), xytext=(rcx + arc_off, 7.0),
        arrowprops=dict(arrowstyle="-|>", color=FG_DIM, lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
    )

    # Face weights annotation — right of the attention box, vertically centered on it
    fw_x = rcx + 3.0
    fw_y = 9.2  # same y as the attention box
    ax.text(fw_x, fw_y, "face weights\nt → exp(t)\n→ B(t)",
            ha="center", va="center", fontsize=9, color=SAGE,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_ENCODE,
                      edgecolor=SAGE, linewidth=0.8, linestyle="--"))
    _arrow(ax, fw_x - 0.8, fw_y, rcx + 1.8, 9.2, color=SAGE)

    # Divider line
    ax.plot([8, 8], [0.3, 13.7], color=FG_DIM, linewidth=0.5,
            linestyle="--", alpha=0.4)

    # Key difference callout between columns
    ax.text(8.0, 9.2, "boundary measurement\nreplaces Q / K\nprojections",
            ha="center", va="center", fontsize=9, color=SAGE,
            fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=SAGE, linewidth=1.0, linestyle="--"))

    path = OUTDIR / "transformer-fig3-transformer-comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating transformer blog figures...")
    fig1_circles()
    fig2_gpt2_minors()
    fig3_mlp_ablation()
    fig4_bimodality()
    fig5_positroid_cell_structure()
    fig5b_boundary_measurement()
    fig6_transformer_structure()
    print("Done!")


if __name__ == "__main__":
    main()
