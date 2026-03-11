"""Generate figures for the positroid blog post.

Saves PNGs to ~/gh/harrisontotty.github.io/images/.

Usage:
    uv run python scripts/generate_blog_figures.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

from positroid.datasets.toy2d import make_moons
from positroid.matroid.positroid import is_cyclic_interval, is_positroid
from positroid.network.train import TrainConfig, train

OUTDIR = Path.home() / "gh" / "harrisontotty.github.io" / "images"

# Blog palette (light theme — white bg, Roboto Mono, muted earth tones)
# Pulled from the syntax-highlight CSS in _layouts/post.html
BG = "#ffffff"
FG = "#1d1b1b"
FG_DIM = "#abafb6"
LINK_BLUE = "#60728d"     # links, strings
SAGE = "#608d7b"          # numbers, variables — use for "positive" / positroid
OLIVE = "#698d60"         # decorators, regex
PLUM = "#84608d"          # entities, symbols
WARM_BROWN = "#8d7260"    # comments
INDIAN_RED = "#cd5c5c"    # errors, functions — use for "negative" / non-positroid
DARK_BLUE = "#385074"     # classes, tags


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


# ── Figure 1: Hyperplane arrangement on two-moons ──


def fig1_hyperplanes() -> None:
    """Train a TP network on moons and plot hyperplanes."""
    rng = np.random.default_rng(42)
    X, y = make_moons(n_samples=300, noise=0.1, rng=rng)

    config = TrainConfig(
        hidden_dim=8,
        learning_rate=0.05,
        epochs=300,
        batch_size=32,
        tp_constrained=True,
        tp_kernel="exponential",
        seed=42,
    )
    network, _ = train(X, y, config)

    arrangement = network.hyperplane_arrangement(0)
    matroid = arrangement.affine_matroid()
    pos = is_positroid(matroid)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    c0 = y == 0
    c1 = y == 1
    ax.scatter(X[c0, 0], X[c0, 1], c=DARK_BLUE, s=12, alpha=0.5,
               zorder=3, label="class 0")
    ax.scatter(X[c1, 0], X[c1, 1], c=INDIAN_RED, s=12, alpha=0.5,
               zorder=3, label="class 1")

    # Plot hyperplane lines: w . x + b = 0 → x2 = -(w1*x1 + b) / w2
    xlims = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    xs = np.linspace(xlims[0], xlims[1], 200)

    for hp in arrangement.hyperplanes:
        w1, w2 = hp.normal
        b = hp.bias
        if abs(w2) > 1e-10:
            ys = -(w1 * xs + b) / w2
            mask = (ys > X[:, 1].min() - 1) & (ys < X[:, 1].max() + 1)
            ax.plot(xs[mask], ys[mask], color=WARM_BROWN, alpha=0.45,
                    linewidth=1.0, zorder=2)
        else:
            xv = -b / w1
            ax.axvline(xv, color=WARM_BROWN, alpha=0.45, linewidth=1.0,
                       zorder=2)

    ax.set_xlim(xlims)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    # Annotation
    status = "positroid" if pos else "non-positroid"
    n_bases = len(matroid.bases)
    total = len(list(combinations(range(matroid.size), matroid.rank)))
    label = f"rank {matroid.rank}, {n_bases}/{total} bases, {status}"
    ax.set_title(f"TP-weight ReLU network on two-moons\n{label}", fontsize=12)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.5,
              edgecolor=FG_DIM, facecolor=BG)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    path = OUTDIR / "positroid-fig1-hyperplanes.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: Cyclic interval vs spread diagram ──


def _draw_circle_nodes(
    ax: plt.Axes,
    n: int,
    radius: float = 1.0,
    highlight: set[int] | None = None,
    highlight_color: str = SAGE,
    draw_arc: bool = False,
    title: str = "",
) -> list[tuple[float, float]]:
    """Draw n nodes on a circle, optionally highlighting a subset."""
    positions = []
    for i in range(n):
        angle = np.pi / 2 - 2 * np.pi * i / n  # start at top, go clockwise
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append((x, y))

    # Draw the circle outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            color=FG_DIM, linewidth=0.5, alpha=0.4)

    # Draw highlighted arc if contiguous
    if draw_arc and highlight:
        elems = sorted(highlight)
        start_idx = elems[0]
        end_idx = elems[-1]
        gaps = [(elems[(i + 1) % len(elems)] - elems[i]) % n
                for i in range(len(elems))]
        if all(g == 1 for g in gaps[:-1]):
            start_angle = 90 - start_idx * (360 / n) + (360 / n) * 0.1
            end_angle = 90 - end_idx * (360 / n) - (360 / n) * 0.1
            arc = Arc(
                (0, 0), 2 * radius, 2 * radius,
                angle=0, theta1=end_angle, theta2=start_angle,
                color=highlight_color, linewidth=6, alpha=0.3,
            )
            ax.add_patch(arc)

    # Draw dashed lines between spread elements
    if highlight and not draw_arc:
        elems = sorted(highlight)
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                x1, y1 = positions[elems[i]]
                x2, y2 = positions[elems[j]]
                ax.plot([x1, x2], [y1, y2], color=highlight_color,
                        linewidth=1.5, linestyle="--", alpha=0.4)

    # Draw nodes
    for i in range(n):
        x, y = positions[i]
        color = highlight_color if (highlight and i in highlight) else FG_DIM
        size = 220 if (highlight and i in highlight) else 150
        ax.scatter(x, y, c=color, s=size, zorder=5, edgecolors="none")
        # Label
        lx = 1.22 * np.cos(np.pi / 2 - 2 * np.pi * i / n)
        ly = 1.22 * np.sin(np.pi / 2 - 2 * np.pi * i / n)
        ax.text(lx, ly, str(i), ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return positions


def fig2_intervals() -> None:
    """Two side-by-side circle diagrams: cyclic interval vs spread."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    _draw_circle_nodes(
        ax1, n=6,
        highlight={2, 3, 4},
        highlight_color=SAGE,
        draw_arc=True,
        title="{2,3,4}: cyclic interval\nalways positroid",
    )

    _draw_circle_nodes(
        ax2, n=6,
        highlight={0, 2, 4},
        highlight_color=INDIAN_RED,
        draw_arc=False,
        title="{0,2,4}: non-interval\ncan break positroid",
    )

    fig.suptitle("Non-basis patterns on [6]", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = OUTDIR / "positroid-fig2-intervals.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: Counterexample dichotomy chart ──


def fig3_dichotomy() -> None:
    """Bar chart showing the 433/0/58/747 dichotomy."""
    categories = [
        "All non-bases are\ncyclic intervals",
        "Has non-interval\nnon-basis",
    ]
    positroid_counts = [433, 58]
    non_positroid_counts = [0, 747]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width / 2, positroid_counts, width,
                   label="positroid", color=SAGE, alpha=0.85, edgecolor="none")
    bars2 = ax.bar(x + width / 2, non_positroid_counts, width,
                   label="non-positroid", color=INDIAN_RED, alpha=0.85,
                   edgecolor="none")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 15,
                str(int(h)), ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=SAGE)
    for bar in bars2:
        h = bar.get_height()
        if h == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 15,
                    "0", ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color=FG)
            ax.annotate(
                "theorem",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(bar.get_x() + bar.get_width() / 2 + 0.35, 120),
                fontsize=10, color=WARM_BROWN, fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=WARM_BROWN, lw=1.2),
            )
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 15,
                    str(int(h)), ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color=INDIAN_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("number of matroids", fontsize=11)
    ax.set_title(
        "1,238 non-uniform matroids (from 2,793 trials): non-basis structure vs positroid status",
        fontsize=12, pad=15,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.5,
              edgecolor=FG_DIM, facecolor=BG)

    ax.set_ylim(0, 870)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, alpha=0.15, color=FG_DIM)

    path = OUTDIR / "positroid-fig3-dichotomy.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 4: Single-removal dichotomy on U(3,6) ──


def fig4_corollary() -> None:
    """Show all 20 3-subsets of [6], colored by cyclic-interval status."""
    n, k = 6, 3
    subsets = list(combinations(range(n), k))

    fig, ax = plt.subplots(figsize=(9, 9))

    # Layout: place subsets on a circle (20 subsets)
    # Arc spacing = 2 * radius * sin(pi/20); must exceed 2 * (inner_r + pad)
    num = len(subsets)
    radius = 4.0
    inner_r = 0.40

    for idx, s in enumerate(subsets):
        angle = np.pi / 2 - 2 * np.pi * idx / num
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)

        is_interval = is_cyclic_interval(frozenset(s), n)
        color = SAGE if is_interval else INDIAN_RED
        bg_alpha = 0.10 if is_interval else 0.06

        # Background circle
        circle_bg = plt.Circle((cx, cy), inner_r + 0.06, color=color,
                                alpha=bg_alpha, zorder=1)
        ax.add_patch(circle_bg)

        # Draw mini-circle with 6 nodes
        for i in range(n):
            a = np.pi / 2 - 2 * np.pi * i / n
            nx = cx + inner_r * np.cos(a)
            ny = cy + inner_r * np.sin(a)

            if i in s:
                ax.scatter(nx, ny, c=color, s=22, zorder=4, edgecolors="none")
            else:
                ax.scatter(nx, ny, c=FG_DIM, s=8, zorder=3, alpha=0.3,
                           edgecolors="none")

        # Label
        label = "{" + ",".join(str(x) for x in s) + "}"
        lx = (radius + inner_r + 0.35) * np.cos(angle)
        ly = (radius + inner_r + 0.35) * np.sin(angle)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=6.5,
                color=color, fontfamily="monospace")

    # Legend — placed above the ring, below the title
    ax.scatter([], [], c=SAGE, s=40, label="cyclic interval (positroid)")
    ax.scatter([], [], c=INDIAN_RED, s=40,
               label="non-interval (non-positroid)")
    ax.legend(
        loc="upper center", fontsize=10, framealpha=0.5,
        edgecolor=FG_DIM, facecolor=BG, markerscale=2,
        bbox_to_anchor=(0.5, 0.97), ncol=2,
    )

    ax.set_title(
        "Single-removal dichotomy: $U(3,6) \\setminus \\{S\\}$\n"
        "6 intervals, 14 non-intervals",
        fontsize=12, pad=15,
    )
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-5.8, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")

    path = OUTDIR / "positroid-fig4-corollary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure: Independence vs dependence (companion post) ──


def fig_independence() -> None:
    """Three lines in general position vs three meeting at a point."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    xs = np.linspace(-0.5, 4.0, 300)

    # Line parameters: (slope, intercept)
    slopes = [1.5, -0.8, 0.15]
    colors_lines = [DARK_BLUE, PLUM, OLIVE]
    labels = ["line 0", "line 1", "line 2"]

    # ── Left panel: general position (triangle) ──
    intercepts_gp = [-1.5, 2.5, 0.2]

    for m, b, color, label in zip(slopes, intercepts_gp, colors_lines, labels):
        ys = m * xs + b
        mask = (ys > -1.5) & (ys < 3.5)
        ax1.plot(xs[mask], ys[mask], color=color, linewidth=2.0, label=label)

    # Mark pairwise intersections (the triangle vertices)
    for i in range(3):
        for j in range(i + 1, 3):
            m1, b1 = slopes[i], intercepts_gp[i]
            m2, b2 = slopes[j], intercepts_gp[j]
            xi = (b2 - b1) / (m1 - m2)
            yi = m1 * xi + b1
            ax1.scatter([xi], [yi], c=FG, s=35, zorder=5, edgecolors="none")

    ax1.set_title("general position\nevery triple is a basis", fontsize=11)

    # ── Right panel: coincidence (all through one point) ──
    meeting_point = (1.8, 0.85)
    intercepts_dep = [meeting_point[1] - m * meeting_point[0] for m in slopes]

    for m, b, color, label in zip(slopes, intercepts_dep, colors_lines, labels):
        ys = m * xs + b
        mask = (ys > -1.5) & (ys < 3.5)
        ax2.plot(xs[mask], ys[mask], color=color, linewidth=2.0, label=label)

    ax2.scatter(
        [meeting_point[0]], [meeting_point[1]], c=INDIAN_RED, s=90,
        zorder=5, marker="o", edgecolors="none",
    )

    ax2.set_title("coincidence\nthis triple is a non-basis", fontsize=11)

    # Style both panels
    for ax in (ax1, ax2):
        ax.legend(
            loc="upper right", fontsize=9, framealpha=0.5,
            edgecolor=FG_DIM, facecolor=BG,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(-0.5, 4.0)
        ax.set_ylim(-1.5, 3.5)

    fig.suptitle(
        "Independence vs dependence in a line arrangement",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = OUTDIR / "positroid-fig-independence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure: Bias shift creates dependency (companion post) ──


def fig_bias_shift() -> None:
    """Show how shifting a single bias creates a dependency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    xs = np.linspace(-0.5, 4.0, 300)

    slopes = [1.5, -0.8, 0.15]
    colors_lines = [DARK_BLUE, PLUM, OLIVE]
    labels = ["line 0", "line 1", "line 2"]

    # ── Left panel: general position (before) ──
    intercepts_before = [-1.5, 2.5, 0.2]

    for m, b, color, label in zip(slopes, intercepts_before, colors_lines, labels):
        ys = m * xs + b
        mask = (ys > -1.5) & (ys < 3.5)
        ax1.plot(xs[mask], ys[mask], color=color, linewidth=2.0, label=label)

    ax1.set_title("before: general position\n(all triples are bases)", fontsize=11)

    # ── Right panel: line 2 shifted so all three meet ──
    # Intersection of lines 0 and 1
    xi = (intercepts_before[1] - intercepts_before[0]) / (slopes[0] - slopes[1])
    yi = slopes[0] * xi + intercepts_before[0]

    # New intercept for line 2 to pass through (xi, yi)
    new_b2 = yi - slopes[2] * xi
    intercepts_after = [intercepts_before[0], intercepts_before[1], new_b2]

    # Draw lines 0 and 1 (unchanged)
    for i in [0, 1]:
        ys = slopes[i] * xs + intercepts_after[i]
        mask = (ys > -1.5) & (ys < 3.5)
        ax2.plot(
            xs[mask], ys[mask], color=colors_lines[i], linewidth=2.0,
            label=labels[i],
        )

    # Draw old line 2 (dashed, dimmed)
    ys_old = slopes[2] * xs + intercepts_before[2]
    mask = (ys_old > -1.5) & (ys_old < 3.5)
    ax2.plot(
        xs[mask], ys_old[mask], color=colors_lines[2], linewidth=1.5,
        linestyle="--", alpha=0.35, label="line 2 (before)",
    )

    # Draw new line 2 (solid)
    ys_new = slopes[2] * xs + intercepts_after[2]
    mask = (ys_new > -1.5) & (ys_new < 3.5)
    ax2.plot(
        xs[mask], ys_new[mask], color=colors_lines[2], linewidth=2.0,
        label="line 2 (shifted)",
    )

    # Arrow showing the shift
    arrow_x = 2.8
    arrow_y_old = slopes[2] * arrow_x + intercepts_before[2]
    arrow_y_new = slopes[2] * arrow_x + intercepts_after[2]
    ax2.annotate(
        "", xy=(arrow_x, arrow_y_new), xytext=(arrow_x, arrow_y_old),
        arrowprops={"arrowstyle": "->", "color": WARM_BROWN, "lw": 1.5},
    )
    ax2.text(
        arrow_x + 0.15, (arrow_y_old + arrow_y_new) / 2, "bias\nshift",
        fontsize=9, color=WARM_BROWN, fontstyle="italic", va="center",
    )

    # Mark meeting point
    ax2.scatter(
        [xi], [yi], c=INDIAN_RED, s=90, zorder=5, marker="o",
        edgecolors="none",
    )

    ax2.set_title(
        "after: one bias shifted\n(triple $\\{0,1,2\\}$ is now a non-basis)",
        fontsize=11,
    )

    # Style both panels
    for ax in (ax1, ax2):
        ax.legend(
            loc="upper right", fontsize=8, framealpha=0.5,
            edgecolor=FG_DIM, facecolor=BG,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(-0.5, 4.0)
        ax.set_ylim(-1.5, 3.5)

    fig.suptitle(
        "Weights set directions, biases set positions",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = OUTDIR / "positroid-fig-bias-shift.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ──

def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating blog figures...")
    fig1_hyperplanes()
    fig_independence()
    fig_bias_shift()
    fig2_intervals()
    fig3_dichotomy()
    fig4_corollary()
    print("Done!")


if __name__ == "__main__":
    main()
