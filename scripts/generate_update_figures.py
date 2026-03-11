"""Generate figures for the non-TP baseline update section.

Saves PNGs to ~/gh/harrisontotty.github.io/images/.

Usage:
    python scripts/generate_update_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch, FancyArrowPatch

OUTDIR = Path.home() / "gh" / "harrisontotty.github.io" / "images"

# Blog palette (matches generate_blog_figures.py exactly)
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


# ── Figure 5: Non-base support patterns (contiguous vs gapped) ──


def _draw_support_circle(
    ax: plt.Axes,
    n: int,
    support: set[int],
    non_bases: list[tuple[int, ...]],
    color: str,
    title: str,
    radius: float = 1.0,
) -> None:
    """Draw n nodes on a circle with support elements highlighted.

    Support elements are large and colored; non-support are small and dim.
    Arcs connect elements within each non-basis.
    """
    positions = []
    for i in range(n):
        angle = np.pi / 2 - 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append((x, y))

    # Draw the circle outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            color=FG_DIM, linewidth=0.5, alpha=0.4)

    # Highlight contiguous arc if support is contiguous
    sorted_support = sorted(support)
    is_contiguous = _is_contiguous_on_circle(sorted_support, n)

    if is_contiguous and len(support) > 1:
        # Find the start/end for the arc
        start_idx, end_idx = _find_arc_endpoints(sorted_support, n)
        start_angle = 90 - start_idx * (360 / n) + (360 / n) * 0.1
        end_angle = 90 - end_idx * (360 / n) - (360 / n) * 0.1
        arc = Arc(
            (0, 0), 2 * radius, 2 * radius,
            angle=0, theta1=end_angle, theta2=start_angle,
            color=color, linewidth=6, alpha=0.25,
        )
        ax.add_patch(arc)
    elif not is_contiguous and len(support) > 1:
        # Draw dashed lines between support elements to show spread
        sorted_s = sorted(support)
        for i in range(len(sorted_s)):
            for j in range(i + 1, len(sorted_s)):
                x1, y1 = positions[sorted_s[i]]
                x2, y2 = positions[sorted_s[j]]
                ax.plot([x1, x2], [y1, y2], color=color,
                        linewidth=1.0, linestyle="--", alpha=0.2)

    # Draw nodes
    for i in range(n):
        x, y = positions[i]
        if i in support:
            node_color = color
            size = 220
        else:
            node_color = FG_DIM
            size = 100
        ax.scatter(x, y, c=node_color, s=size, zorder=5, edgecolors="none")

        # Label
        lx = 1.25 * np.cos(np.pi / 2 - 2 * np.pi * i / n)
        ly = 1.25 * np.sin(np.pi / 2 - 2 * np.pi * i / n)
        label_color = color if i in support else FG_DIM
        fontweight = "bold" if i in support else "normal"
        ax.text(lx, ly, str(i), ha="center", va="center",
                fontsize=12, fontweight=fontweight, color=label_color)

    ax.set_title(title, fontsize=10, pad=12)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def _is_contiguous_on_circle(elements: list[int], n: int) -> bool:
    """Check if elements form a contiguous arc on [n]."""
    if len(elements) <= 1:
        return True
    s = sorted(elements)
    # Try each element as the "start" of the arc
    for start in s:
        rotated = sorted(((e - start) % n) for e in s)
        if rotated == list(range(len(s))):
            return True
    return False


def _find_arc_endpoints(elements: list[int], n: int) -> tuple[int, int]:
    """Find start/end indices of a contiguous arc on the circle."""
    s = sorted(elements)
    # Find the gap (largest step between consecutive elements mod n)
    gaps = [((s[(i + 1) % len(s)] - s[i]) % n) for i in range(len(s))]
    max_gap_idx = int(np.argmax(gaps))
    start = s[(max_gap_idx + 1) % len(s)]
    end = s[max_gap_idx]
    return start, end


def fig5_support_patterns() -> None:
    """Three circle diagrams: TP contiguous tail vs two gapped supports."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: TP exponential — contiguous tail on [10]
    # Typical pattern: non-bases = all C(4,3) subsets of {6,7,8,9}
    _draw_support_circle(
        ax1, n=10,
        support={6, 7, 8, 9},
        non_bases=[],  # not drawn individually
        color=SAGE,
        title="TP exponential (H=10)\nsupport: {6,7,8,9}\ncontiguous — positroid",
    )

    # Panel 2: Negated bidiagonal — gapped support on [6]
    # Non-bases: {1,3,4}, {1,3,5}, {1,4,5}, {3,4,5}
    # Support: {1,3,4,5} — gap at 0 and 2
    _draw_support_circle(
        ax2, n=6,
        support={1, 3, 4, 5},
        non_bases=[(1, 3, 4), (1, 3, 5), (1, 4, 5), (3, 4, 5)],
        color=INDIAN_RED,
        title="negated bidiagonal (H=6)\nsupport: {1,3,4,5}\ngapped — non-positroid",
    )

    # Panel 3: Negated bidiagonal — gapped support on [10]
    # Support: {3,5,6,7,8,9} — gap at 4
    _draw_support_circle(
        ax3, n=10,
        support={3, 5, 6, 7, 8, 9},
        non_bases=[],
        color=INDIAN_RED,
        title="negated bidiagonal (H=10)\nsupport: {3,5,6,7,8,9}\ngapped — non-positroid",
    )

    fig.suptitle(
        "Non-base support: contiguous (TP) vs gapped (non-TP)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = OUTDIR / "positroid-fig5-support.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 6: TP vs negated bidiagonal comparison ──


def fig6_baseline_comparison() -> None:
    """Bar chart comparing non-positroid rates: TP vs negated bidiagonal."""
    modes = ["TP exponential", "negated bidiagonal"]
    positroid = [60, 58]
    non_positroid = [0, 2]

    x = np.arange(len(modes))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width / 2, positroid, width,
                   label="positroid", color=SAGE, alpha=0.85, edgecolor="none")
    bars2 = ax.bar(x + width / 2, non_positroid, width,
                   label="non-positroid", color=INDIAN_RED, alpha=0.85,
                   edgecolor="none")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                str(int(h)), ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=SAGE)

    for bar in bars2:
        h = bar.get_height()
        if h == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                    "0", ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color=FG)
            # Annotation arrow pointing to the zero
            ax.annotate(
                "always\npositroid",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(bar.get_x() + bar.get_width() / 2 + 0.35, 18),
                fontsize=10, color=WARM_BROWN, fontstyle="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=WARM_BROWN, lw=1.2),
            )
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                    str(int(h)), ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color=INDIAN_RED)
            ax.annotate(
                "~3%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(bar.get_x() + bar.get_width() / 2 + 0.35, 18),
                fontsize=10, color=WARM_BROWN, fontstyle="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=WARM_BROWN, lw=1.2),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_ylabel("number of matroids", fontsize=11)
    ax.set_title(
        "60 moons training trials per mode\n"
        "non-positroid rate: TP = 0%, negated bidiagonal = ~3%",
        fontsize=12, pad=15,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.5,
              edgecolor=FG_DIM, facecolor=BG)

    ax.set_ylim(0, 75)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, alpha=0.15, color=FG_DIM)

    path = OUTDIR / "positroid-fig6-baseline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 7: Causal chain diagram ──


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: int = 11,
) -> tuple[float, float]:
    """Draw a rounded box with centered text. Returns center position."""
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.15",
        facecolor=color, alpha=0.12, edgecolor=color,
        linewidth=1.5, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=FG, fontweight="bold", zorder=3)
    return (cx, cy)


def fig7_causal_chain() -> None:
    """Flow diagram: TP → contiguous → positroid, non-TP → gapped → non-positroid."""
    fig, ax = plt.subplots(figsize=(12, 5))

    bw, bh = 2.4, 0.7  # box width, height

    # Top row: TP path (green)
    tp_box = _rounded_box(ax, (1.5, 3.0), bw, bh, "TP weights", SAGE)
    cont_box = _rounded_box(ax, (5.5, 3.0), bw, bh,
                            "contiguous\nnon-base support", SAGE)
    pos_box = _rounded_box(ax, (9.5, 3.0), bw, bh, "positroid", SAGE)

    # Bottom row: non-TP path (red)
    ntp_box = _rounded_box(ax, (1.5, 1.0), bw, bh, "non-TP weights", INDIAN_RED)
    gap_box = _rounded_box(ax, (5.5, 1.0), bw, bh,
                           "gapped\nnon-base support", INDIAN_RED)
    npos_box = _rounded_box(ax, (9.5, 1.0), bw, bh, "non-positroid", INDIAN_RED)

    # Arrows: top row
    # FancyBboxPatch pad=0.15 extends box beyond bw/2, so offset by pad + gap
    pad = 0.15
    gap = 0.12
    edge = bw / 2 + pad + gap
    arrow_kw = dict(
        arrowstyle="->,head_width=0.25,head_length=0.15",
        linewidth=2.0, zorder=4,
    )
    ax.annotate("", xy=(5.5 - edge, 3.0),
                xytext=(1.5 + edge, 3.0),
                arrowprops=dict(**arrow_kw, color=SAGE))
    ax.annotate("", xy=(9.5 - edge, 3.0),
                xytext=(5.5 + edge, 3.0),
                arrowprops=dict(**arrow_kw, color=SAGE))

    # Arrows: bottom row
    ax.annotate("", xy=(5.5 - edge, 1.0),
                xytext=(1.5 + edge, 1.0),
                arrowprops=dict(**arrow_kw, color=INDIAN_RED))
    ax.annotate("", xy=(9.5 - edge, 1.0),
                xytext=(5.5 + edge, 1.0),
                arrowprops=dict(**arrow_kw, color=INDIAN_RED))

    # Arrow labels
    ax.text(3.5, 3.35, "training\ndynamics", ha="center", va="bottom",
            fontsize=9, color=WARM_BROWN, fontstyle="italic")
    ax.text(7.5, 3.35, "theorem", ha="center", va="bottom",
            fontsize=9, color=WARM_BROWN, fontstyle="italic")
    ax.text(3.5, 0.55, "training\ndynamics", ha="center", va="top",
            fontsize=9, color=WARM_BROWN, fontstyle="italic")
    ax.text(7.5, 0.55, "can violate\nnecklace", ha="center", va="top",
            fontsize=9, color=WARM_BROWN, fontstyle="italic")

    # Rate annotations
    ax.text(9.5, 3.0 - bh / 2 - 0.25, "0/60 trials",
            ha="center", va="top", fontsize=10, color=SAGE, fontweight="bold")
    ax.text(9.5, 1.0 + bh / 2 + 0.25, "2/60 trials",
            ha="center", va="bottom", fontsize=10, color=INDIAN_RED,
            fontweight="bold")

    # "both trained by gradient descent" label
    ax.text(1.5, 2.0, "both trained by\ngradient descent",
            ha="center", va="center", fontsize=9, color=FG_DIM,
            fontstyle="italic")

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "The positroid mechanism: TP structure → contiguous support → positroid",
        fontsize=13, pad=15,
    )

    path = OUTDIR / "positroid-fig7-mechanism.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ──


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating update figures...")
    fig5_support_patterns()
    fig6_baseline_comparison()
    fig7_causal_chain()
    print("Done!")


if __name__ == "__main__":
    main()
