"""Manim animation for the pruning blog post.

Creates an animation showing matroid-guided vs random pruning
on a trained ReLU network's hyperplane arrangement.

Usage:
    uv run manim -pql scripts/pruning_animation.py PruningScene
    # For higher quality:
    uv run manim -pqh scripts/pruning_animation.py PruningScene

Output will be in media/videos/pruning_animation/
Copy the .mp4 or render as gif:
    ffmpeg -i media/videos/pruning_animation/720p30/PruningScene.mp4 \
           -vf "fps=15,scale=800:-1" -loop 0 \
           ~/gh/harrisontotty.github.io/images/pruning-animation.gif
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
)

# Blog theme colors (from harrison.totty.dev CSS)
BG_COLOR = "#ffffff"
FG_COLOR = "#1d1b1b"
LINK_BLUE = "#60728d"
FG_DIM = "#abafb6"
BEIGE = "#ddd8cf"

# Semantic colors for the animation
ESSENTIAL_COLOR = "#608d7b"  # sage green for essential neurons
TAIL_COLOR = "#cd5c5c"  # indian red for tail neurons
DATA_CLASS0 = LINK_BLUE  # muted blue
DATA_CLASS1 = "#8d6072"  # muted rose (complement of link blue)
ACCENT_WARM = "#8d7260"  # warm brown for counters

# Blog font
BLOG_FONT = "Roboto Mono"


def _train_network():
    """Train a small network and return all needed data."""
    from positroid.datasets.toy2d import make_moons
    from positroid.experiments.pruning import (
        evaluate_network,
        identify_essential_and_tail,
        prune_full_removal,
    )
    from positroid.network.train import TrainConfig, train

    rng = np.random.default_rng(42)
    X, y = make_moons(n_samples=200, noise=0.1, rng=rng)

    cfg = TrainConfig(
        hidden_dim=16, learning_rate=0.01, epochs=100,
        param_mode="tp_exponential", seed=42,
    )
    net, _ = train(X, y, cfg)
    essential, tail, rank = identify_essential_and_tail(net)
    orig_acc, _ = evaluate_network(net, X, y)

    return net, X, y, essential, tail, rank, orig_acc


class PruningScene(Scene):
    """Animate matroid-guided pruning vs random pruning."""

    def construct(self):
        from positroid.experiments.pruning import evaluate_network, prune_full_removal

        # White background to match blog
        self.camera.background_color = BG_COLOR

        net, X, y, essential, tail, rank, orig_acc = _train_network()
        H = net.layers[0].weight.shape[0]
        W1 = net.layers[0].weight
        b1 = net.layers[0].bias

        # --- Scene setup ---
        title = Text(
            "Matroid-Guided Neural Network Pruning",
            font_size=36, font=BLOG_FONT, color=FG_COLOR,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.5)

        # Axes for the data plot
        axes = Axes(
            x_range=[-1.5, 2.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=6,
            y_length=4,
            tips=False,
            axis_config={"color": BEIGE, "stroke_width": 1},
        )
        axes.shift(DOWN * 0.3)

        # Plot data points
        data_dots = VGroup()
        for i in range(len(X)):
            color = DATA_CLASS0 if y[i] == 0 else DATA_CLASS1
            dot = Dot(
                axes.coords_to_point(X[i, 0], X[i, 1]),
                radius=0.02, color=color, fill_opacity=0.6,
            )
            data_dots.add(dot)

        self.play(Create(axes), FadeIn(data_dots), run_time=1.0)

        # Draw hyperplane lines
        xlims = (-1.5, 2.5)
        xs_arr = np.array([xlims[0], xlims[1]])

        hp_lines = []
        for i in range(H):
            w0, w1_val = W1[i]
            b_val = b1[i]
            if abs(w1_val) > 1e-10:
                y0 = -(w0 * xs_arr[0] + b_val) / w1_val
                y1 = -(w0 * xs_arr[1] + b_val) / w1_val
            else:
                # Vertical line
                xv = -b_val / w0 if abs(w0) > 1e-10 else 0
                y0, y1 = -1.5, 1.5
                xs_arr_local = np.array([xv, xv])

            if abs(w1_val) > 1e-10:
                start = axes.coords_to_point(xs_arr[0], float(y0))
                end = axes.coords_to_point(xs_arr[1], float(y1))
            else:
                start = axes.coords_to_point(float(xv), -1.5)
                end = axes.coords_to_point(float(xv), 1.5)

            is_ess = i in set(essential)
            color = ESSENTIAL_COLOR if is_ess else TAIL_COLOR
            opacity = 0.8 if is_ess else 0.35
            width = 2.5 if is_ess else 1.5

            line = Line(
                start, end,
                color=color,
                stroke_width=width,
                stroke_opacity=opacity,
            )
            hp_lines.append(line)

        # Animate hyperplanes appearing
        hp_group = VGroup(*hp_lines)
        self.play(
            *[Create(line) for line in hp_lines],
            run_time=1.5,
        )

        # Labels
        info = Text(
            f"{H} neurons: {len(essential)} essential + {len(tail)} tail",
            font_size=24, font=BLOG_FONT, color=LINK_BLUE,
        )
        info.next_to(axes, DOWN, buff=0.5)
        self.play(Write(info))
        self.wait(1.0)

        # --- Phase 1: Matroid-guided pruning ---
        phase_label = Text(
            "Matroid-guided pruning", font_size=28, font=BLOG_FONT, color=ESSENTIAL_COLOR,
        )
        phase_label.to_edge(UP, buff=0.3)
        self.play(Transform(title, phase_label))

        acc_label = Text(f"accuracy: {orig_acc:.1%}", font_size=22, font=BLOG_FONT, color=ESSENTIAL_COLOR)
        acc_label.next_to(axes, RIGHT, buff=0.5).shift(UP * 1.0)
        self.play(Write(acc_label))

        removed_count = 0
        removed_label = Text(
            f"removed: {removed_count}/{len(tail)}", font_size=22, font=BLOG_FONT, color=ACCENT_WARM,
        )
        removed_label.next_to(acc_label, DOWN, buff=0.3)
        self.play(Write(removed_label))

        # Remove tail neurons one by one (batched for speed)
        batch_size = max(1, len(tail) // 8)
        for batch_start in range(0, len(tail), batch_size):
            batch_end = min(batch_start + batch_size, len(tail))
            batch_neurons = tail[batch_start:batch_end]
            removed_count += len(batch_neurons)

            animations = [FadeOut(hp_lines[n]) for n in batch_neurons]
            self.play(*animations, run_time=0.5)

            # Update labels
            pruned_net = prune_full_removal(net, tail[:batch_end])
            new_acc, _ = evaluate_network(pruned_net, X, y)

            new_acc_label = Text(
                f"accuracy: {new_acc:.1%}", font_size=22, font=BLOG_FONT, color=ESSENTIAL_COLOR,
            )
            new_acc_label.move_to(acc_label)
            new_removed = Text(
                f"removed: {removed_count}/{len(tail)}",
                font_size=22, font=BLOG_FONT, color=ACCENT_WARM,
            )
            new_removed.move_to(removed_label)
            self.play(
                Transform(acc_label, new_acc_label),
                Transform(removed_label, new_removed),
                run_time=0.3,
            )

        self.wait(1.0)

        # Result callout
        result = Text("zero accuracy loss", font_size=28, font=BLOG_FONT, color=ESSENTIAL_COLOR)
        result.next_to(removed_label, DOWN, buff=0.5)
        self.play(Write(result), run_time=0.5)
        self.wait(1.5)

        # --- Phase 2: Random pruning ---
        # Reset: bring all lines back
        self.play(FadeOut(result), FadeOut(acc_label), FadeOut(removed_label))
        for line in hp_lines:
            line.set_stroke(opacity=0.5)
        self.play(*[FadeIn(line) for line in hp_lines], run_time=0.8)

        phase2_label = Text(
            "Random pruning (same count)", font_size=28, font=BLOG_FONT, color=TAIL_COLOR,
        )
        phase2_label.to_edge(UP, buff=0.3)
        self.play(Transform(title, phase2_label))

        acc_label2 = Text(f"accuracy: {orig_acc:.1%}", font_size=22, font=BLOG_FONT, color=ESSENTIAL_COLOR)
        acc_label2.next_to(axes, RIGHT, buff=0.5).shift(UP * 1.0)
        removed_label2 = Text(
            f"removed: 0/{len(tail)}", font_size=22, font=BLOG_FONT, color=ACCENT_WARM,
        )
        removed_label2.next_to(acc_label2, DOWN, buff=0.3)
        self.play(Write(acc_label2), Write(removed_label2))

        # Random neuron selection
        rand_rng = np.random.default_rng(777)
        random_order = rand_rng.permutation(H).tolist()

        removed_count = 0
        for batch_start in range(0, len(tail), batch_size):
            batch_end = min(batch_start + batch_size, len(tail))
            batch_neurons = random_order[batch_start:batch_end]
            removed_count += len(batch_neurons)

            animations = [FadeOut(hp_lines[n]) for n in batch_neurons]
            self.play(*animations, run_time=0.5)

            pruned_net = prune_full_removal(net, random_order[:batch_end])
            new_acc, _ = evaluate_network(pruned_net, X, y)

            color = ESSENTIAL_COLOR if new_acc >= orig_acc - 0.02 else TAIL_COLOR
            new_acc_label2 = Text(
                f"accuracy: {new_acc:.1%}", font_size=22, font=BLOG_FONT, color=color,
            )
            new_acc_label2.move_to(acc_label2)
            new_removed2 = Text(
                f"removed: {removed_count}/{len(tail)}",
                font_size=22, font=BLOG_FONT, color=ACCENT_WARM,
            )
            new_removed2.move_to(removed_label2)
            self.play(
                Transform(acc_label2, new_acc_label2),
                Transform(removed_label2, new_removed2),
                run_time=0.3,
            )

        self.wait(1.0)

        result2 = Text("significant accuracy loss", font_size=28, font=BLOG_FONT, color=TAIL_COLOR)
        result2.next_to(removed_label2, DOWN, buff=0.5)
        self.play(Write(result2), run_time=0.5)
        self.wait(2.0)
