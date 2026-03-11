"""Experiment: Diagnose optimization gap in positroid transformer modes.

Investigates why positroid attention modes plateau at 97-99% train accuracy
while standard reaches 100% on 10-class digit classification.

Tracks per-epoch loss, accuracy, and gradient norms per parameter group
to identify bottlenecks in the positroid backward pass.

Usage:
    python -m positroid.experiments.optimization_diagnosis
    python -m positroid.experiments.optimization_diagnosis --epochs 500 --modes standard positroid_k2 positroid_k2_trop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from positroid.datasets import DATASETS
from positroid.experiments.transformer_experiment import _build_classifier
from positroid.transformer._utils import Adam, cross_entropy, softmax


@dataclass
class EpochStats:
    """Statistics for one training epoch."""

    epoch: int
    loss: float
    train_accuracy: float
    grad_norms: dict[str, float]


def _param_group_names(mode: str, n_layers: int, n_heads: int) -> list[str]:
    """Generate parameter group names matching model.params() order."""
    names = ["W_embed", "b_embed", "pos_enc"]
    for layer in range(n_layers):
        names.extend([f"L{layer}_LN1_gamma", f"L{layer}_LN1_beta"])
        for head in range(n_heads):
            if mode == "standard":
                names.extend(
                    [
                        f"L{layer}_H{head}_W_Q",
                        f"L{layer}_H{head}_W_K",
                        f"L{layer}_H{head}_W_V",
                        f"L{layer}_H{head}_W_O",
                    ]
                )
            else:
                names.extend(
                    [
                        f"L{layer}_H{head}_face_raw",
                        f"L{layer}_H{head}_W_proj",
                        f"L{layer}_H{head}_W_V",
                        f"L{layer}_H{head}_W_O",
                        f"L{layer}_H{head}_self_bias",
                    ]
                )
        names.extend([f"L{layer}_LN2_gamma", f"L{layer}_LN2_beta"])
        if mode == "positroid_k2_trop":
            names.extend(
                [
                    f"L{layer}_MLP_face_raws",
                    f"L{layer}_MLP_W_enc",
                    f"L{layer}_MLP_W_read",
                    f"L{layer}_MLP_b_read",
                ]
            )
        else:
            names.extend(
                [
                    f"L{layer}_MLP_W1",
                    f"L{layer}_MLP_b1",
                    f"L{layer}_MLP_W2",
                    f"L{layer}_MLP_b2",
                ]
            )
    names.extend(["W_out", "b_out"])
    return names


def _categorize_param(name: str) -> str:
    """Categorize a parameter name into a group."""
    if "face" in name:
        return "face_weights"
    if "W_proj" in name:
        return "W_proj"
    if "self_bias" in name:
        return "self_bias"
    if any(x in name for x in ["W_Q", "W_K"]):
        return "W_QK"
    if "W_V" in name or "W_O" in name:
        return "W_VO"
    if "MLP" in name:
        return "MLP"
    if "LN" in name:
        return "LayerNorm"
    if "embed" in name or "pos_enc" in name:
        return "embedding"
    if "out" in name:
        return "output_head"
    return "other"


def train_with_diagnostics(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    epochs: int = 200,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 42,
    n_layers: int = 2,
    n_heads: int = 2,
) -> list[EpochStats]:
    """Train with per-epoch gradient norm tracking."""
    rng = np.random.default_rng(seed)
    params = model.params()  # type: ignore[attr-defined]
    opt = Adam(params, lr=lr)
    n_samples = X.shape[0]
    param_names = _param_group_names(mode, n_layers, n_heads)
    assert len(param_names) == len(params), (
        f"Name count {len(param_names)} != param count {len(params)}"
    )
    stats: list[EpochStats] = []

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_s, y_s = X[perm], y[perm]
        epoch_loss = 0.0
        n_batches = 0
        grad_norm_accum = {name: 0.0 for name in param_names}

        for start in range(0, n_samples, batch_size):
            X_b = X_s[start : start + batch_size]
            y_b = y_s[start : start + batch_size]

            logits, cache = model.forward(X_b)  # type: ignore[attr-defined]
            probs = softmax(logits, axis=-1)
            loss = cross_entropy(probs, y_b)
            epoch_loss += loss
            n_batches += 1

            bs = X_b.shape[0]
            C = model.n_classes  # type: ignore[attr-defined]
            one_hot = np.zeros((bs, C))
            one_hot[np.arange(bs), y_b.astype(int)] = 1.0
            d_logits = (probs - one_hot) / bs

            d_W_out = d_logits.T @ cache["pooled"]
            d_b_out = d_logits.sum(0)
            d_pooled = d_logits @ model.W_out  # type: ignore[attr-defined]

            d_h_out = np.zeros_like(cache["h_out"])
            d_h_out += d_pooled[:, None, :] / model.n_tokens  # type: ignore[attr-defined]

            all_block_grads: list[list[list[np.ndarray]]] = [
                []
                for _ in model.blocks  # type: ignore[attr-defined]
            ]
            d_h = np.zeros_like(cache["h"])
            for b in range(bs):
                d_x = d_h_out[b]
                for layer_idx in range(
                    len(model.blocks) - 1,  # type: ignore[attr-defined]
                    -1,
                    -1,
                ):
                    bc = cache["block_caches"][b][layer_idx]
                    d_x, bg = model.blocks[layer_idx].backward(  # type: ignore[attr-defined]
                        d_x, bc
                    )
                    all_block_grads[layer_idx].append(bg)
                d_h[b] = d_x

            block_grads_avg: list[np.ndarray] = []
            for layer_idx in range(len(model.blocks)):  # type: ignore[attr-defined]
                n_params = len(all_block_grads[layer_idx][0])
                for p_idx in range(n_params):
                    avg = np.mean(
                        [all_block_grads[layer_idx][b][p_idx] for b in range(bs)],
                        axis=0,
                    )
                    block_grads_avg.append(avg)

            d_pos_enc = d_h.sum(axis=0)
            d_h_flat = d_h.reshape(
                bs,
                model.n_tokens * model.d_model,  # type: ignore[attr-defined]
            )
            d_W_embed = d_h_flat.T @ X_b
            d_b_embed = d_h_flat.sum(0)

            grads = [d_W_embed, d_b_embed, d_pos_enc]
            grads.extend(block_grads_avg)
            grads.extend([d_W_out, d_b_out])

            for name, g in zip(param_names, grads):
                grad_norm_accum[name] += float(np.linalg.norm(g.ravel()))

            opt.step(grads)

        grad_norms = {name: v / n_batches for name, v in grad_norm_accum.items()}

        logits_all, _ = model.forward(X)  # type: ignore[attr-defined]
        preds = np.argmax(logits_all, axis=1)
        accuracy = float(np.mean(preds == y.astype(int)))

        stats.append(
            EpochStats(
                epoch=epoch,
                loss=epoch_loss / n_batches,
                train_accuracy=accuracy,
                grad_norms=grad_norms,
            )
        )

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3d}  loss={stats[-1].loss:.4f}  train_acc={accuracy:.1%}")

    return stats


def print_diagnosis(all_stats: dict[str, list[EpochStats]]) -> None:
    """Print diagnosis summary."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION DIAGNOSIS SUMMARY")
    print("=" * 80)

    # Convergence comparison
    print("\n--- Convergence ---")
    print(
        f"{'Mode':>20}  {'Ep25 Loss':>9}  {'Ep25 Acc':>8}  "
        f"{'Ep100 Loss':>10}  {'Ep100 Acc':>9}  "
        f"{'Final Loss':>10}  {'Final Acc':>9}"
    )
    for mode, stats in all_stats.items():
        ep25 = stats[min(24, len(stats) - 1)]
        ep100 = stats[min(99, len(stats) - 1)]
        final = stats[-1]
        print(
            f"{mode:>20}  {ep25.loss:>9.4f}  {ep25.train_accuracy:>8.1%}  "
            f"{ep100.loss:>10.4f}  {ep100.train_accuracy:>9.1%}  "
            f"{final.loss:>10.4f}  {final.train_accuracy:>9.1%}"
        )

    # Gradient norms by category at key epochs
    print("\n--- Mean Gradient Norms by Category ---")
    checkpoints = [0, 24, 99]
    for mode, stats in all_stats.items():
        print(f"\n  {mode}:")
        for ep_idx in checkpoints:
            if ep_idx >= len(stats):
                continue
            s = stats[ep_idx]
            cats: dict[str, list[float]] = {}
            for name, val in s.grad_norms.items():
                cat = _categorize_param(name)
                cats.setdefault(cat, []).append(val)
            cat_means = {c: np.mean(vs) for c, vs in cats.items()}
            parts = [f"epoch {s.epoch:>3d}:"]
            for cat in sorted(cat_means.keys()):
                parts.append(f"{cat}={cat_means[cat]:.2e}")
            print(f"    {' | '.join(parts)}")

    # Face weight gradient health check
    print("\n--- Face Weight Gradient Analysis ---")
    for mode, stats in all_stats.items():
        face_names = [n for n in stats[0].grad_norms if "face" in n]
        if not face_names:
            continue
        print(f"\n  {mode}:")
        for ep_idx in [0, 24, 99, len(stats) - 1]:
            if ep_idx >= len(stats):
                continue
            s = stats[ep_idx]
            face_norms = [s.grad_norms[n] for n in face_names]
            all_norms = list(s.grad_norms.values())
            face_mean = np.mean(face_norms)
            total_mean = np.mean(all_norms)
            ratio = face_mean / total_mean if total_mean > 0 else 0
            print(
                f"    epoch {s.epoch:>3d}: face_mean={face_mean:.2e}  "
                f"total_mean={total_mean:.2e}  ratio={ratio:.3f}"
            )

    # Parameter value statistics (check for explosion/collapse)
    print("\n--- Face Weight Value Statistics ---")
    for mode, stats in all_stats.items():
        face_names = [n for n in stats[0].grad_norms if "face" in n]
        if not face_names:
            continue
        # We can't easily access param values from stats, but we can
        # look at gradient magnitude trends
        print(f"\n  {mode}: gradient magnitude trend")
        face_norms_over_time = []
        for s in stats:
            face_norms_over_time.append(np.mean([s.grad_norms[n] for n in face_names]))
        # Show trend: first 25, middle, last 25
        first = np.mean(face_norms_over_time[:25])
        if len(face_norms_over_time) > 100:
            mid = np.mean(face_norms_over_time[75:125])
        else:
            mid = np.mean(
                face_norms_over_time[
                    len(face_norms_over_time) // 3 : 2 * len(face_norms_over_time) // 3
                ]
            )
        last = np.mean(face_norms_over_time[-25:])
        print(f"    first_25_epochs={first:.2e}  mid={mid:.2e}  last_25={last:.2e}")
        if last < first * 0.01:
            print("    WARNING: Face weight gradients may be vanishing!")
        elif last > first * 100:
            print("    WARNING: Face weight gradients may be exploding!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimization Gap Diagnosis")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--modes", nargs="+", default=["standard", "positroid_k2", "positroid_k2_trop"]
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x, y = DATASETS["digits_10class_pca10"](n_samples=args.n_samples, rng=rng)
    n = x.shape[0]
    n_test = max(1, int(n * 0.2))
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    x_train, y_train = x[n_test:], y[n_test:]

    n_classes = int(np.max(y)) + 1
    d_input = x.shape[1]

    d_model, n_tokens, n_layers, n_heads, n_param = 16, 4, 2, 2, 8

    all_stats: dict[str, list[EpochStats]] = {}
    for mode in args.modes:
        print(f"\n{'=' * 60}")
        print(f"Training: {mode} ({args.epochs} epochs)")
        print(f"{'=' * 60}")

        model = _build_classifier(
            mode,
            d_input,
            n_classes,
            d_model,
            n_tokens,
            n_layers,
            n_heads,
            n_param,
            args.seed,
        )
        print(f"  params: {model.param_count()}")

        stats = train_with_diagnostics(
            model,
            x_train,
            y_train,
            mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=32,
            seed=args.seed,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        all_stats[mode] = stats

    print_diagnosis(all_stats)


if __name__ == "__main__":
    main()
