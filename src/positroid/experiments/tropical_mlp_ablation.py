"""Experiment: Tropical MLP Ablation.

Tests what makes the tropical MLP work by comparing five MLP variants
inside an otherwise identical transformer (standard attention, same
embedding/output head):

  standard     — StandardMLP with d_ff=4*d_model (full baseline)
  tropical     — TropicalMLP with boundary measurement (positroid det)
  uncons_det   — DetMLP with unconstrained learnable k×n matrices + det
  fixed_det    — DetMLP with fixed random k×n matrices + det
  small_relu   — StandardMLP with d_ff tuned to match tropical param count

The experiment isolates:
  tropical vs uncons_det  → does the positroid constraint on B matter?
  uncons_det vs fixed_det → does learning B matter?
  tropical vs small_relu  → det nonlinearity vs ReLU at matched params?
  fixed_det vs small_relu → is det better than ReLU when nothing is learned?

Usage:
    python -m positroid.experiments.tropical_mlp_ablation
    python -m positroid.experiments.tropical_mlp_ablation --epochs 200 --num-trials 5
"""

from __future__ import annotations

from typing import Any

import argparse
from dataclasses import dataclass
from time import time

import numpy as np

from positroid.datasets import DATASETS
from positroid.transformer.det_mlp import DetMLP
from positroid.transformer.model import (
    StandardMLP,
    StandardMultiHeadAttention,
    TransformerBlock,
    train_classifier,
)
from positroid.transformer.tropical_mlp import TropicalMLP


# ── Configuration ──

ABLATION_MODES = [
    "standard",
    "tropical",
    "uncons_det",
    "fixed_det",
    "small_relu",
]


@dataclass
class AblationResult:
    """Result of one ablation trial."""

    mode: str
    mlp_params: int
    total_params: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    elapsed: float
    seed: int


# ── Model construction ──


class _Classifier:
    """Transformer classifier with pluggable MLP (standard attention fixed)."""

    def __init__(
        self,
        d_input: int,
        n_classes: int,
        d_model: int,
        n_tokens: int,
        blocks: list[TransformerBlock],
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_classes = n_classes

        scale = (2.0 / d_input) ** 0.5
        self.W_embed = rng.standard_normal((n_tokens * d_model, d_input)) * scale
        self.b_embed = np.zeros(n_tokens * d_model)
        self.pos_enc = rng.standard_normal((n_tokens, d_model)) * 0.02

        self.blocks = blocks

        self.W_out = rng.standard_normal((n_classes, d_model)) * (2.0 / d_model) ** 0.5
        self.b_out = np.zeros(n_classes)

    def params(self) -> list[np.ndarray]:
        p: list[np.ndarray] = [self.W_embed, self.b_embed, self.pos_enc]
        for block in self.blocks:
            p.extend(block.params())
        p.extend([self.W_out, self.b_out])
        return p

    def param_count(self) -> int:
        return sum(p.size for p in self.params())

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        batch = X.shape[0]
        h_flat = X @ self.W_embed.T + self.b_embed
        h = h_flat.reshape(batch, self.n_tokens, self.d_model) + self.pos_enc

        all_block_caches = []
        h_out = np.zeros_like(h)
        for b in range(batch):
            x_b = h[b]
            caches_b = []
            for block in self.blocks:
                x_b, bc = block.forward(x_b)
                caches_b.append(bc)
            h_out[b] = x_b
            all_block_caches.append(caches_b)

        pooled = h_out.mean(axis=1)
        logits = pooled @ self.W_out.T + self.b_out

        cache = {
            "X": X,
            "h_flat": h_flat,
            "h": h,
            "h_out": h_out,
            "pooled": pooled,
            "block_caches": all_block_caches,
        }
        return logits, cache


def _build_ablation_model(
    mode: str,
    d_input: int,
    n_classes: int,
    d_model: int,
    n_tokens: int,
    n_layers: int,
    n_heads: int,
    n: int,
    seed: int,
) -> _Classifier:
    """Build classifier with standard attention and the specified MLP variant."""
    rng = np.random.default_rng(seed)

    blocks = []
    for _ in range(n_layers):
        block_seed = int(rng.integers(2**31))

        # All modes use standard attention
        attn = StandardMultiHeadAttention(d_model, n_heads, seed=block_seed)

        mlp_seed = int(rng.integers(2**31))

        if mode == "standard":
            mlp: StandardMLP | TropicalMLP | DetMLP = StandardMLP(d_model, seed=mlp_seed)
        elif mode == "tropical":
            mlp = TropicalMLP(d_model, d_model, n_cells=4, k=2, n=n, seed=mlp_seed)
        elif mode == "uncons_det":
            mlp = DetMLP(
                d_model,
                d_model,
                n_cells=4,
                k=2,
                n=n,
                matrix_mode="unconstrained",
                seed=mlp_seed,
            )
        elif mode == "fixed_det":
            mlp = DetMLP(
                d_model,
                d_model,
                n_cells=4,
                k=2,
                n=n,
                matrix_mode="fixed_random",
                seed=mlp_seed,
            )
        elif mode == "small_relu":
            # Match tropical param count: ~384 params
            # StandardMLP params = 2 * d_ff * d_model + d_ff + d_model
            # Solve: 2 * d_ff * 16 + d_ff + 16 = 384 → d_ff = 11
            mlp = StandardMLP(d_model, d_ff=11, seed=mlp_seed)
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

        blocks.append(TransformerBlock(d_model, attn, mlp))

    clf_seed = int(rng.integers(2**31))
    return _Classifier(d_input, n_classes, d_model, n_tokens, blocks, seed=clf_seed)


def _mlp_param_count(model: _Classifier) -> int:
    """Count only the MLP parameters (excluding attention, embedding, etc)."""
    total = 0
    for block in model.blocks:
        total += sum(p.size for p in block.mlp.params())
    return total


# ── Trial runner ──


def run_trial(
    mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    d_model: int = 16,
    n_tokens: int = 4,
    n_layers: int = 2,
    n_heads: int = 2,
    n: int = 8,
    epochs: int = 200,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 42,
) -> AblationResult:
    """Run a single ablation trial."""
    d_input = x_train.shape[1]
    t0 = time()

    model = _build_ablation_model(
        mode,
        d_input,
        n_classes,
        d_model,
        n_tokens,
        n_layers,
        n_heads,
        n,
        seed,
    )

    history = train_classifier(
        model,
        x_train,
        y_train,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    elapsed = time() - t0

    train_preds = np.argmax(model.forward(x_train)[0], axis=1)
    train_acc = float(np.mean(train_preds == y_train.astype(int)))

    test_preds = np.argmax(model.forward(x_test)[0], axis=1)
    test_acc = float(np.mean(test_preds == y_test.astype(int)))

    return AblationResult(
        mode=mode,
        mlp_params=_mlp_param_count(model),
        total_params=model.param_count(),
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        final_loss=history["losses"][-1] if history["losses"] else float("inf"),
        elapsed=elapsed,
        seed=seed,
    )


# ── Experiment driver ──


def run_ablation(
    modes: list[str] | None = None,
    num_trials: int = 3,
    n_samples: int = 500,
    epochs: int = 200,
    lr: float = 0.005,
    seed: int = 42,
) -> list[AblationResult]:
    """Run the full tropical MLP ablation experiment."""
    if modes is None:
        modes = ABLATION_MODES.copy()

    rng = np.random.default_rng(seed)
    x, y = DATASETS["digits_10class_pca10"](n_samples=n_samples, rng=rng)
    n_classes = int(np.max(y)) + 1

    # Train/test split
    n = x.shape[0]
    n_test = max(1, int(n * 0.2))
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    x_train, y_train = x[n_test:], y[n_test:]
    x_test, y_test = x[:n_test], y[:n_test]

    print(f"Dataset: digits_10class_pca10 ({n_samples} samples, {n_classes} classes)")
    print(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")
    print(f"Epochs: {epochs}, Trials: {num_trials}, LR: {lr}")

    results: list[AblationResult] = []

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode}")
        print(f"{'=' * 60}")

        for trial_idx in range(num_trials):
            trial_seed = seed + trial_idx
            r = run_trial(
                mode,
                x_train,
                y_train,
                x_test,
                y_test,
                n_classes,
                epochs=epochs,
                lr=lr,
                seed=trial_seed,
            )
            results.append(r)
            print(
                f"  trial {trial_idx}  "
                f"mlp_params={r.mlp_params:>4}  "
                f"total={r.total_params:>5}  "
                f"train={r.train_accuracy:>5.1%}  "
                f"test={r.test_accuracy:>5.1%}  "
                f"loss={r.final_loss:.4f}  "
                f"({r.elapsed:.1f}s)"
            )

    return results


def print_summary(results: list[AblationResult]) -> None:
    """Print aggregated results table."""
    print("\n" + "=" * 90)
    print("TROPICAL MLP ABLATION — SUMMARY")
    print("=" * 90)
    print(
        f"{'Mode':>14}  {'MLP Par':>7}  {'Total':>6}  "
        f"{'Train Acc':>12}  {'Test Acc':>12}  {'Loss':>12}"
    )
    print("-" * 90)

    groups: dict[str, list[AblationResult]] = {}
    for r in results:
        groups.setdefault(r.mode, []).append(r)

    for mode in ABLATION_MODES:
        if mode not in groups:
            continue
        trials = groups[mode]
        train_accs = [t.train_accuracy for t in trials]
        test_accs = [t.test_accuracy for t in trials]
        losses = [t.final_loss for t in trials]
        mlp_p = trials[0].mlp_params
        total_p = trials[0].total_params

        def _fmt(vals: list[float], pct: bool = False) -> str:
            m, s = np.mean(vals), np.std(vals)
            if pct:
                return f"{m:>5.1%} ± {s:>4.1%}"
            return f"{m:>6.4f} ± {s:>5.4f}"

        print(
            f"{mode:>14}  {mlp_p:>7}  {total_p:>6}  "
            f"{_fmt(train_accs, True)}  {_fmt(test_accs, True)}  "
            f"{_fmt(losses)}"
        )

    print("=" * 90)

    # Interpretation guide
    print("\nInterpretation:")
    print("  tropical vs uncons_det  → positroid constraint effect")
    print("  uncons_det vs fixed_det → learned vs fixed matrix effect")
    print("  tropical vs small_relu  → det vs ReLU at matched params")
    print("  standard vs small_relu  → effect of param reduction alone")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tropical MLP Ablation")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ABLATION_MODES,
        choices=ABLATION_MODES,
    )
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_ablation(
        modes=args.modes,
        num_trials=args.num_trials,
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
