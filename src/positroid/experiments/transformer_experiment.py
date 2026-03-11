"""Experiment 012: Positroid Transformer Architecture Comparison.

Compares positroid attention (k=2, k=3) against standard QKV attention
on classification tasks. After training, analyzes learned attention
patterns for positroid matroid structure using Proposal E analysis tools.

Modes:
  standard           — standard QKV attention + standard MLP
  positroid_k2       — positroid attention (k=2) + standard MLP
  positroid_k3       — positroid attention (k=3) + standard MLP
  positroid_k2_trop  — positroid attention (k=2) + tropical MLP

Usage:
    python -m positroid.experiments.transformer_experiment

    python -m positroid.experiments.transformer_experiment \
        --datasets digits_10class_pca10 --modes positroid_k2 standard

    python -m positroid.experiments.transformer_experiment --epochs 100 --num-trials 5
"""

from __future__ import annotations

from typing import Any

import argparse
from dataclasses import dataclass, field
from time import time

import numpy as np

from positroid.datasets import DATASETS
from positroid.transformer._utils import softmax
from positroid.transformer.analysis import (
    check_attention_positroid,
    weight_effective_rank,
)
from positroid.transformer.model import (
    StandardMLP,
    StandardMultiHeadAttention,
    TransformerBlock,
    train_classifier,
)
from positroid.transformer.positroid_attention import PositroidMultiHeadAttention
from positroid.transformer.tropical_mlp import TropicalMLP


# ── Configuration ──


MODES = ["standard", "positroid_k2", "positroid_k3", "positroid_k2_trop"]


@dataclass
class TrialResult:
    """Result of one experiment trial."""

    dataset: str
    mode: str
    n_classes: int
    d_model: int
    n_heads: int
    n_layers: int
    num_params: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    attn_positroid_frac: float
    mean_effective_rank: float
    elapsed: float
    seed: int


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    datasets: list[str] = field(
        default_factory=lambda: ["digits_0v1_pca10", "digits_10class_pca10"]
    )
    modes: list[str] = field(default_factory=lambda: MODES.copy())
    num_trials: int = 3
    n_samples: int = 500
    d_model: int = 16
    n_tokens: int = 4
    n_layers: int = 2
    n_heads: int = 2
    n: int = 8
    epochs: int = 50
    lr: float = 0.005
    batch_size: int = 32
    test_frac: float = 0.2
    seed: int = 42


# ── Model construction ──


def _build_classifier(
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
    """Build a classifier for the given attention/MLP mode."""
    rng = np.random.default_rng(seed)

    blocks = []
    for _ in range(n_layers):
        block_seed = int(rng.integers(2**31))

        attn: StandardMultiHeadAttention | PositroidMultiHeadAttention
        if mode == "standard":
            attn = StandardMultiHeadAttention(d_model, n_heads, seed=block_seed)
        elif mode == "positroid_k3":
            attn = PositroidMultiHeadAttention(
                d_model,
                n_heads,
                n,
                k_values=[3] * n_heads,
                seed=block_seed,
            )
        else:
            # positroid_k2 and positroid_k2_trop
            attn = PositroidMultiHeadAttention(
                d_model,
                n_heads,
                n,
                k_values=[2] * n_heads,
                seed=block_seed,
            )

        mlp_seed = int(rng.integers(2**31))
        mlp: StandardMLP | TropicalMLP
        if mode == "positroid_k2_trop":
            mlp = TropicalMLP(
                d_model,
                d_model,
                n_cells=4,
                k=2,
                n=n,
                seed=mlp_seed,
            )
        else:
            mlp = StandardMLP(d_model, seed=mlp_seed)

        blocks.append(TransformerBlock(d_model, attn, mlp))

    clf_seed = int(rng.integers(2**31))
    return _Classifier(d_input, n_classes, d_model, n_tokens, blocks, seed=clf_seed)


class _Classifier:
    """Generic transformer classifier (same interface as PositroidClassifier).

    Embeds input → n_tokens × d_model, processes through transformer blocks,
    mean-pools, and classifies via a linear head.
    """

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        return softmax(logits, axis=-1)


# ── Attention pattern analysis ──


def _analyze_model(model: _Classifier, X: np.ndarray, n_samples: int = 20) -> dict[str, Any]:
    """Analyze attention patterns and weight matrices of a trained model."""
    batch = min(n_samples, X.shape[0])
    X_sub = X[:batch]

    # Embed
    h_flat = X_sub @ model.W_embed.T + model.b_embed
    h = h_flat.reshape(batch, model.n_tokens, model.d_model) + model.pos_enc

    n_positroid = 0
    n_total = 0

    for b in range(batch):
        x_b = h[b]
        for block in model.blocks:
            x_b, bc = block.forward(x_b)
            attn_caches = bc["c_attn"]
            for hc in attn_caches:
                attn_matrix = hc["attn"]
                result = check_attention_positroid(attn_matrix)
                if "error" not in result:
                    n_total += 1
                    if result.get("is_positroid", False):
                        n_positroid += 1

    # Weight effective rank across attention parameters
    effective_ranks: list[float] = []
    for block in model.blocks:
        for p in block.attn.params():
            if p.ndim == 2 and min(p.shape) >= 2:
                info = weight_effective_rank(p)
                effective_ranks.append(float(info["effective_rank"]))

    return {
        "attn_positroid_frac": n_positroid / n_total if n_total > 0 else 0.0,
        "mean_effective_rank": float(np.mean(effective_ranks)) if effective_ranks else 0.0,
    }


# ── Trial and experiment runners ──


def run_trial(
    dataset_name: str,
    mode: str,
    cfg: ExperimentConfig,
    seed: int,
) -> TrialResult:
    """Run a single trial: build, train, evaluate, analyze."""
    data_rng = np.random.default_rng(seed)
    x, y = DATASETS[dataset_name](n_samples=cfg.n_samples, rng=data_rng)
    n_classes = int(np.max(y)) + 1
    d_input = x.shape[1]

    # Train/test split
    n = x.shape[0]
    n_test = max(1, int(n * cfg.test_frac))
    perm = data_rng.permutation(n)
    x, y = x[perm], y[perm]
    x_train, y_train = x[n_test:], y[n_test:]
    x_test, y_test = x[:n_test], y[:n_test]

    t0 = time()

    model = _build_classifier(
        mode,
        d_input,
        n_classes,
        cfg.d_model,
        cfg.n_tokens,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n,
        seed,
    )

    history = train_classifier(
        model,
        x_train,
        y_train,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=seed,
    )

    elapsed = time() - t0

    # Evaluate
    train_preds = np.argmax(model.forward(x_train)[0], axis=1)
    train_acc = float(np.mean(train_preds == y_train.astype(int)))

    test_preds = np.argmax(model.forward(x_test)[0], axis=1)
    test_acc = float(np.mean(test_preds == y_test.astype(int)))

    # Analyze attention patterns
    analysis = _analyze_model(model, x_test)

    return TrialResult(
        dataset=dataset_name,
        mode=mode,
        n_classes=n_classes,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        num_params=model.param_count(),
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        final_loss=history["losses"][-1] if history["losses"] else float("inf"),
        attn_positroid_frac=analysis["attn_positroid_frac"],
        mean_effective_rank=analysis["mean_effective_rank"],
        elapsed=elapsed,
        seed=seed,
    )


def run_experiment(cfg: ExperimentConfig) -> list[TrialResult]:
    """Run the full experiment across datasets, modes, and trials."""
    results: list[TrialResult] = []

    for dataset_name in cfg.datasets:
        if dataset_name not in DATASETS:
            print(f"WARNING: {dataset_name} not in DATASETS, skipping.")
            continue

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        for mode in cfg.modes:
            for trial_idx in range(cfg.num_trials):
                trial_seed = cfg.seed + trial_idx
                trial = run_trial(dataset_name, mode, cfg, trial_seed)
                results.append(trial)
                print(
                    f"  {mode:>18}  trial {trial_idx}  "
                    f"params={trial.num_params:>5}  "
                    f"train={trial.train_accuracy:>5.1%}  "
                    f"test={trial.test_accuracy:>5.1%}  "
                    f"loss={trial.final_loss:.4f}  "
                    f"positroid={trial.attn_positroid_frac:.0%}  "
                    f"rank={trial.mean_effective_rank:.1f}  "
                    f"({trial.elapsed:.1f}s)"
                )

    return results


def print_results(results: list[TrialResult]) -> None:
    """Print aggregated results table."""
    print("\n" + "=" * 105)
    print("POSITROID TRANSFORMER EXPERIMENT — RESULTS")
    print("=" * 105)
    print(
        f"{'Dataset':>22}  {'Mode':>18}  {'Params':>6}  "
        f"{'Train':>8}  {'Test':>8}  {'Loss':>8}  "
        f"{'Positroid%':>10}  {'EffRank':>7}"
    )
    print("-" * 105)

    groups: dict[tuple[str, str], list[TrialResult]] = {}
    for r in results:
        groups.setdefault((r.dataset, r.mode), []).append(r)

    prev_dataset = ""
    for key in sorted(groups.keys()):
        dataset, mode = key
        trials = groups[key]

        if dataset != prev_dataset:
            if prev_dataset:
                print("-" * 105)
            prev_dataset = dataset

        train_accs = [t.train_accuracy for t in trials]
        test_accs = [t.test_accuracy for t in trials]
        losses = [t.final_loss for t in trials]
        pos_fracs = [t.attn_positroid_frac for t in trials]
        ranks = [t.mean_effective_rank for t in trials]
        params = trials[0].num_params

        def _fmt(vals: list[float], pct: bool = False) -> str:
            m, s = np.mean(vals), np.std(vals)
            if pct:
                return f"{m:>5.1%}±{s:.1%}" if s > 0.001 else f"{m:>5.1%}     "
            return f"{m:>5.3f}±{s:.3f}" if s > 0.001 else f"{m:>5.3f}     "

        print(
            f"{dataset:>22}  {mode:>18}  {params:>6}  "
            f"{_fmt(train_accs, True)}  {_fmt(test_accs, True)}  "
            f"{_fmt(losses)}  "
            f"{_fmt(pos_fracs, True)}  "
            f"{np.mean(ranks):>5.1f}"
        )

    print("=" * 105)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Positroid Transformer Architecture Comparison")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["digits_0v1_pca10", "digits_10class_pca10"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=MODES,
    )
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-tokens", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        datasets=args.datasets,
        modes=args.modes,
        num_trials=args.num_trials,
        n_samples=args.n_samples,
        d_model=args.d_model,
        n_tokens=args.n_tokens,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n=args.n,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    results = run_experiment(cfg)
    print_results(results)


if __name__ == "__main__":
    main()
