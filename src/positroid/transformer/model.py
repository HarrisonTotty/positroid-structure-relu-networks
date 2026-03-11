"""Positroid transformer integration: blocks and classifiers.

Combines positroid attention, tropical MLP, and standard components
into full transformer blocks and a classification model for testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from positroid.transformer._utils import Adam, cross_entropy, softmax
from positroid.transformer.positroid_attention import PositroidMultiHeadAttention
from positroid.transformer.tropical_mlp import TropicalMLP


# ── Building Blocks ──


class LayerNorm:
    """Layer normalization over the last dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def params(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        X_norm = (X - mean) / std
        out = self.gamma * X_norm + self.beta
        return out, {"X_norm": X_norm, "std": std}

    def backward(
        self, d_out: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        X_norm = cache["X_norm"]
        std = cache["std"]
        D = self.d_model

        d_gamma = (d_out * X_norm).sum(axis=0)
        d_beta = d_out.sum(axis=0)

        d_X_norm = d_out * self.gamma
        # Layer norm backward (per row)
        d_X = (1.0 / std) * (
            d_X_norm
            - d_X_norm.mean(axis=-1, keepdims=True)
            - X_norm * (d_X_norm * X_norm).mean(axis=-1, keepdims=True)
        )
        return d_X, [d_gamma, d_beta]


class StandardMLP:
    """Standard expand-contract MLP with ReLU (for comparison)."""

    def __init__(self, d_model: int, d_ff: int = 0, seed: int = 42) -> None:
        if d_ff == 0:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        rng = np.random.default_rng(seed)
        scale = (2.0 / d_model) ** 0.5
        self.W1 = rng.standard_normal((d_ff, d_model)) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_model, d_ff)) * (2.0 / d_ff) ** 0.5
        self.b2 = np.zeros(d_model)

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def param_count(self) -> int:
        return self.d_ff * self.d_model + self.d_ff + self.d_model * self.d_ff + self.d_model

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(0, z1)
        out = h1 @ self.W2.T + self.b2
        return out, {"X": X, "z1": z1, "h1": h1}

    def backward(
        self, d_out: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        X, z1, h1 = cache["X"], cache["z1"], cache["h1"]
        d_W2 = d_out.T @ h1
        d_b2 = d_out.sum(0)
        d_h1 = d_out @ self.W2
        d_z1 = d_h1 * (z1 > 0).astype(float)
        d_W1 = d_z1.T @ X
        d_b1 = d_z1.sum(0)
        d_X = d_z1 @ self.W1
        return d_X, [d_W1, d_b1, d_W2, d_b2]


class StandardAttentionHead:
    """Standard QKV dot-product attention head (baseline for comparison)."""

    def __init__(
        self,
        d_model: int,
        d_head: int,
        d_v: int | None = None,
        seed: int = 42,
    ) -> None:
        if d_v is None:
            d_v = d_model
        self.d_model = d_model
        self.d_head = d_head
        self.d_v = d_v

        rng = np.random.default_rng(seed)
        scale = (2.0 / d_model) ** 0.5
        self.W_Q = rng.standard_normal((d_model, d_head)) * scale
        self.W_K = rng.standard_normal((d_model, d_head)) * scale
        self.W_V = rng.standard_normal((d_model, d_v)) * scale
        self.W_O = rng.standard_normal((d_v, d_model)) * (2.0 / d_v) ** 0.5

    def params(self) -> list[np.ndarray]:
        return [self.W_Q, self.W_K, self.W_V, self.W_O]

    def set_params(self, param_list: list[np.ndarray]) -> None:
        self.W_Q, self.W_K, self.W_V, self.W_O = param_list

    def param_count(self) -> int:
        return self.d_model * self.d_head * 2 + self.d_model * self.d_v + self.d_v * self.d_model

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        Q = X @ self.W_Q  # (T, d_head)
        K = X @ self.W_K  # (T, d_head)
        V = X @ self.W_V  # (T, d_v)
        scores = Q @ K.T / (self.d_head**0.5)
        attn = softmax(scores, axis=-1)
        context = attn @ V  # (T, d_v)
        output = context @ self.W_O  # (T, d_model)

        cache = {
            "X": X,
            "Q": Q,
            "K": K,
            "V": V,
            "scores": scores,
            "attn": attn,
            "context": context,
        }
        return output, cache

    def backward(
        self, d_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        X = cache["X"]
        Q, K, V = cache["Q"], cache["K"], cache["V"]
        attn = cache["attn"]
        context = cache["context"]

        d_W_O = context.T @ d_output
        d_context = d_output @ self.W_O.T

        d_attn = d_context @ V.T
        d_V = attn.T @ d_context

        d_W_V = X.T @ d_V
        d_X_v = d_V @ self.W_V.T

        # Softmax backward
        s = (attn * d_attn).sum(axis=-1, keepdims=True)
        d_scores = attn * (d_attn - s)

        # scores = Q @ K.T / sqrt(d_head)
        scale = 1.0 / (self.d_head**0.5)
        d_Q = scale * (d_scores @ K)
        d_K = scale * (d_scores.T @ Q)

        d_W_Q = X.T @ d_Q
        d_W_K = X.T @ d_K
        d_X = d_X_v + d_Q @ self.W_Q.T + d_K @ self.W_K.T

        return d_X, [d_W_Q, d_W_K, d_W_V, d_W_O]


class StandardMultiHeadAttention:
    """Multi-head standard QKV attention (baseline)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int | None = None,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        if d_head is None:
            d_head = d_model // n_heads
        d_v = d_model // n_heads

        rng = np.random.default_rng(seed)
        self.heads = [
            StandardAttentionHead(d_model, d_head, d_v=d_v, seed=int(rng.integers(2**31)))
            for _ in range(n_heads)
        ]

    def params(self) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for h in self.heads:
            result.extend(h.params())
        return result

    def set_params(self, param_list: list[np.ndarray]) -> None:
        n_per = 4
        for i, h in enumerate(self.heads):
            h.set_params(param_list[i * n_per : (i + 1) * n_per])

    def param_count(self) -> int:
        return sum(h.param_count() for h in self.heads)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        output = np.zeros_like(X)
        caches = []
        for h in self.heads:
            h_out, h_cache = h.forward(X)
            output += h_out
            caches.append(h_cache)
        return output, caches

    def backward(
        self, d_output: np.ndarray, caches: list[dict[str, Any]]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        d_X = np.zeros((caches[0]["X"].shape[0], self.d_model))
        all_grads: list[np.ndarray] = []
        for h, cache in zip(self.heads, caches):
            d_X_h, grads_h = h.backward(d_output, cache)
            d_X += d_X_h
            all_grads.extend(grads_h)
        return d_X, all_grads


# ── Transformer Block ──


class TransformerBlock:
    """Generic pre-norm transformer block with pluggable attention and MLP.

    Pre-norm architecture:
        h = x + Attn(LN(x))
        out = h + MLP(LN(h))
    """

    def __init__(self, d_model: int, attn: Any, mlp: Any) -> None:
        self.norm1 = LayerNorm(d_model)
        self.attn = attn
        self.norm2 = LayerNorm(d_model)
        self.mlp = mlp

    def params(self) -> list[np.ndarray]:
        return (  # type: ignore[no-any-return]
            self.norm1.params() + self.attn.params() + self.norm2.params() + self.mlp.params()
        )

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        normed1, c_n1 = self.norm1.forward(X)
        attn_out, c_attn = self.attn.forward(normed1)
        h = X + attn_out

        normed2, c_n2 = self.norm2.forward(h)
        mlp_out, c_mlp = self.mlp.forward(normed2)
        out = h + mlp_out

        cache = {
            "X": X,
            "h": h,
            "c_n1": c_n1,
            "c_attn": c_attn,
            "c_n2": c_n2,
            "c_mlp": c_mlp,
            "normed1": normed1,
            "normed2": normed2,
        }
        return out, cache

    def backward(
        self, d_out: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        d_mlp_out, mlp_grads = self.mlp.backward(d_out, cache["c_mlp"])
        d_normed2, norm2_grads = self.norm2.backward(d_mlp_out, cache["c_n2"])
        d_h = d_out + d_normed2

        d_attn_out, attn_grads = self.attn.backward(d_h, cache["c_attn"])
        d_normed1, norm1_grads = self.norm1.backward(d_attn_out, cache["c_n1"])
        d_X = d_h + d_normed1

        grads = norm1_grads + attn_grads + norm2_grads + mlp_grads
        return d_X, grads


class PositroidTransformerBlock:
    """Single transformer block: positroid attention + MLP + residual + norm.

    Pre-norm architecture:
        h = x + Attn(LN(x))
        out = h + MLP(LN(h))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n: int,
        k_values: list[int] | None = None,
        mlp_type: str = "standard",
        mlp_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.norm1 = LayerNorm(d_model)
        self.attn = PositroidMultiHeadAttention(
            d_model, n_heads, n, k_values, seed=int(rng.integers(2**31))
        )
        self.norm2 = LayerNorm(d_model)

        kw = mlp_kwargs or {}
        if mlp_type == "tropical":
            self.mlp: TropicalMLP | StandardMLP = TropicalMLP(
                d_model, d_model, seed=int(rng.integers(2**31)), **kw
            )
        else:
            self.mlp = StandardMLP(d_model, seed=int(rng.integers(2**31)), **kw)

    def params(self) -> list[np.ndarray]:
        return self.norm1.params() + self.attn.params() + self.norm2.params() + self.mlp.params()

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """X: (T, d_model) -> (T, d_model), cache."""
        normed1, c_n1 = self.norm1.forward(X)
        attn_out, c_attn = self.attn.forward(normed1)
        h = X + attn_out

        normed2, c_n2 = self.norm2.forward(h)
        mlp_out, c_mlp = self.mlp.forward(normed2)
        out = h + mlp_out

        cache = {
            "X": X,
            "h": h,
            "c_n1": c_n1,
            "c_attn": c_attn,
            "c_n2": c_n2,
            "c_mlp": c_mlp,
            "normed1": normed1,
            "normed2": normed2,
        }
        return out, cache

    def backward(
        self, d_out: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward through the block."""
        # MLP residual: out = h + mlp(norm2(h))
        d_mlp_out, mlp_grads = self.mlp.backward(d_out, cache["c_mlp"])
        d_normed2, norm2_grads = self.norm2.backward(d_mlp_out, cache["c_n2"])
        d_h = d_out + d_normed2

        # Attention residual: h = X + attn(norm1(X))
        d_attn_out, attn_grads = self.attn.backward(d_h, cache["c_attn"])
        d_normed1, norm1_grads = self.norm1.backward(d_attn_out, cache["c_n1"])
        d_X = d_h + d_normed1

        grads = norm1_grads + attn_grads + norm2_grads + mlp_grads
        return d_X, grads


# ── Classifier ──


class PositroidClassifier:
    """Classification model using positroid transformer blocks.

    Splits input features into n_tokens chunks, processes through
    transformer blocks with positroid attention, pools, and classifies.

    This allows testing positroid attention on non-sequential data
    (e.g., tabular datasets) by treating feature groups as tokens.
    """

    def __init__(
        self,
        d_input: int,
        n_classes: int,
        d_model: int = 32,
        n_tokens: int = 4,
        n_layers: int = 2,
        n_heads: int = 2,
        n: int = 8,
        k_values: list[int] | None = None,
        mlp_type: str = "standard",
        mlp_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.d_input = d_input
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_classes = n_classes

        # Embedding: d_input → n_tokens * d_model
        scale = (2.0 / d_input) ** 0.5
        self.W_embed = rng.standard_normal((n_tokens * d_model, d_input)) * scale
        self.b_embed = np.zeros(n_tokens * d_model)

        # Learnable positional encoding
        self.pos_enc = rng.standard_normal((n_tokens, d_model)) * 0.02

        # Transformer blocks
        self.blocks = [
            PositroidTransformerBlock(
                d_model,
                n_heads,
                n,
                k_values,
                mlp_type,
                mlp_kwargs,
                seed=int(rng.integers(2**31)),
            )
            for _ in range(n_layers)
        ]

        # Output head
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
        """Forward pass.

        Args:
            X: (batch, d_input)

        Returns:
            logits: (batch, n_classes)
            cache: intermediates.
        """
        batch = X.shape[0]

        h_flat = X @ self.W_embed.T + self.b_embed  # (batch, T*d)
        h = h_flat.reshape(batch, self.n_tokens, self.d_model) + self.pos_enc

        # Per-sample transformer (attention is T×T per sample)
        all_block_caches = []
        h_out = np.zeros_like(h)
        for b in range(batch):
            x_b = h[b]  # (T, d_model)
            caches_b = []
            for block in self.blocks:
                x_b, bc = block.forward(x_b)
                caches_b.append(bc)
            h_out[b] = x_b
            all_block_caches.append(caches_b)

        pooled = h_out.mean(axis=1)  # (batch, d_model)
        logits = pooled @ self.W_out.T + self.b_out  # (batch, n_classes)

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
        """Predict class probabilities."""
        logits, _ = self.forward(X)
        return softmax(logits, axis=-1)


def train_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a transformer classifier with Adam + softmax cross-entropy.

    The model must expose: params(), forward(X), n_classes, n_tokens, d_model,
    W_embed, b_embed, pos_enc, W_out, b_out, blocks (list of TransformerBlock).

    Returns dict with losses and accuracies per epoch.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    n_samples = X.shape[0]
    history: dict[str, list[float]] = {"losses": [], "accuracies": []}

    for _epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_s, y_s = X[perm], y[perm]
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            X_b = X_s[start : start + batch_size]
            y_b = y_s[start : start + batch_size]

            logits, cache = model.forward(X_b)
            probs = softmax(logits, axis=-1)
            loss = cross_entropy(probs, y_b)
            epoch_loss += loss
            n_batches += 1

            # Backward (numerical gradients for now — full backprop through
            # the classifier requires chaining block.backward, which works
            # but is expensive; for training we use finite-difference-free
            # Adam on the forward loss)
            bs = X_b.shape[0]
            C = model.n_classes
            one_hot = np.zeros((bs, C))
            one_hot[np.arange(bs), y_b.astype(int)] = 1.0
            d_logits = (probs - one_hot) / bs

            # Backprop through output head
            d_W_out = d_logits.T @ cache["pooled"]
            d_b_out = d_logits.sum(0)
            d_pooled = d_logits @ model.W_out  # (batch, d_model)

            # Backprop through mean pooling
            d_h_out = np.zeros_like(cache["h_out"])
            d_h_out += d_pooled[:, None, :] / model.n_tokens

            # Backprop through transformer blocks (per sample)
            all_block_grads: list[list[np.ndarray]] = [[] for _ in model.blocks]
            d_h = np.zeros_like(cache["h"])
            for b in range(bs):
                d_x = d_h_out[b]
                for layer_idx in range(len(model.blocks) - 1, -1, -1):
                    bc = cache["block_caches"][b][layer_idx]
                    d_x, bg = model.blocks[layer_idx].backward(d_x, bc)
                    all_block_grads[layer_idx].append(bg)
                d_h[b] = d_x

            # Average block gradients across batch
            block_grads_avg: list[np.ndarray] = []
            for layer_idx in range(len(model.blocks)):
                n_params = len(all_block_grads[layer_idx][0])
                for p_idx in range(n_params):
                    avg = np.mean(
                        [all_block_grads[layer_idx][b][p_idx] for b in range(bs)],
                        axis=0,
                    )
                    block_grads_avg.append(avg)

            # Backprop through pos_enc and embedding
            d_pos_enc = d_h.sum(axis=0)  # (T, d_model)
            d_h_flat = d_h.reshape(bs, model.n_tokens * model.d_model)
            d_W_embed = d_h_flat.T @ X_b
            d_b_embed = d_h_flat.sum(0)

            grads = [d_W_embed, d_b_embed, d_pos_enc]
            grads.extend(block_grads_avg)
            grads.extend([d_W_out, d_b_out])

            opt.step(grads)

        history["losses"].append(epoch_loss / n_batches)

        # Epoch accuracy
        logits_all, _ = model.forward(X)
        preds = np.argmax(logits_all, axis=1)
        accuracy = float(np.mean(preds == y.astype(int)))
        history["accuracies"].append(accuracy)

    return history
