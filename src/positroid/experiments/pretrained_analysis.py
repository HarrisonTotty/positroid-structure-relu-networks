"""Experiment: Analyze pretrained GPT-2 weights for positroid structure.

Tests Conjecture 3: whether trained transformer weight matrices exhibit
approximate total positivity after appropriate normalization.

Also measures effective rank (validating arXiv:2602.10496 findings)
and attempts boundary measurement fit for small submatrices.

Requires: safetensors, huggingface-hub (not in core deps; install separately).

Usage:
    python -m positroid.experiments.pretrained_analysis
    python -m positroid.experiments.pretrained_analysis --layers 0 1 2 --max-order 3
"""

from __future__ import annotations

from typing import Any

import argparse
from itertools import combinations

import numpy as np

from positroid.transformer.analysis import fit_boundary_measurement


def load_gpt2_weights(cache_dir: str | None = None) -> dict[str, np.ndarray]:
    """Load GPT-2 small weights as numpy arrays via safetensors.

    Downloads from HuggingFace Hub if not cached.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    path = hf_hub_download(
        "openai-community/gpt2",
        "model.safetensors",
        cache_dir=cache_dir,
    )
    return load_file(path)  # type: ignore[no-any-return]


def extract_attention_heads(
    weights: dict[str, np.ndarray], layer: int
) -> dict[str, list[np.ndarray]]:
    """Extract per-head Q, K, V, O weight matrices for one layer.

    GPT-2 stores combined QKV as c_attn.weight (768, 2304) and
    output projection as c_proj.weight (768, 768).
    GPT-2 Conv1D: output = input @ weight + bias, so weight is (d_in, d_out).

    Returns dict with keys 'W_Q', 'W_K', 'W_V', 'W_O', each a list of
    12 matrices of shape (768, 64) or (64, 768).
    """
    prefix = f"h.{layer}.attn"
    c_attn = weights[f"{prefix}.c_attn.weight"]  # (768, 2304)
    c_proj = weights[f"{prefix}.c_proj.weight"]  # (768, 768)

    d_model = c_attn.shape[0]  # 768
    n_heads = 12
    d_head = d_model // n_heads  # 64

    W_Q, W_K, W_V = [], [], []
    for h in range(n_heads):
        W_Q.append(c_attn[:, h * d_head : (h + 1) * d_head])  # (768, 64)
        W_K.append(c_attn[:, d_model + h * d_head : d_model + (h + 1) * d_head])
        W_V.append(c_attn[:, 2 * d_model + h * d_head : 2 * d_model + (h + 1) * d_head])

    # Output projection: splits across heads on the input dimension
    W_O = []
    for h in range(n_heads):
        W_O.append(c_proj[h * d_head : (h + 1) * d_head, :])  # (64, 768)

    return {"W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_O": W_O}


def extract_mlp_weights(weights: dict[str, np.ndarray], layer: int) -> dict[str, np.ndarray]:
    """Extract MLP weights for one layer.

    GPT-2: c_fc.weight (768, 3072), c_proj.weight (3072, 768).
    """
    prefix = f"h.{layer}.mlp"
    return {
        "W_up": weights[f"{prefix}.c_fc.weight"],  # (768, 3072)
        "W_down": weights[f"{prefix}.c_proj.weight"],  # (3072, 768)
    }


def analyze_effective_rank(W: np.ndarray) -> dict[str, Any]:
    """Compute effective rank at multiple thresholds."""
    sv = np.linalg.svd(W, compute_uv=False)
    total = sv.sum()
    cumulative = np.cumsum(sv) / total if total > 0 else np.ones_like(sv)

    return {
        "shape": W.shape,
        "full_rank": min(W.shape),
        "rank_99": int(np.searchsorted(cumulative, 0.99) + 1),
        "rank_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "rank_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "top5_energy": float(cumulative[4]) if len(cumulative) > 4 else 1.0,
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "sv_top5": sv[:5].tolist(),
        "sv_ratio_1_5": float(sv[0] / sv[4]) if len(sv) > 4 and sv[4] > 0 else float("inf"),
    }


def check_submatrix_tp(
    W: np.ndarray, sub_size: int = 8, n_samples: int = 50, seed: int = 42
) -> dict[str, Any]:
    """Check total positivity of random contiguous submatrices.

    Samples n_samples contiguous sub_size x sub_size submatrices and
    checks the fraction of positive minors up to order 2 (and optionally 3).
    """
    rng = np.random.default_rng(seed)
    rows, cols = W.shape
    if rows < sub_size or cols < sub_size:
        sub_size = min(rows, cols)

    results = []
    for _ in range(n_samples):
        r0 = rng.integers(0, rows - sub_size + 1)
        c0 = rng.integers(0, cols - sub_size + 1)
        sub = W[r0 : r0 + sub_size, c0 : c0 + sub_size]

        # Check order-1 and order-2 minors
        n_pos = 0
        n_neg = 0
        n_zero = 0
        tol = 1e-8

        # Order 1: entries
        for v in sub.ravel():
            if v > tol:
                n_pos += 1
            elif v < -tol:
                n_neg += 1
            else:
                n_zero += 1

        # Order 2: 2x2 minors
        for r1, r2 in combinations(range(sub_size), 2):
            for c1, c2 in combinations(range(sub_size), 2):
                det = sub[r1, c1] * sub[r2, c2] - sub[r1, c2] * sub[r2, c1]
                if det > tol:
                    n_pos += 1
                elif det < -tol:
                    n_neg += 1
                else:
                    n_zero += 1

        total = n_pos + n_neg + n_zero
        results.append(
            {
                "frac_positive": n_pos / total if total > 0 else 0,
                "frac_negative": n_neg / total if total > 0 else 0,
                "n_negative": n_neg,
            }
        )

    frac_pos = np.mean([r["frac_positive"] for r in results])
    frac_neg = np.mean([r["frac_negative"] for r in results])
    any_neg = sum(1 for r in results if r["n_negative"] > 0)

    return {
        "sub_size": sub_size,
        "n_samples": n_samples,
        "mean_frac_positive": float(frac_pos),
        "mean_frac_negative": float(frac_neg),
        "submatrices_with_negatives": any_neg,
        "frac_with_negatives": any_neg / n_samples,
    }


def check_abs_tp(
    W: np.ndarray, sub_size: int = 8, n_samples: int = 50, seed: int = 42
) -> dict[str, Any]:
    """Check if |W| (absolute values) is closer to TP than W.

    If negating some rows/columns makes W more TP-like, abs(W) will
    have a higher fraction of positive minors.
    """
    return check_submatrix_tp(np.abs(W), sub_size, n_samples, seed)


def try_boundary_fit(W: np.ndarray, k: int = 4, seed: int = 42) -> dict[str, Any] | None:
    """Attempt boundary measurement fit on a small submatrix.

    Extracts a k x n submatrix (from the SVD low-rank approximation)
    and fits boundary measurement face weights.
    """
    s, Vt = np.linalg.svd(W, compute_uv=True, full_matrices=False)[1:]
    if len(s) < k:
        return None

    # Take top-k singular directions, project onto a k x n matrix
    # where n = min(cols, 2*k+2) to keep it tractable
    n = min(W.shape[1], 3 * k)
    # Use the top-k right singular vectors, truncated to n columns
    target = np.diag(s[:k]) @ Vt[:k, :n]  # (k, n)

    # Ensure all entries positive (boundary measurement produces positive matrices)
    # Try with absolute values
    target_abs = np.abs(target)
    # Scale so max entry is O(1)
    scale = target_abs.max()
    if scale > 0:
        target_abs = target_abs / scale

    if target_abs.shape[1] <= target_abs.shape[0]:
        return None

    result = fit_boundary_measurement(target_abs, k, max_iter=2000, lr=0.01)
    return {
        "k": k,
        "n": n,
        "relative_error": result["relative_error"],
        "absolute_error": result["error"],
        "sv_captured": float(np.sum(s[:k]) / np.sum(s)),
    }


def analyze_layer(weights: dict[str, np.ndarray], layer: int) -> dict[str, Any]:
    """Full analysis of one transformer layer."""
    heads = extract_attention_heads(weights, layer)
    mlp = extract_mlp_weights(weights, layer)

    result: dict[str, Any] = {"layer": layer, "attention": {}, "mlp": {}}

    # Attention: per-head analysis
    for mat_name in ["W_Q", "W_K", "W_V", "W_O"]:
        head_results = []
        for h, W in enumerate(heads[mat_name]):
            rank_info = analyze_effective_rank(W)
            tp_info = check_submatrix_tp(W, sub_size=8, n_samples=30, seed=42 + h)
            head_results.append(
                {
                    "head": h,
                    "rank_info": rank_info,
                    "tp_info": tp_info,
                }
            )
        result["attention"][mat_name] = head_results

    # MLP analysis
    for mat_name, W in mlp.items():
        rank_info = analyze_effective_rank(W)
        tp_info = check_submatrix_tp(W, sub_size=8, n_samples=30)
        abs_tp_info = check_abs_tp(W, sub_size=8, n_samples=30)
        result["mlp"][mat_name] = {
            "rank_info": rank_info,
            "tp_info": tp_info,
            "abs_tp_info": abs_tp_info,
        }

    return result


def print_layer_summary(layer_result: dict[str, Any]) -> None:
    """Print summary for one layer."""
    layer = layer_result["layer"]
    print(f"\n{'=' * 70}")
    print(f"Layer {layer}")
    print(f"{'=' * 70}")

    # Attention effective rank
    print("\n  Attention Effective Rank (99% energy):")
    print(f"    {'Matrix':>4}  {'Mean':>5}  {'Min':>4}  {'Max':>4}  {'Top5 Energy':>11}")
    for mat_name in ["W_Q", "W_K", "W_V", "W_O"]:
        heads = layer_result["attention"][mat_name]
        ranks = [h["rank_info"]["rank_99"] for h in heads]
        top5 = [h["rank_info"]["top5_energy"] for h in heads]
        print(
            f"    {mat_name:>4}  {np.mean(ranks):>5.1f}  {min(ranks):>4d}  "
            f"{max(ranks):>4d}  {np.mean(top5):>11.3f}"
        )

    # Attention TP fraction
    print("\n  Attention Minor Positivity (8x8 submatrices, order 1-2):")
    print(f"    {'Matrix':>4}  {'Mean %Pos':>9}  {'Mean %Neg':>9}  {'%Subs w/ Neg':>12}")
    for mat_name in ["W_Q", "W_K", "W_V", "W_O"]:
        heads = layer_result["attention"][mat_name]
        frac_pos = [h["tp_info"]["mean_frac_positive"] for h in heads]
        frac_neg = [h["tp_info"]["mean_frac_negative"] for h in heads]
        with_neg = [h["tp_info"]["frac_with_negatives"] for h in heads]
        print(
            f"    {mat_name:>4}  {np.mean(frac_pos):>9.1%}  {np.mean(frac_neg):>9.1%}  "
            f"{np.mean(with_neg):>12.1%}"
        )

    # MLP
    print("\n  MLP Effective Rank (99% energy):")
    for mat_name, info in layer_result["mlp"].items():
        r = info["rank_info"]
        print(
            f"    {mat_name:>6}: rank_99={r['rank_99']:>4d} / {r['full_rank']}  "
            f"top5_energy={r['top5_energy']:.3f}  "
            f"condition={r['condition_number']:.1f}"
        )

    print("\n  MLP Minor Positivity (raw vs |W|):")
    for mat_name, info in layer_result["mlp"].items():
        tp = info["tp_info"]
        abs_tp = info["abs_tp_info"]
        print(
            f"    {mat_name:>6}: raw %pos={tp['mean_frac_positive']:.1%}  "
            f"|W| %pos={abs_tp['mean_frac_positive']:.1%}  "
            f"raw %neg={tp['mean_frac_negative']:.1%}  "
            f"|W| %neg={abs_tp['mean_frac_negative']:.1%}"
        )


def print_cross_layer_summary(all_results: list[dict[str, Any]]) -> None:
    """Print summary across all analyzed layers."""
    print("\n" + "=" * 70)
    print("CROSS-LAYER SUMMARY")
    print("=" * 70)

    # Effective rank trend across layers
    print("\n  Mean Effective Rank (99% energy) by Layer:")
    print(
        f"  {'Layer':>5}  {'W_Q':>5}  {'W_K':>5}  {'W_V':>5}  {'W_O':>5}  {'MLP_up':>6}  {'MLP_dn':>6}"
    )
    for r in all_results:
        layer = r["layer"]
        ranks = {}
        for mat_name in ["W_Q", "W_K", "W_V", "W_O"]:
            ranks[mat_name] = np.mean([h["rank_info"]["rank_99"] for h in r["attention"][mat_name]])
        ranks["up"] = r["mlp"]["W_up"]["rank_info"]["rank_99"]
        ranks["dn"] = r["mlp"]["W_down"]["rank_info"]["rank_99"]
        print(
            f"  {layer:>5d}  {ranks['W_Q']:>5.1f}  {ranks['W_K']:>5.1f}  "
            f"{ranks['W_V']:>5.1f}  {ranks['W_O']:>5.1f}  "
            f"{ranks['up']:>6d}  {ranks['dn']:>6d}"
        )

    # TP fraction trend
    print("\n  Mean Minor Positivity (%) by Layer:")
    print(
        f"  {'Layer':>5}  {'W_Q':>5}  {'W_K':>5}  {'W_V':>5}  {'W_O':>5}  {'MLP_up':>6}  {'MLP_dn':>6}"
    )
    for r in all_results:
        layer = r["layer"]
        tp = {}
        for mat_name in ["W_Q", "W_K", "W_V", "W_O"]:
            tp[mat_name] = np.mean(
                [h["tp_info"]["mean_frac_positive"] for h in r["attention"][mat_name]]
            )
        tp["up"] = r["mlp"]["W_up"]["tp_info"]["mean_frac_positive"]
        tp["dn"] = r["mlp"]["W_down"]["tp_info"]["mean_frac_positive"]
        print(
            f"  {layer:>5d}  {tp['W_Q']:>5.1%}  {tp['W_K']:>5.1%}  "
            f"{tp['W_V']:>5.1%}  {tp['W_O']:>5.1%}  "
            f"{tp['up']:>5.1%}  {tp['dn']:>5.1%}"
        )

    print("\n  (Boundary measurement fitting results below)")


def run_boundary_fits(weights: dict[str, np.ndarray], layers: list[int]) -> None:
    """Run boundary measurement fits on representative submatrices."""
    print("\n" + "=" * 70)
    print("BOUNDARY MEASUREMENT FIT ANALYSIS")
    print("=" * 70)

    for layer in layers:
        heads = extract_attention_heads(weights, layer)
        print(f"\n  Layer {layer}:")

        for mat_name in ["W_V", "W_Q"]:
            # Try head 0 and head 6 (representative)
            for h_idx in [0, 6]:
                W = heads[mat_name][h_idx]  # (768, 64) or (64, 768)
                for k in [3, 5]:
                    result = try_boundary_fit(W, k=k, seed=42)
                    if result is not None:
                        print(
                            f"    {mat_name} head {h_idx:>2d} k={k}: "
                            f"rel_err={result['relative_error']:.4f}  "
                            f"sv_captured={result['sv_captured']:.3f}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained Weight Analysis")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 3, 6, 9, 11],
        help="Layer indices to analyze (default: 0, 3, 6, 9, 11)",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    print("Loading GPT-2 weights...")
    weights = load_gpt2_weights(args.cache_dir)

    # Show available keys
    print(f"Loaded {len(weights)} weight tensors")
    attn_keys = [k for k in weights if "attn" in k and "weight" in k]
    print(f"Attention weight keys (sample): {attn_keys[:4]}")

    all_results = []
    for layer in args.layers:
        print(f"\nAnalyzing layer {layer}...")
        result = analyze_layer(weights, layer)
        all_results.append(result)
        print_layer_summary(result)

    print_cross_layer_summary(all_results)
    run_boundary_fits(weights, args.layers)


if __name__ == "__main__":
    main()
