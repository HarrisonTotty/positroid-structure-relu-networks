"""Experiment: Test Conjecture 2 — do GPT-2 attention patterns form positroid matroids?

Runs GPT-2-small on actual text, extracts attention matrices (after softmax),
and checks whether submatrices exhibit positroid structure.

KEY INSIGHT: Full T×T causal attention matrices are lower-triangular with
positive diagonal → full rank → uniform matroid → trivially positroid.
To get non-trivial structure, we extract rectangular submatrices:
  q_window query rows × n_keys key columns
where all queries can see all keys (no causal masking in the subblock).

We check two things:
1. TOTAL NONNEGATIVITY: Are all maximal minors (Plücker coordinates) of the
   attention submatrix nonnegative? If yes → point in Gr_+(q, n) → positroid.
2. MATROID STRUCTURE: Is the column matroid (defined by nonzero maximal minors)
   a positroid? This uses the Grassmann necklace algorithm.

The matroid is constructed directly from maximal minor signs (much faster than
the SVD-based linear_matroid_from_vectors for large matrices).

Usage:
    python -m positroid.experiments.attention_positroid
    python -m positroid.experiments.attention_positroid --n-keys 16 --q-window 4
"""

from __future__ import annotations

from typing import Any

import argparse
import sys
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from positroid.matroid.matroid import Matroid
from positroid.matroid.positroid import is_positroid


@dataclass
class SubmatrixAnalysis:
    """Analysis for one rectangular attention submatrix."""

    layer: int
    head: int
    input_id: int
    window_id: int
    q_start: int
    k_start: int
    q_window: int
    n_keys: int
    # Matroid from maximal minors
    matroid_rank: int
    n_bases: int
    is_uniform: bool
    is_positroid: bool
    # Total nonnegativity
    n_minors_positive: int
    n_minors_negative: int
    n_minors_zero: int
    frac_nonneg: float  # (positive + zero) / total
    min_minor: float
    max_minor: float
    # Attention stats
    entropy: float
    max_attn: float
    mean_attn: float
    errors: list[str] = field(default_factory=list)


def maximal_minors(sub: np.ndarray) -> list[tuple[frozenset[int], float]]:
    """Compute all maximal (q×q) minors of a q×n matrix.

    Returns list of (column_subset, determinant) pairs.
    The column_subset is a frozenset of column indices.
    """
    q, n = sub.shape
    result = []
    for cols in combinations(range(n), q):
        det = float(np.linalg.det(sub[:, list(cols)]))
        result.append((frozenset(cols), det))
    return result


def _make_matroid_trusted(
    ground_set: frozenset[int],
    bases: frozenset[frozenset[int]],
    rank: int,
) -> Matroid:
    """Create a Matroid bypassing exchange axiom validation.

    Only use when the bases are known to be valid (e.g., from matrix column matroids).
    """
    mat = object.__new__(Matroid)
    mat._ground_set = ground_set
    mat._bases = bases
    mat._rank = rank
    return mat


def matroid_from_minors(
    minors: list[tuple[frozenset[int], float]],
    n: int,
    tol: float = 1e-10,
) -> Matroid:
    """Construct matroid from maximal minor values.

    Bases are the column subsets with |det| > tol.
    Skips exchange axiom validation since matrix column matroids are valid by construction.
    """
    bases: set[frozenset[int]] = set()
    for cols, det in minors:
        if abs(det) > tol:
            bases.add(cols)

    ground = frozenset(range(n))
    if not bases:
        return _make_matroid_trusted(ground, frozenset([frozenset()]), 0)

    fb = frozenset(bases)
    rank = len(next(iter(fb)))
    return _make_matroid_trusted(ground, fb, rank)


def analyze_submatrix_minors(sub: np.ndarray, tol: float = 1e-10) -> dict[str, Any]:
    """Analyze maximal minors of a q×n attention submatrix.

    Returns matroid info, total nonnegativity stats, and minor statistics.
    """
    q, n = sub.shape
    minors = maximal_minors(sub)

    n_pos = sum(1 for _, d in minors if d > tol)
    n_neg = sum(1 for _, d in minors if d < -tol)
    n_zero = sum(1 for _, d in minors if abs(d) <= tol)
    total = len(minors)

    det_vals = [d for _, d in minors]
    min_det = min(det_vals) if det_vals else 0.0
    max_det = max(det_vals) if det_vals else 0.0

    # Build matroid from nonzero minors
    mat = matroid_from_minors(minors, n, tol=tol)
    positroid = is_positroid(mat) if len(mat.bases) > 0 else True

    return {
        "rank": mat.rank,
        "n_bases": len(mat.bases),
        "is_uniform": mat.is_uniform(),
        "is_positroid": positroid,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "frac_nonneg": (n_pos + n_zero) / total if total else 0.0,
        "min_minor": min_det,
        "max_minor": max_det,
    }


def load_gpt2_with_attention() -> tuple[Any, Any]:
    """Load GPT-2-small with attention output enabled."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2", output_attentions=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def get_attention_patterns(
    model: object,
    tokenizer: object,
    text: str,
    max_len: int,
) -> np.ndarray:
    """Run GPT-2, return (n_layers, n_heads, T, T) attention weights."""
    import torch

    tokens = tokenizer(  # type: ignore[operator]
        text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
    )
    device = next(model.parameters()).device  # type: ignore[attr-defined]
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)  # type: ignore[operator]
    attentions = [a[0].cpu().numpy() for a in outputs.attentions]
    return np.stack(attentions)


def extract_windows(
    attn: np.ndarray,
    q_window: int,
    n_keys: int,
    n_windows: int,
    seed: int,
) -> list[tuple[np.ndarray, int, int]]:
    """Extract rectangular subwindows from a T×T causal attention matrix.

    Ensures all queries can see all keys (no causal mask zeros in subblock).
    """
    rng = np.random.default_rng(seed)
    T = attn.shape[0]
    min_q_start = n_keys  # earliest query that sees n_keys keys at position 0
    max_q_start = T - q_window
    if min_q_start > max_q_start:
        return []

    windows = []
    for _ in range(n_windows):
        q_start = rng.integers(min_q_start, max_q_start + 1)
        max_k_start = q_start - n_keys
        if max_k_start < 0:
            continue
        k_start = rng.integers(0, max_k_start + 1)
        sub = attn[q_start : q_start + q_window, k_start : k_start + n_keys]
        windows.append((sub, q_start, k_start))
    return windows


def run_analysis(
    sub: np.ndarray,
    layer: int,
    head: int,
    input_id: int,
    window_id: int,
    q_start: int,
    k_start: int,
) -> SubmatrixAnalysis:
    """Full analysis of one rectangular attention submatrix."""
    q, n = sub.shape
    info = analyze_submatrix_minors(sub)

    # Attention stats
    row_sums = np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
    probs = np.clip(sub / row_sums, 1e-12, 1.0)
    entropy = float(-np.mean(np.sum(probs * np.log(probs), axis=1)))

    return SubmatrixAnalysis(
        layer=layer,
        head=head,
        input_id=input_id,
        window_id=window_id,
        q_start=q_start,
        k_start=k_start,
        q_window=q,
        n_keys=n,
        matroid_rank=info["rank"],
        n_bases=info["n_bases"],
        is_uniform=info["is_uniform"],
        is_positroid=info["is_positroid"],
        n_minors_positive=info["n_pos"],
        n_minors_negative=info["n_neg"],
        n_minors_zero=info["n_zero"],
        frac_nonneg=info["frac_nonneg"],
        min_minor=info["min_minor"],
        max_minor=info["max_minor"],
        entropy=entropy,
        max_attn=float(np.max(sub)),
        mean_attn=float(np.mean(sub)),
    )


def run_random_baseline(
    q_window: int,
    n_keys: int,
    n_trials: int,
    seed: int = 99,
) -> dict[str, Any]:
    """Positroid and TN check on random nonneg matrices as baseline."""
    rng = np.random.default_rng(seed)
    n_positroid = 0
    n_uniform = 0
    n_tn = 0  # totally nonneg (all maximal minors >= 0)
    frac_nonneg_vals = []
    ranks = []

    for _ in range(n_trials):
        raw = rng.standard_normal((q_window, n_keys))
        sub = np.exp(raw - raw.max(axis=1, keepdims=True))
        sub /= sub.sum(axis=1, keepdims=True)

        info = analyze_submatrix_minors(sub)
        n_positroid += info["is_positroid"]
        n_uniform += info["is_uniform"]
        n_tn += info["n_neg"] == 0
        frac_nonneg_vals.append(info["frac_nonneg"])
        ranks.append(info["rank"])

    return {
        "n_positroid": n_positroid,
        "n_uniform": n_uniform,
        "n_tn": n_tn,
        "n_trials": n_trials,
        "pct_positroid": n_positroid / n_trials,
        "pct_uniform": n_uniform / n_trials,
        "pct_tn": n_tn / n_trials,
        "mean_frac_nonneg": float(np.mean(frac_nonneg_vals)),
        "mean_rank": float(np.mean(ranks)),
    }


TEXTS = {
    "wikipedia": (
        "The amplituhedron is a geometric object introduced in 2013 by Nima "
        "Arkani-Hamed and Jaroslav Trnka. It provides a new way to compute "
        "scattering amplitudes in planar N=4 supersymmetric Yang-Mills theory. "
        "The traditional approach uses Feynman diagrams, which involve summing "
        "over many possible particle interactions. The amplituhedron reformulates "
        "this calculation as a problem in geometry, specifically as the volume of "
        "a higher-dimensional polytope. This connection between physics and geometry "
        "has profound implications for our understanding of space, time, and quantum "
        "mechanics. The totally nonnegative Grassmannian plays a central role in "
        "the construction, connecting it to the theory of positroids and matroid "
        "stratifications studied by mathematicians like Alexander Postnikov."
    ),
    "code": (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n\n"
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n**0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n\n"
        "primes = [x for x in range(100) if is_prime(x)]"
    ),
    "repetitive": (
        "The cat sat on the mat. The dog sat on the log. The cat sat on the mat. "
        "The dog sat on the log. The cat sat on the mat. The dog sat on the log. "
        "The cat sat on the mat. The dog sat on the log. The cat sat on the mat. "
        "The dog sat on the log. The cat sat on the mat. The dog sat on the log."
    ),
    "narrative": (
        "Alice opened the door and found herself in a long, low hall, which was "
        "lit up by a row of lamps hanging from the roof. There were doors all "
        "round the hall, but they were all locked; and when Alice had been all "
        "the way down one side and up the other, trying every door, she walked "
        "sadly down the middle, wondering how she was ever to get out again."
    ),
}


def print_summary(
    results: list[SubmatrixAnalysis],
    baseline: dict[str, Any],
    q_window: int,
    n_keys: int,
) -> None:
    """Print comprehensive summary."""
    print("\n" + "=" * 80)
    print("CONJECTURE 2: ATTENTION PATTERN POSITROID STRUCTURE")
    print(
        f"Submatrix: {q_window} queries × {n_keys} keys  "
        f"(C({n_keys},{q_window})={len(list(combinations(range(n_keys), q_window)))} "
        f"maximal minors)"
    )
    print("=" * 80)

    computed = [r for r in results if not r.errors]

    # --- GPT-2 results ---
    n_pos = sum(1 for r in computed if r.is_positroid)
    n_uni = sum(1 for r in computed if r.is_uniform)
    n_tn = sum(1 for r in computed if r.n_minors_negative == 0)
    pct_pos = n_pos / len(computed) if computed else 0.0
    pct_tn = n_tn / len(computed) if computed else 0.0

    print(f"\n  GPT-2 attention ({len(computed)} submatrices):")
    print(f"    Positroid matroid:     {n_pos}/{len(computed)} ({pct_pos:.1%})")
    print(f"    Uniform matroid:       {n_uni}/{len(computed)} ({n_uni / len(computed):.1%})")
    print(f"    Totally nonneg (TN):   {n_tn}/{len(computed)} ({pct_tn:.1%})")

    frac_nonneg = [r.frac_nonneg for r in computed]
    print(f"    Mean frac nonneg:      {np.mean(frac_nonneg):.3f}")

    ranks = [r.matroid_rank for r in computed]
    print(f"    Matroid rank: mean={np.mean(ranks):.1f}  min={min(ranks)}  max={max(ranks)}")

    # --- Random baseline ---
    print(f"\n  Random nonneg baseline ({baseline['n_trials']} trials):")
    print(
        f"    Positroid matroid:     {baseline['n_positroid']}/{baseline['n_trials']} "
        f"({baseline['pct_positroid']:.1%})"
    )
    print(
        f"    Uniform matroid:       {baseline['n_uniform']}/{baseline['n_trials']} "
        f"({baseline['pct_uniform']:.1%})"
    )
    print(
        f"    Totally nonneg (TN):   {baseline['n_tn']}/{baseline['n_trials']} "
        f"({baseline['pct_tn']:.1%})"
    )
    print(f"    Mean frac nonneg:      {baseline['mean_frac_nonneg']:.3f}")
    print(f"    Mean rank:             {baseline['mean_rank']:.1f}")

    # --- Comparison ---
    print(f"\n  Comparison:")
    delta_pos = pct_pos - baseline["pct_positroid"]
    delta_tn = pct_tn - baseline["pct_tn"]
    print(
        f"    Positroid: GPT-2={pct_pos:.1%} vs random={baseline['pct_positroid']:.1%}"
        f"  (delta={delta_pos:+.1%})"
    )
    print(
        f"    TN:        GPT-2={pct_tn:.1%} vs random={baseline['pct_tn']:.1%}"
        f"  (delta={delta_tn:+.1%})"
    )

    # --- Layer breakdown ---
    print("\n" + "-" * 60)
    print("LAYER-BY-LAYER")
    print("-" * 60)
    layers = sorted(set(r.layer for r in computed))
    for layer in layers:
        lr = [r for r in computed if r.layer == layer]
        lp = sum(1 for r in lr if r.is_positroid)
        lt = sum(1 for r in lr if r.n_minors_negative == 0)
        lu = sum(1 for r in lr if r.is_uniform)
        fn = np.mean([r.frac_nonneg for r in lr])
        rk = np.mean([r.matroid_rank for r in lr])
        ent = np.mean([r.entropy for r in lr])
        print(
            f"  L{layer:>2d}: pos={lp:>3d}/{len(lr):<3d} ({lp / len(lr):.0%})  "
            f"TN={lt:>3d}  uni={lu:>3d}  "
            f"frac_nn={fn:.2f}  rank={rk:.1f}  ent={ent:.2f}"
        )

    # --- Per-head detail ---
    print("\n" + "-" * 60)
    print("PER-HEAD POSITROID RATE (across inputs × windows)")
    print("-" * 60)
    heads = sorted(set(r.head for r in computed))
    for layer in layers:
        parts = []
        for head in heads:
            hr = [r for r in computed if r.layer == layer and r.head == head]
            if not hr:
                continue
            hp = sum(1 for r in hr if r.is_positroid)
            parts.append(f"H{head}:{hp}/{len(hr)}")
        print(f"  L{layer:>2d}: {' '.join(parts)}")

    # --- Per-input ---
    print("\n" + "-" * 60)
    print("PER-INPUT")
    print("-" * 60)
    input_ids = sorted(set(r.input_id for r in computed))
    for iid in input_ids:
        ir = [r for r in computed if r.input_id == iid]
        ip = sum(1 for r in ir if r.is_positroid)
        it = sum(1 for r in ir if r.n_minors_negative == 0)
        print(
            f"  Input {iid}: positroid={ip}/{len(ir)} ({ip / len(ir):.1%})  "
            f"TN={it}/{len(ir)} ({it / len(ir):.1%})"
        )

    # --- Verdict ---
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    print(
        f"\n  Positroid rate:  GPT-2={pct_pos:.1%}  random={baseline['pct_positroid']:.1%}"
        f"  delta={delta_pos:+.1%}"
    )
    print(
        f"  TN rate:         GPT-2={pct_tn:.1%}  random={baseline['pct_tn']:.1%}"
        f"  delta={delta_tn:+.1%}"
    )
    print(
        f"  Frac nonneg:     GPT-2={np.mean(frac_nonneg):.3f}"
        f"  random={baseline['mean_frac_nonneg']:.3f}"
    )

    if pct_pos > 0.9 and delta_pos > 0.1:
        print("\n  STRONG SUPPORT for Conjecture 2")
    elif pct_pos > baseline["pct_positroid"] + 0.1:
        print("\n  MODERATE SUPPORT — GPT-2 is more positroid than random")
    elif abs(delta_pos) < 0.05:
        print("\n  INCONCLUSIVE — GPT-2 matches random baseline")
    elif delta_pos < -0.1:
        print("\n  NO SUPPORT — GPT-2 is LESS positroid than random")
    else:
        print("\n  WEAK/NO SUPPORT for Conjecture 2")

    if pct_tn > 0.5 and delta_tn > 0.1:
        print("  TOTAL NONNEGATIVITY signal detected!")
    elif pct_tn > baseline["pct_tn"] + 0.05:
        print("  Slight TN signal (above random baseline)")
    else:
        print("  No total nonnegativity signal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Conjecture 2: Attention Pattern Positroid Structure"
    )
    parser.add_argument(
        "--q-window", type=int, default=4, help="Query rows per submatrix (default: 4)"
    )
    parser.add_argument(
        "--n-keys", type=int, default=16, help="Key columns per submatrix (default: 16)"
    )
    parser.add_argument(
        "--n-windows", type=int, default=5, help="Windows per (head, input) pair (default: 5)"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None, help="Layers to analyze (default: all 12)"
    )
    parser.add_argument(
        "--baseline-trials", type=int, default=500, help="Random baseline trials (default: 500)"
    )
    parser.add_argument("--texts", nargs="+", default=None, help="Text keys (default: all)")
    args = parser.parse_args()

    from math import comb

    n_minors = comb(args.n_keys, args.q_window)
    print("=" * 60)
    print("Conjecture 2: Attention Pattern Positroid Structure")
    print(f"Config: q={args.q_window} queries × n={args.n_keys} keys")
    print(f"  C({args.n_keys},{args.q_window})={n_minors} maximal minors per submatrix")
    print(f"  T={args.seq_len}, {args.n_windows} windows/head")
    print("=" * 60)
    sys.stdout.flush()

    # Random baseline
    print(f"\nRunning random nonneg baseline ({args.baseline_trials} trials)...", flush=True)
    baseline = run_random_baseline(
        args.q_window,
        args.n_keys,
        args.baseline_trials,
    )
    print(
        f"  Done: {baseline['pct_positroid']:.1%} positroid, "
        f"{baseline['pct_tn']:.1%} TN, rank={baseline['mean_rank']:.1f}",
        flush=True,
    )

    # Load model
    print("\nLoading GPT-2-small...", flush=True)
    model, tokenizer = load_gpt2_with_attention()
    print("  Done.", flush=True)

    text_keys = args.texts or list(TEXTS.keys())
    layers_to_check = args.layers
    all_results: list[SubmatrixAnalysis] = []

    for input_id, text_key in enumerate(text_keys):
        text = TEXTS[text_key]
        print(f"\nInput {input_id}: '{text_key}'", flush=True)
        attn_all = get_attention_patterns(model, tokenizer, text, args.seq_len)
        actual_T = attn_all.shape[2]
        n_layers, n_heads = attn_all.shape[:2]
        print(f"  T={actual_T}, {n_layers}L × {n_heads}H", flush=True)

        layers = layers_to_check or list(range(n_layers))

        for layer in layers:
            if layer >= n_layers:
                continue
            for head in range(n_heads):
                attn = attn_all[layer, head]
                windows = extract_windows(
                    attn,
                    args.q_window,
                    args.n_keys,
                    args.n_windows,
                    seed=input_id * 1000 + layer * 100 + head,
                )

                tags = []
                tn_tags = []
                for w_id, (sub, q_start, k_start) in enumerate(windows):
                    result = run_analysis(
                        sub,
                        layer,
                        head,
                        input_id,
                        w_id,
                        q_start,
                        k_start,
                    )
                    all_results.append(result)
                    tags.append("P" if result.is_positroid else ".")
                    tn_tags.append("+" if result.n_minors_negative == 0 else "-")

                pos_str = "".join(tags)
                tn_str = "".join(tn_tags)
                n_p = sum(1 for t in tags if t == "P")
                n_t = sum(1 for t in tn_tags if t == "+")
                print(
                    f"    L{layer}H{head:>2d}: pos=[{pos_str}]={n_p}  tn=[{tn_str}]={n_t}",
                    flush=True,
                )

    print_summary(all_results, baseline, args.q_window, args.n_keys)


if __name__ == "__main__":
    main()
