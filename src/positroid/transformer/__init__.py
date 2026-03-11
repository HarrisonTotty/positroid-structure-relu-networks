"""Positroid transformer architecture components.

Implements five proposals from the Cognihedron at Scale brainstorming document:
- Proposal A: Positroid Attention (positroid_attention.py)
- Proposal B: Positroid MoE (positroid_moe.py)
- Proposal C: Tropical MLP (tropical_mlp.py)
- Proposal D: Positroid LoRA (positroid_lora.py)
- Proposal E: Empirical Analysis (analysis.py)
- Integration: model.py
"""

from positroid.transformer.positroid_attention import (
    PositroidAttentionHead,
    PositroidMultiHeadAttention,
)
from positroid.transformer.tropical_mlp import TropicalMLP
from positroid.transformer.positroid_moe import PositroidMoE, PositroidRouter
from positroid.transformer.positroid_lora import PositroidLoRA
from positroid.transformer.model import (
    StandardAttentionHead,
    StandardMultiHeadAttention,
    TransformerBlock,
    PositroidTransformerBlock,
    PositroidClassifier,
)

__all__ = [
    "PositroidAttentionHead",
    "PositroidMultiHeadAttention",
    "StandardAttentionHead",
    "StandardMultiHeadAttention",
    "TransformerBlock",
    "TropicalMLP",
    "PositroidRouter",
    "PositroidMoE",
    "PositroidLoRA",
    "PositroidTransformerBlock",
    "PositroidClassifier",
]
