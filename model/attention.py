from __future__ import annotations

"""
model/attention.py
──────────────────
A single BERT-style transformer block:

    x → MHA(x, x, x, key_padding_mask) → residual → LayerNorm
      → FFN(x)                          → residual → LayerNorm

Uses pre-LN (layer norm applied before each sub-layer) for more
stable training, especially with small datasets like MovieLens 1M.

The block is bidirectional — NO causal/look-ahead mask is applied,
so every position can attend to every other position in both directions.
PAD positions are excluded via key_padding_mask.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Linear(d → d_ff) → GELU → Dropout → Linear(d_ff → d) → Dropout

    Args:
        hidden_size:       model dimension d.
        intermediate_size: inner dimension d_ff (typically 4 × hidden_size).
        dropout:           dropout probability.
    """

    def __init__(
        self,
        hidden_size:       int,
        intermediate_size: int,
        dropout:           float = 0.1,
    ) -> None:
        super().__init__()
        self.dense_in  = nn.Linear(hidden_size, intermediate_size)
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_in(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    One BERT4Rec transformer block (pre-LN variant).

    Pre-LN layout:
        h  = x + MHA( LayerNorm(x) )
        h' = h + FFN( LayerNorm(h) )

    Compared to the post-LN layout used in the original BERT paper,
    pre-LN avoids the gradient explosion risk at the start of training
    and generally needs no warm-up schedule.

    Args:
        hidden_size:               model dimension d.
        num_attention_heads:       number of attention heads H.
        intermediate_size:         FFN inner dimension (default: 4 × d).
        hidden_dropout_prob:       dropout on FFN output and residuals.
        attention_probs_dropout:   dropout on attention weights.
    """

    def __init__(
        self,
        hidden_size:              int,
        num_attention_heads:      int,
        intermediate_size:        int,
        hidden_dropout_prob:      float = 0.1,
        attention_probs_dropout:  float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head self-attention
        # batch_first=True → input/output shape is [B, L, d]
        self.attention = nn.MultiheadAttention(
            embed_dim    = hidden_size,
            num_heads    = num_attention_heads,
            dropout      = attention_probs_dropout,
            batch_first  = True,
        )

        # Feed-forward network
        self.ffn = FeedForward(
            hidden_size       = hidden_size,
            intermediate_size = intermediate_size,
            dropout           = hidden_dropout_prob,
        )

        # Layer norms (pre-LN: applied before each sub-layer)
        self.ln_attn = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ln_ffn  = nn.LayerNorm(hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        x:               Tensor,
        padding_mask:    Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:             FloatTensor [B, L, d]
            padding_mask:  BoolTensor  [B, L]  — True where token is PAD.
                           Passed as key_padding_mask so PAD positions are
                           excluded from attention score computation.

        Returns:
            Tensor [B, L, d]
        """
        # ── 1. Pre-LN multi-head self-attention ──────────────────────────────
        residual = x
        x_norm   = self.ln_attn(x)

        attn_out, _ = self.attention(
            query           = x_norm,
            key             = x_norm,
            value           = x_norm,
            key_padding_mask= padding_mask,  # [B, L] bool, True = ignore
            need_weights    = False,
        )

        x = residual + self.dropout(attn_out)

        # ── 2. Pre-LN feed-forward ───────────────────────────────────────────
        residual = x
        x        = residual + self.ffn(self.ln_ffn(x))

        return x   # [B, L, d]
