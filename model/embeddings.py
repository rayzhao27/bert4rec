from __future__ import annotations

"""
model/embeddings.py
───────────────────
Embedding layer for BERT4Rec.

Combines two learned lookup tables:
    item_embeddings   E  ∈ ℝ^(vocab_size × hidden_size)
    position_embeddings P ∈ ℝ^(max_seq_len × hidden_size)

Output: LayerNorm( E[item_ids] + P[positions] ) → Dropout
"""

import torch
import torch.nn as nn
from torch import Tensor


class BERTEmbeddings(nn.Module):
    """
    Args:
        vocab_size:    total token vocabulary (num_items + 2 for PAD and MASK).
        hidden_size:   embedding / model dimension d.
        max_seq_len:   maximum sequence length (positional table size).
        dropout:       dropout probability applied after layer norm.
        pad_token_id:  token id treated as padding (default 0).
    """

    def __init__(
        self,
        vocab_size:   int,
        hidden_size:  int,
        max_seq_len:  int,
        dropout:      float = 0.1,
        pad_token_id: int   = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id

        # Learned item embedding table — PAD token will be zeroed via padding_idx
        self.item_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_token_id,
        )

        # Learned absolute position embeddings
        # Position 0 is reserved for PAD slots; real positions start at 1
        self.position_embeddings = nn.Embedding(max_seq_len + 1, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids:  LongTensor [B, L]  — item id sequence (0 = PAD).

        Returns:
            Tensor [B, L, hidden_size]
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── item embeddings ──────────────────────────────────────────────────
        item_emb = self.item_embeddings(input_ids)   # [B, L, d]

        # ── positional embeddings ────────────────────────────────────────────
        # Build a position index tensor [1, L] with 0 wherever the token is PAD
        # so PAD positions receive the zero-embedding from position 0.
        positions = torch.arange(1, L + 1, device=device).unsqueeze(0)  # [1, L]
        pad_mask  = (input_ids == self.pad_token_id)                      # [B, L]
        positions = positions.expand(B, -1).masked_fill(pad_mask, 0)     # [B, L]
        pos_emb   = self.position_embeddings(positions)                   # [B, L, d]

        # ── combine ──────────────────────────────────────────────────────────
        embeddings = item_emb + pos_emb               # [B, L, d]
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
