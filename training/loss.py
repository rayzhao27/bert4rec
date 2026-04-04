from __future__ import annotations

"""
training/loss.py
────────────────
Masked-item prediction loss for BERT4Rec.

CrossEntropyLoss with ignore_index=-100 so that only the positions
selected for masking contribute to the gradient — all other positions
(real items and PAD) are skipped automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedItemLoss(nn.Module):
    """
    Cross-entropy loss over the item vocabulary, evaluated only at
    positions where labels != IGNORE_ID (-100).

    Args:
        vocab_size:   total vocabulary size (used for shape assertion).
        ignore_index: label value that marks non-masked positions (default -100).
        label_smoothing: optional label smoothing in [0, 1) (default 0.0).
    """

    IGNORE_ID: int = -100

    def __init__(
        self,
        vocab_size:      int,
        ignore_index:    int   = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size   = vocab_size
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            ignore_index    = ignore_index,
            label_smoothing = label_smoothing,
            reduction       = "mean",
        )

    def forward(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, dict]:
        """
        Args:
            logits:  FloatTensor [B, L, V]  — raw model output.
            labels:  LongTensor  [B, L]     — target item ids at masked
                     positions; IGNORE_ID everywhere else.

        Returns:
            loss:    scalar Tensor
            metrics: dict with 'loss', 'n_masked', 'ppl' (perplexity)
        """
        B, L, V = logits.shape

        # Flatten to [B*L, V] and [B*L] for CrossEntropyLoss
        loss = self.ce(logits.view(B * L, V), labels.view(B * L))

        # Count active (non-ignored) positions for logging
        n_masked = (labels != self.ignore_index).sum().item()
        ppl      = torch.exp(loss).item() if loss.isfinite() else float("inf")

        return loss, {
            "loss":     loss.item(),
            "ppl":      ppl,
            "n_masked": n_masked,
        }


def masked_accuracy(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> float:
    """
    Top-1 accuracy over masked positions only — useful as a sanity-check
    metric during training (not an official rec metric).

    Args:
        logits:       FloatTensor [B, L, V]
        labels:       LongTensor  [B, L]
        ignore_index: label value to skip (default -100)

    Returns:
        accuracy in [0, 1]
    """
    with torch.no_grad():
        preds  = logits.argmax(dim=-1)          # [B, L]
        mask   = labels != ignore_index          # [B, L] bool
        if mask.sum() == 0:
            return 0.0
        correct = (preds[mask] == labels[mask]).float().sum()
        return (correct / mask.sum()).item()
