"""
data/dataset.py
───────────────
PyTorch Dataset and DataLoader factory for BERT4Rec.

Three dataset modes mirror the leave-one-out split:

    TRAIN  mode  ─  random BERT-style masking over seq[:-2]
    VAL    mode  ─  mask only the last item of seq[:-1] (fixed evaluation)
    TEST   mode  ─  mask only the last item of seq      (fixed evaluation)

Masking strategy (identical to the original BERT4Rec paper, Sun et al. 2019):
    • With probability mask_prob, replace an item with [MASK].
    • With probability 0.1 of the masked set, replace with a random item.
    • With probability 0.1 of the masked set, keep the original item.
    • The remaining 80 % are replaced with [MASK].
    (Only applied during training; evaluation always masks the final position.)

Special tokens:
    PAD   = 0                    padding token
    MASK  = num_items + 1        [MASK] token  (always the last vocab slot)

Output batch tensors (all shape [B, max_seq_len]):
    input_ids      padded + masked item sequence
    labels         original item ids at masked positions, -100 elsewhere
    padding_mask   1 where token is PAD (for attention key_padding_mask)
"""

from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import joblib

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ── Token constants ────────────────────────────────────────────────────────────
PAD_ID   = 0
IGNORE_ID = -100   # label value ignored by CrossEntropyLoss


class Split(str, Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


# ── Dataset ───────────────────────────────────────────────────────────────────

class BERT4RecDataset(Dataset):
    """
    Args:
        sequences:    dict mapping user_id → list[item_id] (1-based).
        num_items:    total number of distinct items (PAD=0, MASK=num_items+1).
        max_seq_len:  maximum sequence length (sequences are right-truncated
                      then left-padded to this length).
        mask_prob:    fraction of positions randomly masked (training only).
        split:        Split.TRAIN | Split.VAL | Split.TEST
        seed:         random seed for reproducible masking (training).
    """

    def __init__(
        self,
        sequences:   dict[int, list[int]],
        num_items:   int,
        max_seq_len: int  = 200,
        mask_prob:   float = 0.2,
        split:       Split = Split.TRAIN,
        seed:        int   = 42,
    ) -> None:
        self.sequences   = list(sequences.values())
        self.user_ids    = list(sequences.keys())
        self.num_items   = num_items
        self.max_seq_len = max_seq_len
        self.mask_prob   = mask_prob
        self.split       = split
        self.mask_id     = num_items + 1   # [MASK] token id
        self._rng        = random.Random(seed)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _truncate_and_pad(self, seq: list[int]) -> list[int]:
        """Right-truncate to max_seq_len, then left-pad with PAD_ID."""
        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        return [PAD_ID] * pad_len + seq

    def _mask_train(self, seq: list[int]) -> tuple[list[int], list[int]]:
        """
        Apply BERT-style random masking to *seq* (training mode).
        Returns (masked_seq, labels) where labels[i] = original item if masked,
        else IGNORE_ID.
        """
        masked = seq.copy()
        labels = [IGNORE_ID] * len(seq)

        for i, item in enumerate(seq):
            if item == PAD_ID:
                continue
            if self._rng.random() < self.mask_prob:
                labels[i] = item
                r = self._rng.random()
                if r < 0.8:
                    masked[i] = self.mask_id                              # 80 % → [MASK]
                elif r < 0.9:
                    masked[i] = self._rng.randint(1, self.num_items)     # 10 % → random item
                # else: 10 % → keep original (masked[i] unchanged)

        return masked, labels

    def _mask_eval(self, seq: list[int]) -> tuple[list[int], list[int]]:
        """
        Mask only the final non-PAD position (val / test mode).
        The model must predict the target item at that single position.
        """
        masked = seq.copy()
        labels = [IGNORE_ID] * len(seq)

        # Find last real (non-PAD) position
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != PAD_ID:
                labels[i] = seq[i]
                masked[i] = self.mask_id
                break

        return masked, labels

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raw_seq = self.sequences[idx]
        seq     = self._truncate_and_pad(raw_seq)

        if self.split == Split.TRAIN:
            input_ids, labels = self._mask_train(seq)
        else:
            input_ids, labels = self._mask_eval(seq)

        input_ids_t  = torch.tensor(input_ids, dtype=torch.long)
        labels_t     = torch.tensor(labels,    dtype=torch.long)
        padding_mask = (input_ids_t == PAD_ID)   # True where padded

        return {
            "input_ids":    input_ids_t,
            "labels":       labels_t,
            "padding_mask": padding_mask,
            "user_id":      torch.tensor(self.user_ids[idx], dtype=torch.long),
        }


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    processed_dir: str | Path,
    max_seq_len:   int   = 200,
    mask_prob:     float = 0.2,
    batch_size:    int   = 256,
    num_workers:   int   = 4,
    seed:          int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """
    Load preprocessed sequences and return (train_loader, val_loader,
    test_loader, stats) ready for training.

    Args:
        processed_dir:  path to data/processed/ (output of preprocess.py).
        max_seq_len:    maximum sequence length; truncates + left-pads.
        mask_prob:      masking probability used during training.
        batch_size:     samples per mini-batch.
        num_workers:    DataLoader worker processes.
        seed:           reproducibility seed.

    Returns:
        train_loader, val_loader, test_loader, stats dict.
    """
    processed_dir = Path(processed_dir)

    # ── load sequences ──
    train_seqs: dict[int, list[int]] = joblib.load(processed_dir / "train_seqs.pkl")
    val_seqs:   dict[int, list[int]] = joblib.load(processed_dir / "val_seqs.pkl")
    test_seqs:  dict[int, list[int]] = joblib.load(processed_dir / "test_seqs.pkl")
    item_enc                         = joblib.load(processed_dir / "item_encoder.pkl")

    num_items = len(item_enc.classes_)   # does NOT include PAD or MASK

    common = dict(num_items=num_items, max_seq_len=max_seq_len, seed=seed)

    train_ds = BERT4RecDataset(train_seqs, mask_prob=mask_prob, split=Split.TRAIN, **common)
    val_ds   = BERT4RecDataset(val_seqs,   mask_prob=mask_prob, split=Split.VAL,   **common)
    test_ds  = BERT4RecDataset(test_seqs,  mask_prob=mask_prob, split=Split.TEST,  **common)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    stats = {
        "num_items":    num_items,
        "vocab_size":   num_items + 2,    # PAD + items + MASK
        "mask_token_id": num_items + 1,
        "num_train_users": len(train_seqs),
        "num_val_users":   len(val_seqs),
        "num_test_users":  len(test_seqs),
        "train_batches":   len(train_loader),
    }

    return train_loader, val_loader, test_loader, stats


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")

    train_loader, val_loader, test_loader, stats = build_dataloaders(
        processed_dir="data/processed",
        max_seq_len=200,
        mask_prob=0.2,
        batch_size=256,
    )

    print("\n── Dataset stats ─────────────────────────────────────")
    print(json.dumps(stats, indent=2))

    # Inspect a single batch
    batch = next(iter(train_loader))
    print("\n── Train batch shapes ────────────────────────────────")
    for k, v in batch.items():
        print(f"  {k:15s}  {tuple(v.shape)}  dtype={v.dtype}")

    # Confirm [MASK] tokens appear in input_ids
    mask_id      = stats["mask_token_id"]
    n_masked     = (batch["input_ids"] == mask_id).sum().item()
    n_labels     = (batch["labels"]    != IGNORE_ID).sum().item()
    n_pad        = batch["padding_mask"].sum().item()
    print(f"\n  [MASK] tokens in batch : {n_masked}")
    print(f"  labelled positions     : {n_labels}")
    print(f"  [PAD]  tokens in batch : {n_pad}")

    assert n_masked <= n_labels, "More [MASK] tokens than labelled positions!"
    mask_ratio = n_masked / n_labels
    assert 0.70 <= mask_ratio <= 0.90, f"Unexpected mask ratio: {mask_ratio:.2f} (expected ~0.80)"

    print(f"\n  [MASK] covers {mask_ratio:.1%} of labelled positions (expect ~80%)")
    print("\n  Smoke-test passed ✓")
