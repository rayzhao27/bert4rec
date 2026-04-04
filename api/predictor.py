from __future__ import annotations

"""
api/predictor.py
────────────────
Loads the BERT4Rec checkpoint once at startup and exposes a
thread-safe predict() method used by the FastAPI route handlers.

The Predictor is instantiated as a module-level singleton and
attached to app.state during the FastAPI lifespan so it is shared
across all requests without reloading the model.
"""

import json
import logging
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.bert4rec import BERT4Rec, build_model

logger = logging.getLogger(__name__)


class Predictor:
    """
    Wraps BERT4Rec for inference.

    Args:
        checkpoint_path: path to best_model.pt saved by the trainer.
        data_dir:        project data directory (used to load stats.json).
        device:          torch device string or torch.device.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_dir:        str | Path,
        device:          str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir        = Path(data_dir)

        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self._model:         BERT4Rec | None = None
        self._mask_token_id: int             = 0
        self._vocab_size:    int             = 0
        self._model_version: str             = "unknown"

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model checkpoint into memory.  Call once at app startup."""
        logger.info("Loading checkpoint from %s …", self.checkpoint_path)

        ckpt  = torch.load(self.checkpoint_path, map_location=self.device)
        cfg   = ckpt["cfg"]

        # Resolve vocab_size from stats.json (same fix as evaluator)
        stats_path = self.data_dir / "processed" / "stats.json"
        stats      = json.loads(stats_path.read_text())
        cfg["vocab_size"] = stats["num_items"] + 2   # PAD + items + MASK

        self._mask_token_id = cfg["vocab_size"] - 1
        self._vocab_size    = cfg["vocab_size"]
        self._model_version = f"epoch_{ckpt['epoch']}"

        self._model = build_model(cfg).to(self.device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()

        logger.info(
            "Model ready  |  device=%s  |  vocab=%d  |  version=%s",
            self.device, self._vocab_size, self._model_version,
        )

    def unload(self) -> None:
        """Release model from memory.  Call at app shutdown."""
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict(
        self,
        user_history: list[int],
        top_k:        int  = 10,
        max_seq_len:  int  = 200,
    ) -> list[tuple[int, float]]:
        """
        Generate top-K recommendations for a single user.

        Args:
            user_history: chronological list of item ids (most recent last).
            top_k:        number of items to return.
            max_seq_len:  maximum sequence length (must match training config).

        Returns:
            List of (item_id, score) tuples sorted by score descending.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # ── Build input tensor ────────────────────────────────────────────
        # Truncate to max_seq_len - 1 to leave room for [MASK] at the end
        history = user_history[-(max_seq_len - 1):]

        # Left-pad with zeros to max_seq_len
        pad_len   = max_seq_len - len(history) - 1
        input_seq = [0] * pad_len + history + [self._mask_token_id]

        input_ids = torch.tensor(
            [input_seq], dtype=torch.long, device=self.device
        )   # [1, max_seq_len]

        # ── Forward pass ──────────────────────────────────────────────────
        padding_mask = (input_ids == 0)
        logits       = self._model(input_ids, padding_mask)   # [1, L, V]

        # Extract logits at the [MASK] position (last position)
        mask_logits = logits[0, -1, :]   # [V]

        # ── Filter seen items ─────────────────────────────────────────────
        seen = set(history)
        seen.discard(0)
        for item_id in seen:
            if item_id < self._vocab_size:
                mask_logits[item_id] = float("-inf")

        # ── Top-K ─────────────────────────────────────────────────────────
        k          = min(top_k, self._vocab_size)
        scores, ids = torch.topk(mask_logits, k=k)

        return list(zip(ids.cpu().tolist(), scores.cpu().tolist()))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device_name(self) -> str:
        return str(self.device)
