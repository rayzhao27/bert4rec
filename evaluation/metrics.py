from __future__ import annotations

"""
evaluation/metrics.py
──────────────────────
Hit Rate and NDCG at K for sequential recommendation.

Both metrics follow the standard leave-one-out evaluation protocol:
  • One target item per user (the held-out last interaction)
  • Model ranks ALL items it hasn't seen (full-corpus ranking)
  • Metrics are averaged across all users

Definitions
───────────
HR@K (Hit Rate at K)
    Fraction of users for whom the target item appears in the top-K list.
    Binary per user — either the target is in the list or it isn't.

    HR@K = (# users with target in top-K) / (# users)

NDCG@K (Normalised Discounted Cumulative Gain at K)
    Rewards finding the target higher in the ranked list.
    Position 1 scores 1/log2(2)=1.0; position K scores 1/log2(K+1).

    DCG@K  = 1 / log2(rank + 1)   if target is in top-K, else 0
    NDCG@K = DCG@K / IDCG@K       IDCG = 1.0  (perfect rank = 1)

MRR (Mean Reciprocal Rank)
    Average of 1/rank across users where target appears in top-K.
    Included as a bonus metric — not in the original BERT4Rec paper
    but commonly reported alongside HR and NDCG.
"""

import math
from dataclasses import dataclass, field


@dataclass
class MetricAccumulator:
    """
    Running accumulator for HR@K, NDCG@K, and MRR across batches.

    Usage:
        acc = MetricAccumulator(k_values=[5, 10, 20])
        acc.update(scores, target_ids)   # call once per batch
        results = acc.compute()          # call once at end of epoch
    """

    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._hits:   dict[int, float] = {k: 0.0 for k in self.k_values}
        self._ndcg:   dict[int, float] = {k: 0.0 for k in self.k_values}
        self._mrr:    dict[int, float] = {k: 0.0 for k in self.k_values}
        self._n_users: int = 0

    def update(
        self,
        ranked_lists: list[list[int]],
        targets:      list[int],
    ) -> None:
        """
        Accumulate metrics for one batch of users.

        Args:
            ranked_lists: list of B ranked item-id lists, highest score first.
                          Each inner list should contain at least max(k_values)
                          elements (remaining items after filtering seen items).
            targets:      list of B ground-truth item ids (one per user).
        """
        for ranked, target in zip(ranked_lists, targets):
            self._n_users += 1
            for k in self.k_values:
                top_k = ranked[:k]
                if target in top_k:
                    self._hits[k] += 1.0
                    rank = top_k.index(target) + 1          # 1-indexed
                    self._ndcg[k] += 1.0 / math.log2(rank + 1)
                    self._mrr[k]  += 1.0 / rank

    def compute(self) -> dict[str, float]:
        """
        Return averaged metrics over all accumulated users.

        Returns dict with keys like 'HR@5', 'NDCG@5', 'MRR@5', etc.
        """
        if self._n_users == 0:
            return {}
        n = self._n_users
        results: dict[str, float] = {}
        for k in self.k_values:
            results[f"HR@{k}"]   = self._hits[k] / n
            results[f"NDCG@{k}"] = self._ndcg[k] / n
            results[f"MRR@{k}"]  = self._mrr[k]  / n
        return results


# ── Functional API (single-sample, useful for unit tests) ─────────────────────

def hit_rate_at_k(ranked: list[int], target: int, k: int) -> float:
    """1.0 if target is in ranked[:k], else 0.0."""
    return 1.0 if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[int], target: int, k: int) -> float:
    """1/log2(rank+1) if target is in ranked[:k], else 0.0."""
    top_k = ranked[:k]
    if target not in top_k:
        return 0.0
    rank = top_k.index(target) + 1   # 1-indexed
    return 1.0 / math.log2(rank + 1)


def mrr_at_k(ranked: list[int], target: int, k: int) -> float:
    """1/rank if target is in ranked[:k], else 0.0."""
    top_k = ranked[:k]
    if target not in top_k:
        return 0.0
    return 1.0 / (top_k.index(target) + 1)
