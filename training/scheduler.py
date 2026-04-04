from __future__ import annotations

"""
training/scheduler.py
──────────────────────
Learning-rate schedule: linear warm-up → cosine decay.

    lr
    ▲
    │       /‾‾‾‾‾\
    │      /        \___
    │     /              ‾‾‾___
    │    /                      ‾‾‾___  ← lr_min
    │___/
    └──────────────────────────────────► step
       ↑warmup_steps

Two helpers are provided:

  get_scheduler()   — wraps PyTorch's LambdaLR; drop-in with any optimizer.
  WarmupCosineScheduler — a standalone nn.Module version if you prefer
                          to manage the schedule yourself.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _warmup_cosine_lambda(
    current_step:   int,
    warmup_steps:   int,
    total_steps:    int,
    lr_min_ratio:   float,
) -> float:
    """
    Returns the LR multiplier for *current_step*.

    Phases:
        [0, warmup_steps)          — linear ramp from 0 → 1
        [warmup_steps, total_steps) — cosine decay from 1 → lr_min_ratio
    """
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)

    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min_ratio + (1.0 - lr_min_ratio) * cosine


def get_scheduler(
    optimizer:     Optimizer,
    warmup_steps:  int,
    total_steps:   int,
    lr_min_ratio:  float = 0.1,
    last_epoch:    int   = -1,
) -> LambdaLR:
    """
    Build a LambdaLR scheduler with linear warm-up + cosine decay.

    Args:
        optimizer:    the optimizer whose LR this scheduler controls.
        warmup_steps: number of steps to linearly ramp up the LR.
        total_steps:  total training steps (epochs × steps_per_epoch).
        lr_min_ratio: floor of the cosine decay expressed as a fraction
                      of the peak LR  (e.g. 0.1 → decays to 10% of peak).
        last_epoch:   used for checkpoint resumption (default -1 = fresh start).

    Returns:
        LambdaLR scheduler — call .step() once per optimizer step.

    Usage:
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = get_scheduler(optimizer, warmup_steps=100, total_steps=4800)
        ...
        loss.backward()
        optimizer.step()
        scheduler.step()
    """
    fn = lambda step: _warmup_cosine_lambda(   # noqa: E731
        step, warmup_steps, total_steps, lr_min_ratio
    )
    return LambdaLR(optimizer, lr_lambda=fn, last_epoch=last_epoch)


def compute_total_steps(epochs: int, steps_per_epoch: int) -> int:
    """Convenience helper so callers never forget the multiplication."""
    return epochs * steps_per_epoch
