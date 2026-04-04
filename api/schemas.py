from __future__ import annotations

"""
api/schemas.py
──────────────
Pydantic request and response models for the BERT4Rec inference API.
"""

from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    """
    Request body for POST /recommend.

    Args:
        user_history: chronological list of item ids the user has interacted
                      with, most recent last.  At least 1 item required.
        top_k:        number of recommendations to return (1–100, default 10).
    """

    user_history: list[int] = Field(
        ...,
        min_length=1,
        description="Chronological list of item ids (most recent last).",
        examples=[[42, 17, 88, 5, 231]],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return.",
    )

    @field_validator("user_history")
    @classmethod
    def items_must_be_positive(cls, v: list[int]) -> list[int]:
        if any(i <= 0 for i in v):
            raise ValueError("All item ids must be positive integers (> 0).")
        return v


class RecommendedItem(BaseModel):
    """A single recommended item with its score."""
    item_id: int   = Field(..., description="Recommended item id.")
    score:   float = Field(..., description="Raw logit score (higher = more relevant).")


class RecommendResponse(BaseModel):
    """
    Response body for POST /recommend.
    """
    recommendations: list[RecommendedItem] = Field(
        ..., description="Top-K recommended items, ranked by score descending."
    )
    model_version:   str = Field(..., description="Checkpoint epoch used for inference.")
    num_input_items: int = Field(..., description="Number of items in the input history.")


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status:        str  = Field(..., examples=["ok"])
    model_loaded:  bool = Field(..., description="True if model checkpoint is loaded.")
    vocab_size:    int  = Field(..., description="Model vocabulary size.")
    model_version: str  = Field(..., description="Checkpoint epoch.")
    device:        str  = Field(..., description="Inference device (cpu / mps / cuda).")
