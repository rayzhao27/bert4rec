from __future__ import annotations

"""
api/routes.py
─────────────
FastAPI route handlers for the BERT4Rec recommendation API.

Endpoints:
    GET  /health       — liveness check + model status
    POST /recommend    — generate top-K recommendations for a user
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns model status and configuration. Use this to verify the service is up.",
)
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    return HealthResponse(
        status        = "ok",
        model_loaded  = predictor.is_loaded,
        vocab_size    = predictor.vocab_size,
        model_version = predictor.model_version,
        device        = predictor.device_name,
    )


# ── POST /recommend ───────────────────────────────────────────────────────────

@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get top-K recommendations",
    description=(
        "Given a user's interaction history (chronological list of item ids), "
        "returns the top-K recommended items the user is most likely to interact "
        "with next.  Items already in the history are excluded from results."
    ),
)
def recommend(request: Request, body: RecommendRequest) -> RecommendResponse:
    predictor = request.app.state.predictor

    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    try:
        results = predictor.predict(
            user_history = body.user_history,
            top_k        = body.top_k,
        )
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "recommend  |  input_len=%d  top_k=%d  latency=%.1fms",
        len(body.user_history), body.top_k, latency_ms,
    )

    return RecommendResponse(
        recommendations = [
            RecommendedItem(item_id=item_id, score=round(score, 4))
            for item_id, score in results
        ],
        model_version   = predictor.model_version,
        num_input_items = len(body.user_history),
    )
