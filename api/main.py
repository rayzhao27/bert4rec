from __future__ import annotations

"""
api/main.py
────────────
FastAPI application factory for the BERT4Rec inference service.

Startup:  loads the model checkpoint into memory once.
Shutdown: releases the model and frees memory cleanly.

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for the auto-generated Swagger UI.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import Predictor
from api.routes import router

logging.basicConfig(
    level   = os.getenv("LOG_LEVEL", "INFO"),
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config (override via environment variables) ───────────────────────────────
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
DATA_DIR        = os.getenv("DATA_DIR",        "data")
DEVICE          = os.getenv("DEVICE",          None)   # None = auto-select


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Starting BERT4Rec inference service …")

    predictor = Predictor(
        checkpoint_path = CHECKPOINT_PATH,
        data_dir        = DATA_DIR,
        device          = DEVICE,
    )
    predictor.load()
    app.state.predictor = predictor

    logger.info("Service ready.")
    yield

    logger.info("Shutting down …")
    app.state.predictor.unload()


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title       = "BERT4Rec Recommendation API",
        description=(
            "Sequential movie recommendation powered by BERT4Rec "
            "(Sun et al., 2019), trained on MovieLens 1M.\n\n"
            "Benchmarked on MovieLens 1M: HR@10 = 0.29, NDCG@10 = 0.16 "
            "(paper reports HR@10 = 0.27, NDCG@10 = 0.14)."
        ),
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    # CORS — allow all origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # Register routes
    app.include_router(router)

    return app


app = create_app()
