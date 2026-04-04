"""
data/download.py
────────────────
Downloads and extracts the MovieLens 1M dataset from GroupLens.

Output layout (inside --data_dir):
    raw/
        movies.dat
        ratings.dat
        users.dat
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_FILES = {"movies.dat", "ratings.dat", "users.dat"}
CHUNK_SIZE = 8_192  # bytes per download chunk


# ── Core functions ─────────────────────────────────────────────────────────────

def download_zip(url: str, timeout: int = 60) -> bytes:
    """Stream-download a ZIP from *url* and return its raw bytes."""
    logger.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    buffer = io.BytesIO()

    with tqdm(total=total, unit="B", unit_scale=True, desc="ml-1m.zip") as bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            buffer.write(chunk)
            bar.update(len(chunk))

    buffer.seek(0)
    return buffer.read()


def extract_zip(raw_bytes: bytes, dest_dir: Path) -> None:
    """Extract *raw_bytes* (a ZIP archive) into *dest_dir/raw/*."""
    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        # The archive nests files inside ml-1m/; flatten into raw/
        for member in zf.infolist():
            filename = Path(member.filename).name
            if not filename or filename not in EXPECTED_FILES:
                continue
            target = raw_dir / filename
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            logger.info("  extracted → %s", target)


def verify(data_dir: Path) -> bool:
    """Return True if all expected .dat files are present."""
    raw_dir = data_dir / "raw"
    present = {p.name for p in raw_dir.glob("*.dat")}
    missing = EXPECTED_FILES - present
    if missing:
        logger.warning("Missing files: %s", missing)
        return False
    logger.info("All dataset files verified in %s", raw_dir)
    return True


def download_movielens(data_dir: str | Path = "data", force: bool = False) -> Path:
    """
    High-level entry point.  Downloads ML-1M into *data_dir/raw/* unless the
    files already exist (skip unless *force=True*).

    Returns the path to the raw/ directory.
    """
    data_dir = Path(data_dir)

    if not force and verify(data_dir):
        logger.info("Dataset already present – skipping download. Use --force to re-download.")
        return data_dir / "raw"

    raw_bytes = download_zip(ML1M_URL)
    extract_zip(raw_bytes, data_dir)
    verify(data_dir)
    return data_dir / "raw"


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MovieLens 1M dataset")
    p.add_argument("--data_dir", default="data", help="Root data directory (default: data/)")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s | %(message)s")
    download_movielens(data_dir=args.data_dir, force=args.force)
