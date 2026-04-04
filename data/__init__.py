"""data package — MovieLens 1M pipeline for BERT4Rec."""

from data.dataset import BERT4RecDataset, Split, build_dataloaders
from data.download import download_movielens
from data.preprocess import preprocess

__all__ = [
    "download_movielens",
    "preprocess",
    "BERT4RecDataset",
    "Split",
    "build_dataloaders",
]
