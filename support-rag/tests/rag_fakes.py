"""Offline stand-ins for embedding models (deterministic hashing encoders)."""

import hashlib
import re
from collections.abc import Sequence

from qdrant_client import models

DIM = 4096


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bucket(token: str) -> int:
    return int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16) % DIM


class HashingDense:
    """Deterministic bag-of-words encoder: fast, offline stand-in for a real embedding model."""

    dim = DIM

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for t in _tokens(text):
            v[_bucket(t)] += 1.0
        return v

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class HashingSparse:
    def _vec(self, text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for t in _tokens(text):
            counts[_bucket(t)] = counts.get(_bucket(t), 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_passages(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._vec(text)
