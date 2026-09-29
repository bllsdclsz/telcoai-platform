"""Dense and sparse text encoders (fastembed / ONNX: CPU-friendly, no torch dependency)."""

import os
from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models

# onnxruntime >= 1.30 refuses weights outside the model directory. On Linux the Hugging Face cache
# symlinks files into a shared blobs/ dir, which breaks models with external weights
# (e5-large's model.onnx_data). Real files in the snapshot avoid that; must be set before
# huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


class DenseEncoder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SparseEncoder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> list[models.SparseVector]: ...
    def embed_query(self, text: str) -> models.SparseVector: ...


# Instruction prefixes some models were trained with; without them retrieval quality drops.
_PREFIXES = {"e5": ("query: ", "passage: ")}


class FastEmbedDense:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        family = next((k for k in _PREFIXES if k in model_name.lower()), None)
        self._query_prefix, self._passage_prefix = _PREFIXES.get(family or "", ("", ""))
        self.dim = len(self.embed_query("dimension probe"))

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        texts = [self._passage_prefix + t for t in texts]
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(self._query_prefix + text))).tolist()


class FastEmbedSparse:
    """BM25-style keyword vectors; Qdrant applies the IDF weighting at query time."""

    def __init__(self, model_name: str) -> None:
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name)

    def embed_passages(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.passage_embed(list(texts))
        ]

    def embed_query(self, text: str) -> models.SparseVector:
        e = next(iter(self._model.query_embed(text)))
        return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
