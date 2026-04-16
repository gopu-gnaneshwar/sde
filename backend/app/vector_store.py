from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import models
from app.config import Settings


@dataclass(slots=True)
class IndexedSegment:
    segment_id: int
    asset_id: str
    embedding: list[float]


@dataclass(slots=True)
class VectorMatch:
    segment_id: int
    score: float


class VectorIndex(Protocol):
    def sync_asset(self, asset_id: str, segments: Sequence[models.Segment]) -> None:
        ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        asset_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[VectorMatch]:
        ...


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left)) or 1.0
    right_norm = math.sqrt(sum(item * item for item in right)) or 1.0
    return numerator / (left_norm * right_norm)


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class InMemoryVectorIndex:
    def __init__(self):
        self._entries: dict[int, IndexedSegment] = {}

    def sync_asset(self, asset_id: str, segments: Sequence[models.Segment]) -> None:
        stale_ids = [segment_id for segment_id, entry in self._entries.items() if entry.asset_id == asset_id]
        for segment_id in stale_ids:
            self._entries.pop(segment_id, None)

        for segment in segments:
            if segment.id is None or not segment.embedding:
                continue
            self._entries[segment.id] = IndexedSegment(
                segment_id=segment.id,
                asset_id=asset_id,
                embedding=_normalize(segment.embedding),
            )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        asset_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[VectorMatch]:
        normalized_query = _normalize(query_vector)
        allowed_asset_ids = set(asset_ids) if asset_ids else None
        ranked = [
            VectorMatch(segment_id=entry.segment_id, score=_cosine_similarity(normalized_query, entry.embedding))
            for entry in self._entries.values()
            if allowed_asset_ids is None or entry.asset_id in allowed_asset_ids
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]


class FaissVectorIndex(InMemoryVectorIndex):
    def __init__(self):
        super().__init__()
        import faiss

        self.faiss = faiss
        self._index = None
        self._segment_ids: list[int] = []

    def sync_asset(self, asset_id: str, segments: Sequence[models.Segment]) -> None:
        super().sync_asset(asset_id, segments)
        self._rebuild_index()

    def search(
        self,
        query_vector: Sequence[float],
        *,
        asset_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[VectorMatch]:
        if asset_ids or self._index is None:
            return super().search(query_vector, asset_ids=asset_ids, limit=limit)

        import numpy as np

        query = np.array([_normalize(query_vector)], dtype="float32")
        score_matrix, index_matrix = self._index.search(query, limit)
        matches: list[VectorMatch] = []
        for score, raw_index in zip(score_matrix[0], index_matrix[0], strict=True):
            if raw_index < 0:
                continue
            matches.append(VectorMatch(segment_id=self._segment_ids[raw_index], score=float(score)))
        return matches

    def _rebuild_index(self) -> None:
        if not self._entries:
            self._index = None
            self._segment_ids = []
            return

        import numpy as np

        ordered_entries = list(self._entries.values())
        matrix = np.array([entry.embedding for entry in ordered_entries], dtype="float32")
        self._segment_ids = [entry.segment_id for entry in ordered_entries]
        dimension = matrix.shape[1]
        self._index = self.faiss.IndexFlatIP(dimension)
        self._index.add(matrix)


def build_vector_index(
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> VectorIndex:
    backend = settings.vector_backend.lower()
    index: VectorIndex

    if backend in {"faiss", "auto"}:
        try:
            index = FaissVectorIndex()
        except Exception:
            index = InMemoryVectorIndex()
    else:
        index = InMemoryVectorIndex()

    session = session_factory()
    try:
        segments = session.scalars(select(models.Segment)).all()
        asset_map: dict[str, list[models.Segment]] = {}
        for segment in segments:
            asset_map.setdefault(segment.asset_id, []).append(segment)
    finally:
        session.close()

    for asset_id, asset_segments in asset_map.items():
        index.sync_asset(asset_id, asset_segments)

    return index
