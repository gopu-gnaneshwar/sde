from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app import models
from app.ai import TextAIClient
from app.transcription import TranscriptSegment
from app.vector_store import VectorIndex


@dataclass(slots=True)
class Chunk:
    position: int
    content: str
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(slots=True)
class RetrievedSegment:
    segment: models.Segment
    score: float


def chunk_text(text: str, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    normalized = text.strip()
    if not normalized:
        return []

    paragraphs = [item.strip() for item in re.split(r"\n{2,}", normalized) if item.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[Chunk] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(Chunk(position=len(chunks), content=current))

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            chunks.append(Chunk(position=len(chunks), content=paragraph[start:end].strip()))
            if end == len(paragraph):
                break
            start = max(end - overlap_chars, start + 1)
        current = ""

    if current:
        chunks.append(Chunk(position=len(chunks), content=current))

    return chunks


def chunk_transcript_segments(segments: Sequence[TranscriptSegment], *, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[TranscriptSegment] = []
    current_size = 0

    for segment in segments:
        segment_length = len(segment.text)
        if buffer and current_size + segment_length > max_chars:
            chunks.append(_transcript_chunk(buffer, len(chunks)))
            buffer = []
            current_size = 0

        buffer.append(segment)
        current_size += segment_length

    if buffer:
        chunks.append(_transcript_chunk(buffer, len(chunks)))

    return chunks


def _transcript_chunk(segments: Sequence[TranscriptSegment], position: int) -> Chunk:
    return Chunk(
        position=position,
        content=" ".join(item.text.strip() for item in segments if item.text.strip()),
        start_seconds=segments[0].start_seconds,
        end_seconds=segments[-1].end_seconds,
    )


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def lexical_overlap_score(query: str, content: str) -> float:
    query_terms = {item for item in re.findall(r"[a-z0-9]+", query.lower()) if len(item) > 2}
    content_terms = {item for item in re.findall(r"[a-z0-9]+", content.lower()) if len(item) > 2}
    if not query_terms or not content_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


class Retriever:
    def __init__(
        self,
        session: Session,
        ai_client: TextAIClient,
        *,
        vector_index: VectorIndex | None = None,
    ):
        self.session = session
        self.ai_client = ai_client
        self.vector_index = vector_index

    def search(
        self,
        query: str,
        *,
        asset_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[RetrievedSegment]:
        query_vector = self.ai_client.embed_texts([query])[0]

        statement = select(models.Segment).options(joinedload(models.Segment.asset))
        if asset_ids is not None:
            statement = statement.where(models.Segment.asset_id.in_(list(asset_ids)))

        candidate_segment_ids: list[int] = []
        if self.vector_index is not None:
            semantic_hits = self.vector_index.search(
                query_vector,
                asset_ids=asset_ids,
                limit=max(limit * 5, limit),
            )
            candidate_segment_ids = [hit.segment_id for hit in semantic_hits]

        if candidate_segment_ids:
            statement = statement.where(models.Segment.id.in_(candidate_segment_ids))

        segments = self.session.scalars(statement).all()
        ranked = []
        for segment in segments:
            semantic_score = cosine_similarity(query_vector, segment.embedding)
            lexical_score = lexical_overlap_score(query, segment.content)
            ranked.append(RetrievedSegment(segment=segment, score=semantic_score + (0.35 * lexical_score)))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]
