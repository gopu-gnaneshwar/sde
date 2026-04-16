from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.ai import ContextBlock, TextAIClient, stream_text_chunks
from app.cache import CacheBackend
from app.config import Settings
from app.extractors import ExtractionResult, MediaTextExtractor, PDFTextExtractor
from app.retrieval import Retriever, chunk_text, chunk_transcript_segments
from app.storage import FileStorage
from app.transcription import Transcriber
from app.utils import format_seconds, truncate_text
from app.vector_store import VectorIndex


class ProcessingError(RuntimeError):
    pass


@dataclass(slots=True)
class PreparedChat:
    context: list[ContextBlock]
    sources: list[dict]
    timestamp_matches: list[dict]


@dataclass(slots=True)
class ChatResult:
    answer: str
    sources: list[dict]
    timestamp_matches: list[dict]

    def as_payload(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "timestamp_matches": self.timestamp_matches,
        }


class DocumentService:
    def __init__(
        self,
        session: Session,
        settings: Settings,
        storage: FileStorage,
        ai_client: TextAIClient,
        transcriber: Transcriber,
        cache: CacheBackend,
        vector_index: VectorIndex,
    ):
        self.session = session
        self.settings = settings
        self.storage = storage
        self.ai_client = ai_client
        self.cache = cache
        self.vector_index = vector_index
        self.pdf_extractor = PDFTextExtractor()
        self.media_extractor = MediaTextExtractor(transcriber)

    def list_assets(self, user_id: str) -> list[models.Asset]:
        statement = (
            select(models.Asset)
            .join(models.UserAsset, models.UserAsset.asset_id == models.Asset.id)
            .where(models.UserAsset.user_id == user_id)
            .order_by(models.Asset.created_at.desc())
        )
        return self.session.scalars(statement).all()

    def get_asset(self, user_id: str, asset_id: str) -> models.Asset | None:
        statement = (
            select(models.Asset)
            .join(models.UserAsset, models.UserAsset.asset_id == models.Asset.id)
            .where(models.UserAsset.user_id == user_id, models.Asset.id == asset_id)
        )
        return self.session.scalar(statement)

    def create_asset(self, user_id: str, upload_file: UploadFile) -> models.Asset:
        from app.utils import detect_media_type

        filename = upload_file.filename or "upload.bin"
        media_type = detect_media_type(filename, upload_file.content_type)
        asset_id = str(uuid4())
        stored = self.storage.save_upload(upload_file, prefix=asset_id)

        asset = models.Asset(
            id=asset_id,
            original_filename=filename,
            stored_filename=stored.stored_filename,
            media_type=media_type,
            mime_type=upload_file.content_type or "application/octet-stream",
            storage_path=str(stored.path),
            processing_status="processing",
        )
        self.session.add(asset)
        self.session.add(models.UserAsset(user_id=user_id, asset_id=asset_id))
        self.session.flush()

        try:
            extraction = self._extract_content(asset)
            asset.extracted_text = extraction.text
            asset.duration_seconds = extraction.duration_seconds
            asset.summary = self.ai_client.summarize(
                title=asset.original_filename,
                media_type=asset.media_type,
                text=extraction.text,
            )
            asset.processing_status = "ready"
            self._replace_segments(asset, extraction)
            self.session.flush()
            self.vector_index.sync_asset(asset.id, asset.segments)
            self.session.commit()
        except Exception as exc:  # pragma: no cover - hit through tests with custom exceptions
            asset.processing_status = "failed"
            asset.error_message = str(exc)
            self.session.commit()
            raise ProcessingError(str(exc)) from exc

        self.session.refresh(asset)
        return asset

    def chat(self, user_id: str, question: str, asset_ids: Sequence[str] | None = None) -> ChatResult:
        scoped_asset_ids = self._resolve_asset_ids(user_id, asset_ids)
        cache_key = self._cache_key("chat", user_id, question, scoped_asset_ids)
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, dict):
            return ChatResult(
                answer=str(cached.get("answer", "")),
                sources=list(cached.get("sources", [])),
                timestamp_matches=list(cached.get("timestamp_matches", [])),
            )

        prepared = self._prepare_chat(question, scoped_asset_ids)
        answer = self.ai_client.answer(question=question, context=prepared.context)
        result = ChatResult(
            answer=answer,
            sources=prepared.sources,
            timestamp_matches=prepared.timestamp_matches,
        )
        self.cache.set_json(cache_key, result.as_payload(), ttl_seconds=self.settings.chat_cache_ttl_seconds)
        return result

    def stream_chat(
        self,
        user_id: str,
        question: str,
        asset_ids: Sequence[str] | None = None,
    ) -> Iterator[dict[str, object]]:
        scoped_asset_ids = self._resolve_asset_ids(user_id, asset_ids)
        cache_key = self._cache_key("chat", user_id, question, scoped_asset_ids)
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, dict):
            result = ChatResult(
                answer=str(cached.get("answer", "")),
                sources=list(cached.get("sources", [])),
                timestamp_matches=list(cached.get("timestamp_matches", [])),
            )
            for chunk in stream_text_chunks(result.answer):
                yield {"type": "answer_delta", "delta": chunk}
            yield {
                "type": "metadata",
                "sources": result.sources,
                "timestamp_matches": result.timestamp_matches,
            }
            yield {"type": "done"}
            return

        prepared = self._prepare_chat(question, scoped_asset_ids)
        answer_parts: list[str] = []
        for chunk in self.ai_client.stream_answer(question=question, context=prepared.context):
            answer_parts.append(chunk)
            yield {"type": "answer_delta", "delta": chunk}

        result = ChatResult(
            answer="".join(answer_parts).strip(),
            sources=prepared.sources,
            timestamp_matches=prepared.timestamp_matches,
        )
        self.cache.set_json(cache_key, result.as_payload(), ttl_seconds=self.settings.chat_cache_ttl_seconds)
        yield {
            "type": "metadata",
            "sources": result.sources,
            "timestamp_matches": result.timestamp_matches,
        }
        yield {"type": "done"}

    def find_topic_matches(self, user_id: str, asset_id: str, topic: str) -> list[dict]:
        asset = self.get_asset(user_id, asset_id)
        if asset is None:
            return []

        cache_key = self._cache_key("topic", user_id, topic, [asset_id])
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, list):
            return [item for item in cached if isinstance(item, dict)]

        retriever = Retriever(self.session, self.ai_client, vector_index=self.vector_index)
        hits = retriever.search(topic, asset_ids=[asset_id], limit=self.settings.max_search_results)
        matches = [
            {
                "asset_id": asset.id,
                "filename": asset.original_filename,
                "media_type": asset.media_type,
                "label": truncate_text(hit.segment.content, 100),
                "excerpt": truncate_text(hit.segment.content, 220),
                "start_seconds": hit.segment.start_seconds,
                "end_seconds": hit.segment.end_seconds,
                "start_label": format_seconds(hit.segment.start_seconds),
                "end_label": format_seconds(hit.segment.end_seconds),
            }
            for hit in hits
            if hit.segment.start_seconds is not None
        ]
        self.cache.set_json(cache_key, matches, ttl_seconds=self.settings.topic_cache_ttl_seconds)
        return matches

    def _prepare_chat(self, question: str, asset_ids: Sequence[str]) -> PreparedChat:
        if not asset_ids:
            return PreparedChat(context=[], sources=[], timestamp_matches=[])

        retriever = Retriever(self.session, self.ai_client, vector_index=self.vector_index)
        hits = retriever.search(
            question,
            asset_ids=list(asset_ids),
            limit=self.settings.max_search_results,
        )

        context = [
            ContextBlock(
                asset_id=hit.segment.asset.id,
                filename=hit.segment.asset.original_filename,
                media_type=hit.segment.asset.media_type,
                content=hit.segment.content,
                start_seconds=hit.segment.start_seconds,
                end_seconds=hit.segment.end_seconds,
            )
            for hit in hits
        ]
        sources = [
            {
                "asset_id": hit.segment.asset.id,
                "filename": hit.segment.asset.original_filename,
                "media_type": hit.segment.asset.media_type,
                "excerpt": truncate_text(hit.segment.content, 220),
                "score": round(hit.score, 4),
                "start_seconds": hit.segment.start_seconds,
                "end_seconds": hit.segment.end_seconds,
                "timestamp_label": format_seconds(hit.segment.start_seconds),
            }
            for hit in hits
        ]
        timestamp_matches = [
            {
                "asset_id": hit.segment.asset.id,
                "filename": hit.segment.asset.original_filename,
                "media_type": hit.segment.asset.media_type,
                "label": truncate_text(hit.segment.content, 100),
                "excerpt": truncate_text(hit.segment.content, 220),
                "start_seconds": hit.segment.start_seconds,
                "end_seconds": hit.segment.end_seconds,
                "start_label": format_seconds(hit.segment.start_seconds),
                "end_label": format_seconds(hit.segment.end_seconds),
            }
            for hit in hits
            if hit.segment.start_seconds is not None
        ]
        return PreparedChat(context=context, sources=sources, timestamp_matches=timestamp_matches)

    def _resolve_asset_ids(self, user_id: str, asset_ids: Sequence[str] | None) -> list[str]:
        statement = select(models.UserAsset.asset_id).where(models.UserAsset.user_id == user_id)
        if asset_ids is not None:
            statement = statement.where(models.UserAsset.asset_id.in_(list(asset_ids)))
        return self.session.scalars(statement).all()

    def _extract_content(self, asset: models.Asset) -> ExtractionResult:
        path = self.asset_path(asset)
        if asset.media_type == "pdf":
            return self.pdf_extractor.extract(path)
        return self.media_extractor.extract(path, asset.media_type)

    def _replace_segments(self, asset: models.Asset, extraction: ExtractionResult) -> None:
        chunks = (
            chunk_text(
                extraction.text,
                max_chars=self.settings.max_chunk_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
            )
            if asset.media_type == "pdf"
            else chunk_transcript_segments(
                extraction.segments,
                max_chars=min(self.settings.max_chunk_chars, 90),
            )
        )

        embeddings = self.ai_client.embed_texts([item.content for item in chunks]) if chunks else []
        asset.segments.clear()
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            asset.segments.append(
                models.Segment(
                    position=chunk.position,
                    content=chunk.content,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    embedding=embedding,
                )
            )

    @staticmethod
    def asset_path(asset: models.Asset) -> Path:
        return Path(asset.storage_path)

    @staticmethod
    def _cache_key(scope: str, user_id: str, query: str, asset_ids: Sequence[str]) -> str:
        payload = json.dumps(
            {
                "scope": scope,
                "user_id": user_id,
                "query": query,
                "asset_ids": list(asset_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"mediamind:{scope}:{digest}"
