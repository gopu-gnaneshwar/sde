from __future__ import annotations

from uuid import uuid4

from app import models
from app.ai import MockTextAIClient
from app.dependencies import build_container
from app.services import DocumentService
from conftest import FakeTranscriber


def build_service(settings):
    container = build_container(settings, ai_client=MockTextAIClient(), transcriber=FakeTranscriber())
    session = container.session_factory()
    user, _, _ = container.auth_manager.register(session, f"user-{uuid4().hex[:8]}", "supersecret1")
    service = DocumentService(
        session=session,
        settings=settings,
        storage=container.storage,
        ai_client=container.ai_client,
        transcriber=container.transcriber,
        cache=container.cache,
        vector_index=container.vector_index,
    )
    return session, user, service


def add_asset(service: DocumentService, user_id: str, *, media_type: str, content: str):
    asset = models.Asset(
        id=str(uuid4()),
        original_filename=f"demo.{ 'pdf' if media_type == 'pdf' else 'mp3'}",
        stored_filename=f"demo.{ 'pdf' if media_type == 'pdf' else 'mp3'}",
        media_type=media_type,
        mime_type="application/pdf" if media_type == "pdf" else "audio/mpeg",
        storage_path=f"/tmp/{uuid4().hex}",
        extracted_text=content,
        summary="summary",
        processing_status="ready",
        duration_seconds=18.0 if media_type == "audio" else None,
    )
    service.session.add(asset)
    service.session.add(models.UserAsset(user_id=user_id, asset_id=asset.id))
    service.session.flush()
    service.session.add(
        models.Segment(
            asset_id=asset.id,
            position=0,
            content=content,
            start_seconds=12.0 if media_type == "audio" else None,
            end_seconds=18.0 if media_type == "audio" else None,
            embedding=service.ai_client.embed_texts([content])[0],
        )
    )
    service.session.commit()
    service.session.refresh(asset)
    service.vector_index.sync_asset(asset.id, asset.segments)
    return asset


def test_document_service_chat_uses_cache_and_handles_empty_scope(settings, monkeypatch):
    session, user, service = build_service(settings)

    empty_result = service.chat(user.id, "What happened?")
    assert empty_result.answer == "I could not find relevant context in the uploaded files."

    asset = add_asset(service, user.id, media_type="pdf", content="Revenue improved sharply.")
    first = service.chat(user.id, "What happened to revenue?", [asset.id])
    assert "Revenue improved sharply" in first.answer

    monkeypatch.setattr(
        service.ai_client,
        "answer",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cache should have been used")),
    )
    second = service.chat(user.id, "What happened to revenue?", [asset.id])
    assert second.answer == first.answer

    session.close()


def test_document_service_streams_uncached_results_and_uses_topic_cache(settings):
    session, user, service = build_service(settings)
    asset = add_asset(service, user.id, media_type="audio", content="Revenue was discussed in the closing section.")

    events = list(service.stream_chat(user.id, "Where is revenue discussed?", [asset.id]))
    assert any(event["type"] == "answer_delta" for event in events)
    assert any(event["type"] == "metadata" for event in events)
    assert events[-1]["type"] == "done"

    assert service.find_topic_matches(user.id, "missing", "revenue") == []

    cache_key = service._cache_key("topic", user.id, "revenue", [asset.id])
    service.cache.set_json(cache_key, [{"asset_id": asset.id, "label": "cached"}, "ignore-me"], ttl_seconds=30)
    cached_matches = service.find_topic_matches(user.id, asset.id, "revenue")
    assert cached_matches == [{"asset_id": asset.id, "label": "cached"}]

    session.close()
