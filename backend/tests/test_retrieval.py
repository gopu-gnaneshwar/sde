from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.ai import MockTextAIClient
from app.database import build_engine, build_session_factory, init_db
from app.retrieval import Retriever, chunk_text, chunk_transcript_segments, cosine_similarity
from app.transcription import TranscriptSegment


def test_chunk_text_splits_long_paragraphs():
    chunks = chunk_text("A" * 140 + "\n\n" + "B" * 140, max_chars=120, overlap_chars=12)
    assert len(chunks) >= 3
    assert chunks[0].position == 0


def test_chunk_transcript_segments_groups_timestamps():
    chunks = chunk_transcript_segments(
        [
            TranscriptSegment(text="alpha", start_seconds=0.0, end_seconds=2.0),
            TranscriptSegment(text="beta" * 15, start_seconds=2.0, end_seconds=8.0),
            TranscriptSegment(text="gamma", start_seconds=8.0, end_seconds=10.0),
        ],
        max_chars=35,
    )
    assert len(chunks) == 3
    assert chunks[0].start_seconds == 0.0
    assert chunks[-1].end_seconds == 10.0


def test_cosine_similarity_guards_and_scores():
    assert cosine_similarity(None, [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0
    assert round(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 4) == 1.0


def test_retriever_orders_hits(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'retrieval.db'}"
    engine = build_engine(database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session: Session = session_factory()
    ai_client = MockTextAIClient()

    asset = models.Asset(
        id="asset-1",
        original_filename="doc.pdf",
        stored_filename="doc.pdf",
        media_type="pdf",
        mime_type="application/pdf",
        storage_path=str(tmp_path / "doc.pdf"),
        extracted_text="Revenue improved.",
        summary="summary",
        processing_status="ready",
    )
    session.add(asset)
    session.flush()
    session.add_all(
        [
            models.Segment(
                asset_id=asset.id,
                position=0,
                content="Revenue improved sharply.",
                embedding=ai_client.embed_texts(["Revenue improved sharply."])[0],
            ),
            models.Segment(
                asset_id=asset.id,
                position=1,
                content="Office renovation details.",
                embedding=ai_client.embed_texts(["Office renovation details."])[0],
            ),
        ]
    )
    session.commit()

    hits = Retriever(session, ai_client).search("What happened to revenue?", asset_ids=["asset-1"], limit=2)

    assert len(hits) == 2
    assert hits[0].segment.content == "Revenue improved sharply."
    session.close()


def test_retriever_honors_empty_asset_scope(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'empty-scope.db'}"
    engine = build_engine(database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session: Session = session_factory()
    ai_client = MockTextAIClient()

    asset = models.Asset(
        id="asset-1",
        original_filename="doc.pdf",
        stored_filename="doc.pdf",
        media_type="pdf",
        mime_type="application/pdf",
        storage_path=str(tmp_path / "doc.pdf"),
        extracted_text="Revenue improved.",
        summary="summary",
        processing_status="ready",
    )
    session.add(asset)
    session.add(
        models.Segment(
            asset_id=asset.id,
            position=0,
            content="Revenue improved sharply.",
            embedding=ai_client.embed_texts(["Revenue improved sharply."])[0],
        )
    )
    session.commit()

    hits = Retriever(session, ai_client).search("What happened to revenue?", asset_ids=[], limit=2)

    assert hits == []
    session.close()
