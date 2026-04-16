from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from app import models
from app.cache import InMemoryCache, RateLimitExceeded, RateLimiter, RedisCache, build_cache_backend
from app.config import Settings
from app.database import build_engine, build_session_factory, init_db
from app.vector_store import FaissVectorIndex, InMemoryVectorIndex, build_vector_index


def test_in_memory_cache_expires_and_increments():
    cache = InMemoryCache()
    cache.set_json("chat", {"answer": "hello"}, ttl_seconds=1)
    assert cache.get_json("chat") == {"answer": "hello"}

    assert cache.increment("limit", ttl_seconds=5) == 1
    assert cache.increment("limit", ttl_seconds=5) == 2

    time.sleep(1.05)
    assert cache.get_json("chat") is None


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(InMemoryCache(), limit=2, window_seconds=60)
    limiter.check(scope="chat", identifier="tester")
    limiter.check(scope="chat", identifier="tester")

    with pytest.raises(RateLimitExceeded):
        limiter.check(scope="chat", identifier="tester")


def test_redis_cache_round_trip_and_fallback(monkeypatch):
    class FakeRedisClient:
        def __init__(self):
            self.values: dict[str, str] = {}
            self.expirations: dict[str, int] = {}

        def ping(self):
            return True

        def get(self, key: str):
            return self.values.get(key)

        def set(self, *, name: str, value: str, ex: int):
            self.values[name] = value
            self.expirations[name] = ex

        def incr(self, key: str):
            current = int(self.values.get(key, "0")) + 1
            self.values[key] = str(current)
            return current

        def expire(self, key: str, ttl: int):
            self.expirations[key] = ttl

    fake_client = FakeRedisClient()
    fake_module = SimpleNamespace(Redis=SimpleNamespace(from_url=lambda url, decode_responses: fake_client))
    monkeypatch.setitem(sys.modules, "redis", fake_module)

    cache = RedisCache("redis://example")
    cache.set_json("chat", {"answer": "hello"}, ttl_seconds=30)
    assert cache.get_json("chat") == {"answer": "hello"}
    assert cache.increment("limit", ttl_seconds=45) == 1
    assert fake_client.expirations["limit"] == 45

    monkeypatch.setattr("app.cache.RedisCache", lambda redis_url: (_ for _ in ()).throw(RuntimeError("down")))
    fallback = build_cache_backend(Settings(redis_url="redis://example"))
    assert isinstance(fallback, InMemoryCache)


def test_in_memory_vector_index_searches_best_match():
    index = InMemoryVectorIndex()
    asset = models.Asset(
        id="asset-1",
        original_filename="demo.pdf",
        stored_filename="demo.pdf",
        media_type="pdf",
        mime_type="application/pdf",
        storage_path="/tmp/demo.pdf",
        extracted_text="content",
        summary="summary",
        processing_status="ready",
    )
    first_segment = models.Segment(id=1, asset_id=asset.id, position=0, content="Revenue", embedding=[1.0, 0.0])
    second_segment = models.Segment(id=2, asset_id=asset.id, position=1, content="Hiring", embedding=[0.0, 1.0])

    index.sync_asset(asset.id, [first_segment, second_segment])
    matches = index.search([1.0, 0.0], limit=2)

    assert matches[0].segment_id == 1
    assert matches[0].score > matches[1].score


def test_in_memory_vector_index_skips_invalid_segments_and_filters_assets():
    index = InMemoryVectorIndex()
    index.sync_asset(
        "asset-1",
        [
            models.Segment(id=None, asset_id="asset-1", position=0, content="skip-id", embedding=[1.0, 0.0]),
            models.Segment(id=10, asset_id="asset-1", position=1, content="keep", embedding=[1.0, 0.0]),
            models.Segment(id=11, asset_id="asset-1", position=2, content="skip-embedding", embedding=None),
        ],
    )

    matches = index.search([1.0, 0.0], asset_ids=["asset-1"], limit=5)
    assert [match.segment_id for match in matches] == [10]


def test_faiss_vector_index_rebuilds_and_searches():
    index = FaissVectorIndex()
    index.sync_asset("asset-1", [])
    assert index.search([1.0, 0.0], limit=1) == []

    first = models.Segment(id=1, asset_id="asset-1", position=0, content="Revenue", embedding=[1.0, 0.0])
    second = models.Segment(id=2, asset_id="asset-2", position=1, content="Hiring", embedding=[0.0, 1.0])
    index.sync_asset("asset-1", [first])
    index.sync_asset("asset-2", [second])

    global_matches = index.search([1.0, 0.0], limit=2)
    filtered_matches = index.search([1.0, 0.0], asset_ids=["asset-2"], limit=2)

    assert global_matches[0].segment_id == 1
    assert filtered_matches[0].segment_id == 2


def test_build_vector_index_rehydrates_existing_segments(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'vector.db'}"
    engine = build_engine(database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session: Session = session_factory()

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
    session.add(models.Segment(asset_id=asset.id, position=0, content="Revenue improved", embedding=[0.9, 0.1]))
    session.commit()
    session.close()

    index = build_vector_index(
        Settings(
            database_url=database_url,
            storage_dir=tmp_path / "data",
            vector_backend="memory",
        ),
        session_factory,
    )

    matches = index.search([1.0, 0.0], limit=1)
    assert matches[0].score > 0.5


def test_build_vector_index_falls_back_when_faiss_unavailable(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'vector-fallback.db'}"
    engine = build_engine(database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)

    monkeypatch.setattr("app.vector_store.FaissVectorIndex", lambda: (_ for _ in ()).throw(RuntimeError("no faiss")))
    index = build_vector_index(
        Settings(
            database_url=database_url,
            storage_dir=tmp_path / "data",
            vector_backend="auto",
        ),
        session_factory,
    )

    assert isinstance(index, InMemoryVectorIndex)
