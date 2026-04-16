from __future__ import annotations

from fastapi.testclient import TestClient

from app.ai import MockTextAIClient
from app.config import Settings
from app.dependencies import build_container
from app.main import create_app
from conftest import FakeTranscriber


def build_limited_client(tmp_path, *, limit: int) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'limited.db'}",
        storage_dir=tmp_path / "data",
        ai_provider="mock",
        rate_limit_requests=limit,
        vector_backend="memory",
    )
    container = build_container(settings, ai_client=MockTextAIClient(), transcriber=FakeTranscriber())
    return TestClient(create_app(container=container))


def test_auth_register_rate_limit(tmp_path):
    with build_limited_client(tmp_path, limit=1) as client:
        first = client.post("/api/auth/register", json={"username": "first", "password": "supersecret1"})
        second = client.post("/api/auth/register", json={"username": "second", "password": "supersecret1"})

    assert first.status_code == 201
    assert second.status_code == 429


def test_auth_login_rate_limit(tmp_path):
    with build_limited_client(tmp_path, limit=1) as client:
        client.post("/api/auth/register", json={"username": "tester", "password": "supersecret1"})
        first = client.post("/api/auth/login", json={"username": "tester", "password": "supersecret1"})
        second = client.post("/api/auth/login", json={"username": "tester", "password": "supersecret1"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_chat_rate_limit(tmp_path):
    with build_limited_client(tmp_path, limit=1) as client:
        register = client.post("/api/auth/register", json={"username": "tester", "password": "supersecret1"})
        headers = {"Authorization": f"Bearer {register.json()['access_token']}"}

        first = client.post("/api/chat", headers=headers, json={"question": "What happened?", "asset_ids": None})
        second = client.post("/api/chat", headers=headers, json={"question": "What happened?", "asset_ids": None})

    assert first.status_code == 200
    assert second.status_code == 429
