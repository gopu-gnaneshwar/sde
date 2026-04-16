from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ai import MockTextAIClient
from app.config import Settings
from app.dependencies import build_container
from app.main import create_app
from app.transcription import TranscriptSegment


class FakeTranscriber:
    def transcribe(self, file_path: Path) -> list[TranscriptSegment]:
        suffix = file_path.suffix.lower()
        if suffix == ".mp3":
            return [
                TranscriptSegment(text="The team opened with the product roadmap.", start_seconds=0.0, end_seconds=6.0),
                TranscriptSegment(text="Hiring plans were expanded in the Bengaluru office.", start_seconds=18.0, end_seconds=28.0),
                TranscriptSegment(text="Quarterly revenue rose by eighteen percent year over year.", start_seconds=32.0, end_seconds=42.0),
            ]

        return [
            TranscriptSegment(text="The demo starts with onboarding.", start_seconds=0.0, end_seconds=5.0),
            TranscriptSegment(text="Security controls are explained halfway through the recording.", start_seconds=20.0, end_seconds=31.0),
        ]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "data",
        ai_provider="mock",
        max_chunk_chars=120,
        chunk_overlap_chars=15,
        rate_limit_requests=100,
        vector_backend="memory",
    )


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    container = build_container(settings, ai_client=MockTextAIClient(), transcriber=FakeTranscriber())
    app = create_app(container=container)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_payload(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={"username": "tester", "password": "supersecret1"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture()
def auth_headers(auth_payload: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_payload['access_token']}"}


@pytest.fixture()
def api_key_headers(auth_payload: dict[str, str]) -> dict[str, str]:
    return {"X-API-Key": auth_payload["api_key"]}
