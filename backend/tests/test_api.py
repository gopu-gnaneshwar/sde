from __future__ import annotations

import json

from app.extractors import ExtractionResult
from app.transcription import TranscriptSegment


def patch_pdf(monkeypatch, text: str) -> None:
    monkeypatch.setattr(
        "app.extractors.PDFTextExtractor.extract",
        lambda self, file_path: ExtractionResult(
            text=text,
            segments=[TranscriptSegment(text=text)],
            duration_seconds=None,
        ),
    )


def test_healthcheck(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_protected_routes_require_authentication(client):
    response = client.get("/api/assets")
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication credentials were not provided."


def test_register_login_me_and_rotate_api_key(client):
    register = client.post("/api/auth/register", json={"username": "tester", "password": "supersecret1"})
    assert register.status_code == 201
    register_payload = register.json()
    assert register_payload["user"]["username"] == "tester"
    assert register_payload["api_key"].startswith("mm_")

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {register_payload['access_token']}"})
    assert me.status_code == 200
    assert me.json()["username"] == "tester"

    login = client.post("/api/auth/login", json={"username": "tester", "password": "supersecret1"})
    assert login.status_code == 200
    login_payload = login.json()
    assert login_payload["api_key"] is None

    rotate = client.post(
        "/api/auth/api-key",
        headers={"Authorization": f"Bearer {register_payload['access_token']}"},
    )
    assert rotate.status_code == 200
    assert rotate.json()["api_key"].startswith("mm_")


def test_api_key_authentication_can_list_assets(client, api_key_headers):
    response = client.get("/api/assets", headers=api_key_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_upload_pdf_list_fetch_chat_and_stream(client, auth_headers, monkeypatch):
    patch_pdf(
        monkeypatch,
        "Quarterly revenue grew 18 percent. Customer churn declined and expansion revenue improved.",
    )

    upload = client.post(
        "/api/assets/upload",
        headers=auth_headers,
        files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert upload.status_code == 200
    asset = upload.json()["asset"]
    assert asset["media_type"] == "pdf"
    assert asset["summary"]
    assert asset["file_url"].endswith("/content")

    listing = client.get("/api/assets", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.json()[0]["id"] == asset["id"]

    detail = client.get(f"/api/assets/{asset['id']}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["original_filename"] == "report.pdf"

    content = client.get(f"/api/assets/{asset['id']}/content", headers=auth_headers)
    assert content.status_code == 200
    assert content.headers["content-type"].startswith("application/pdf")

    chat = client.post(
        "/api/chat",
        headers=auth_headers,
        json={"question": "What happened to quarterly revenue?", "asset_ids": [asset["id"]]},
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert "quarterly revenue" in payload["answer"].lower()
    assert payload["timestamp_matches"] == []
    assert payload["sources"][0]["filename"] == "report.pdf"

    stream = client.post(
        "/api/chat/stream",
        headers=auth_headers,
        json={"question": "What happened to quarterly revenue?", "asset_ids": [asset["id"]]},
    )
    assert stream.status_code == 200
    events = [json.loads(line) for line in stream.text.splitlines() if line.strip()]
    assert any(event["type"] == "answer_delta" for event in events)
    metadata_event = next(event for event in events if event["type"] == "metadata")
    assert metadata_event["sources"][0]["filename"] == "report.pdf"
    assert events[-1]["type"] == "done"


def test_upload_audio_topic_search_and_chat(client, auth_headers):
    upload = client.post(
        "/api/assets/upload",
        headers=auth_headers,
        files={"file": ("earnings.mp3", b"fake-audio", "audio/mpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()["asset"]
    assert asset["media_type"] == "audio"
    assert asset["duration_label"] == "00:42"

    topics = client.post(
        f"/api/assets/{asset['id']}/topics",
        headers=auth_headers,
        json={"topic": "hiring plans"},
    )
    assert topics.status_code == 200
    matches = topics.json()
    assert matches[0]["start_label"] == "00:18"
    assert "Hiring plans" in matches[0]["excerpt"]

    chat = client.post(
        "/api/chat",
        headers=auth_headers,
        json={"question": "Where are the hiring plans discussed?", "asset_ids": [asset["id"]]},
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["timestamp_matches"][0]["start_seconds"] == 18.0
    assert payload["sources"][0]["timestamp_label"] in {"00:18", "00:32", "00:00"}


def test_upload_video_uses_media_pipeline(client, auth_headers, monkeypatch):
    from pathlib import Path

    def fake_extract_audio(path: Path) -> Path:
        audio_path = path.with_suffix(".mp3")
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr("app.extractors.extract_audio_track", fake_extract_audio)
    upload = client.post(
        "/api/assets/upload",
        headers=auth_headers,
        files={"file": ("demo.webm", b"fake-video", "video/webm")},
    )
    assert upload.status_code == 200
    asset = upload.json()["asset"]
    assert asset["media_type"] == "video"
    assert asset["duration_label"] == "00:42"


def test_upload_rejects_unsupported_files(client, auth_headers):
    response = client.post(
        "/api/assets/upload",
        headers=auth_headers,
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_returns_processing_error(client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        "app.extractors.PDFTextExtractor.extract",
        lambda self, file_path: (_ for _ in ()).throw(RuntimeError("extract failed")),
    )
    response = client.post(
        "/api/assets/upload",
        headers=auth_headers,
        files={"file": ("broken.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "extract failed"


def test_user_scoping_hides_other_users_assets(client, monkeypatch):
    patch_pdf(monkeypatch, "Shared report content.")

    first = client.post("/api/auth/register", json={"username": "first", "password": "supersecret1"})
    second = client.post("/api/auth/register", json={"username": "second", "password": "supersecret1"})
    first_headers = {"Authorization": f"Bearer {first.json()['access_token']}"}
    second_headers = {"Authorization": f"Bearer {second.json()['access_token']}"}

    upload = client.post(
        "/api/assets/upload",
        headers=first_headers,
        files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    asset_id = upload.json()["asset"]["id"]

    listing = client.get("/api/assets", headers=second_headers)
    assert listing.status_code == 200
    assert listing.json() == []

    detail = client.get(f"/api/assets/{asset_id}", headers=second_headers)
    assert detail.status_code == 404


def test_not_found_routes(client, auth_headers):
    assert client.get("/api/assets/missing", headers=auth_headers).status_code == 404
    assert client.get("/api/assets/missing/content", headers=auth_headers).status_code == 404
    assert client.post("/api/assets/missing/topics", headers=auth_headers, json={"topic": "security"}).status_code == 404
