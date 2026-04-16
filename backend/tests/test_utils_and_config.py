from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.dependencies import build_container
from app.utils import detect_media_type, format_seconds, safe_stem, truncate_text


def test_settings_parse_origins_and_create_directories(tmp_path):
    settings = Settings(
        storage_dir=tmp_path / "data",
        cors_origins="http://localhost:5173, http://example.com",
        redis_url="",
    )
    settings.ensure_directories()

    assert settings.cors_origins == ["http://localhost:5173", "http://example.com"]
    assert settings.redis_url is None
    assert settings.uploads_dir.exists()


def test_settings_load_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_NAME", "Configured App")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "env-data"))
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

    settings = Settings()

    assert settings.app_name == "Configured App"
    assert settings.cors_origins == ["http://localhost:8000", "http://127.0.0.1:8000"]


def test_build_container_creates_storage_and_session(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'db.sqlite3'}", storage_dir=tmp_path / "files")
    container = build_container(settings=settings)

    assert container.storage.root == settings.uploads_dir
    assert container.cache is not None
    assert container.vector_index is not None
    session = container.session_factory()
    try:
        assert session is not None
    finally:
        session.close()


def test_utils_cover_type_detection_and_formatting():
    assert detect_media_type("file.pdf") == "pdf"
    assert detect_media_type("voice.mp3") == "audio"
    assert detect_media_type("movie.webm") == "video"
    assert safe_stem("Quarterly Revenue!.pdf") == "quarterly-revenue-pdf"
    assert truncate_text("one two three", 100) == "one two three"
    assert format_seconds(65) == "01:05"
    assert format_seconds(None) is None


def test_detect_media_type_rejects_unknown():
    try:
        detect_media_type("notes.txt", "text/plain")
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")
