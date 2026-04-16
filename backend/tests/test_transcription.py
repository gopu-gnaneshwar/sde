from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.transcription import OpenAITranscriber, UnavailableTranscriber, build_transcriber


class StubAudioTranscriptions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model_dump=lambda: {
                "segments": [
                    {"text": "Opening remarks", "start": 0.0, "end": 4.0},
                    {"text": "Hiring update", "start": 5.0, "end": 8.0},
                ]
            }
        )


def test_unavailable_transcriber_raises(tmp_path):
    file_path = tmp_path / "clip.mp3"
    file_path.write_bytes(b"audio")
    with pytest.raises(RuntimeError):
        UnavailableTranscriber().transcribe(file_path)


def test_openai_transcriber_maps_segments(tmp_path):
    file_path = tmp_path / "clip.mp3"
    file_path.write_bytes(b"audio")
    stub = SimpleNamespace(audio=SimpleNamespace(transcriptions=StubAudioTranscriptions()))
    transcriber = OpenAITranscriber(api_key="test-key", model="whisper-1", client=stub)

    segments = transcriber.transcribe(file_path)

    assert len(segments) == 2
    assert segments[1].text == "Hiring update"
    assert stub.audio.transcriptions.calls[0]["model"] == "whisper-1"


def test_build_transcriber_switches_by_configuration(monkeypatch):
    settings = Settings(ai_provider="auto", openai_api_key="token")
    sentinel = object()
    monkeypatch.setattr("app.transcription.OpenAITranscriber", lambda api_key, model: sentinel)

    assert build_transcriber(settings) is sentinel
    assert isinstance(build_transcriber(Settings(ai_provider="mock")), UnavailableTranscriber)
