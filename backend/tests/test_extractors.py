from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from app.extractors import MediaTextExtractor, PDFTextExtractor, extract_audio_track
from app.transcription import TranscriptSegment


class DummyTranscriber:
    def __init__(self, segments):
        self.segments = segments
        self.paths = []

    def transcribe(self, file_path: Path):
        self.paths.append(file_path)
        return self.segments


def test_pdf_text_extractor_joins_pages(monkeypatch):
    reader = SimpleNamespace(
        pages=[
            SimpleNamespace(extract_text=lambda: "First page"),
            SimpleNamespace(extract_text=lambda: ""),
            SimpleNamespace(extract_text=lambda: "Second page"),
        ]
    )
    monkeypatch.setattr("app.extractors.PdfReader", lambda path: reader)

    result = PDFTextExtractor().extract(Path("dummy.pdf"))

    assert result.text == "First page\n\nSecond page"
    assert result.segments[0].text == result.text


def test_media_text_extractor_builds_text_and_duration(tmp_path):
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    transcriber = DummyTranscriber(
        [
            TranscriptSegment(text="Intro", start_seconds=0.0, end_seconds=4.0),
            TranscriptSegment(text="Topic match", start_seconds=4.0, end_seconds=9.0),
        ]
    )

    result = MediaTextExtractor(transcriber).extract(source, "audio")

    assert result.text == "Intro Topic match"
    assert result.duration_seconds == 9.0
    assert transcriber.paths == [source]


def test_media_text_extractor_cleans_up_temp_audio(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    temp_audio = tmp_path / "temp.mp3"
    temp_audio.write_bytes(b"audio")
    transcriber = DummyTranscriber([TranscriptSegment(text="Scene", start_seconds=0.0, end_seconds=3.0)])
    extractor = MediaTextExtractor(transcriber)

    monkeypatch.setattr("app.extractors.extract_audio_track", lambda path: temp_audio)

    extractor.extract(source, "video")

    assert not temp_audio.exists()


def test_extract_audio_track_wraps_missing_ffmpeg(monkeypatch, tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "app.extractors.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ffmpeg")),
    )

    try:
        extract_audio_track(video)
    except RuntimeError as exc:
        assert "ffmpeg must be installed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError")


def test_extract_audio_track_wraps_subprocess_error(monkeypatch, tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")

    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg", stderr="bad input")

    monkeypatch.setattr("app.extractors.subprocess.run", raise_error)

    try:
        extract_audio_track(video)
    except RuntimeError as exc:
        assert "bad input" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError")
