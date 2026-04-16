from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.transcription import TranscriptSegment, Transcriber


@dataclass(slots=True)
class ExtractionResult:
    text: str
    segments: list[TranscriptSegment]
    duration_seconds: float | None = None


class PDFTextExtractor:
    def extract(self, file_path: Path) -> ExtractionResult:
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
        segments = [TranscriptSegment(text=text)] if text else []
        return ExtractionResult(text=text, segments=segments)


class MediaTextExtractor:
    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    def extract(self, file_path: Path, media_type: str) -> ExtractionResult:
        source_path = self._prepare_source(file_path, media_type)
        try:
            segments = self.transcriber.transcribe(source_path)
        finally:
            if source_path != file_path and source_path.exists():
                source_path.unlink()

        text = " ".join(item.text.strip() for item in segments if item.text.strip()).strip()
        duration = max((item.end_seconds or 0.0) for item in segments) if segments else None
        return ExtractionResult(text=text, segments=segments, duration_seconds=duration or None)

    def _prepare_source(self, file_path: Path, media_type: str) -> Path:
        if media_type != "video":
            return file_path
        return extract_audio_track(file_path)


def extract_audio_track(video_path: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    output_path = Path(temp_file.name)
    temp_file.close()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "mp3",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg must be installed to process video uploads.") from exc
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to extract audio: {exc.stderr}") from exc

    return output_path
