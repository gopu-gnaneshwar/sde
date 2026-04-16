from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


class Transcriber(Protocol):
    def transcribe(self, file_path: Path) -> list[TranscriptSegment]:
        ...


class UnavailableTranscriber:
    def transcribe(self, file_path: Path) -> list[TranscriptSegment]:
        raise RuntimeError(
            "Audio and video transcription requires an OpenAI API key. "
            "If OPENAI_API_KEY is not configured, PDF uploads still work, but media files cannot be transcribed or indexed for topic/timestamp search."
        )


class OpenAITranscriber:
    def __init__(self, api_key: str, model: str, client=None):
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client

    def transcribe(self, file_path: Path) -> list[TranscriptSegment]:
        with file_path.open("rb") as handle:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        payload = response.model_dump() if hasattr(response, "model_dump") else response
        raw_segments = payload.get("segments", [])
        return [
            TranscriptSegment(
                text=(item.get("text") or "").strip(),
                start_seconds=item.get("start"),
                end_seconds=item.get("end"),
            )
            for item in raw_segments
            if (item.get("text") or "").strip()
        ]


def build_transcriber(settings: Settings) -> Transcriber:
    provider = settings.ai_provider.lower()
    if provider == "mock":
        return UnavailableTranscriber()
    if settings.openai_api_key:
        return OpenAITranscriber(settings.openai_api_key, settings.openai_transcription_model)
    return UnavailableTranscriber()
