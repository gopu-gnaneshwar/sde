from __future__ import annotations

import mimetypes
import re
from pathlib import Path


PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return stem or "file"


def detect_media_type(filename: str, content_type: str | None = None) -> str:
    suffix = Path(filename).suffix.lower()
    mime = (content_type or mimetypes.guess_type(filename)[0] or "").lower()

    if suffix in PDF_EXTENSIONS or mime == "application/pdf":
        return "pdf"
    if suffix in AUDIO_EXTENSIONS or mime.startswith("audio/"):
        return "audio"
    if suffix in VIDEO_EXTENSIONS or mime.startswith("video/"):
        return "video"
    raise ValueError(f"Unsupported file type: {filename}")


def truncate_text(text: str, limit: int = 240) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def format_seconds(value: float | None) -> str | None:
    if value is None:
        return None
    total = int(value)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
