from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.utils import format_seconds


class AssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    media_type: str
    mime_type: str
    summary: str
    processing_status: str
    error_message: str | None
    duration_seconds: float | None
    created_at: datetime
    file_url: str
    duration_label: str | None = None
    text_preview: str = ""

    @classmethod
    def from_model(cls, asset, *, file_url: str, text_preview: str):
        return cls(
            id=asset.id,
            original_filename=asset.original_filename,
            media_type=asset.media_type,
            mime_type=asset.mime_type,
            summary=asset.summary,
            processing_status=asset.processing_status,
            error_message=asset.error_message,
            duration_seconds=asset.duration_seconds,
            created_at=asset.created_at,
            file_url=file_url,
            duration_label=format_seconds(asset.duration_seconds),
            text_preview=text_preview,
        )


class UploadResponse(BaseModel):
    asset: AssetResponse


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime

    @classmethod
    def from_model(cls, user):
        return cls(id=user.id, username=user.username, created_at=user.created_at)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    api_key: str | None = None


class ApiKeyResponse(BaseModel):
    api_key: str


class TopicQueryRequest(BaseModel):
    topic: str = Field(min_length=2)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2)
    asset_ids: list[str] | None = None


class SourceSnippet(BaseModel):
    asset_id: str
    filename: str
    media_type: str
    excerpt: str
    score: float
    start_seconds: float | None = None
    end_seconds: float | None = None
    timestamp_label: str | None = None


class TopicMatch(BaseModel):
    asset_id: str
    filename: str
    media_type: str
    label: str
    excerpt: str
    start_seconds: float
    end_seconds: float | None = None
    start_label: str
    end_label: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
    timestamp_matches: list[TopicMatch]
