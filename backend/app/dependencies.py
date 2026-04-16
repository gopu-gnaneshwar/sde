from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from app.ai import TextAIClient, build_text_ai_client
from app.auth import AuthManager
from app.cache import CacheBackend, RateLimiter, build_cache_backend
from app.config import Settings, get_settings
from app.database import build_engine, build_session_factory, init_db
from app.storage import FileStorage
from app.transcription import Transcriber, build_transcriber
from app.vector_store import VectorIndex, build_vector_index


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    session_factory: sessionmaker[Session]
    storage: FileStorage
    ai_client: TextAIClient
    transcriber: Transcriber
    cache: CacheBackend
    rate_limiter: RateLimiter
    auth_manager: AuthManager
    vector_index: VectorIndex


def build_container(
    settings: Settings | None = None,
    *,
    ai_client: TextAIClient | None = None,
    transcriber: Transcriber | None = None,
) -> AppContainer:
    settings = settings or get_settings()
    settings.ensure_directories()
    engine = build_engine(settings.database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    cache = build_cache_backend(settings)

    return AppContainer(
        settings=settings,
        session_factory=session_factory,
        storage=FileStorage(settings.uploads_dir),
        ai_client=ai_client or build_text_ai_client(settings),
        transcriber=transcriber or build_transcriber(settings),
        cache=cache,
        rate_limiter=RateLimiter(
            cache,
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
        auth_manager=AuthManager(settings),
        vector_index=build_vector_index(settings, session_factory),
    )
