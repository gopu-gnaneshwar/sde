from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.auth import AuthError
from app.cache import RateLimitExceeded
from app.dependencies import AppContainer, build_container
from app.schemas import (
    ApiKeyResponse,
    AssetResponse,
    AuthRequest,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    TopicMatch,
    TopicQueryRequest,
    UploadResponse,
    UserResponse,
)
from app.services import DocumentService, ProcessingError
from app.utils import truncate_text


def create_app(
    *,
    container: AppContainer | None = None,
) -> FastAPI:
    container = container or build_container()
    app = FastAPI(title=container.settings.app_name)
    app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=container.settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_session(request: Request):
        session = request.app.state.container.session_factory()
        try:
            yield session
        finally:
            session.close()

    def get_document_service(
        request: Request,
        session: Session = Depends(get_session),
    ) -> DocumentService:
        active_container = request.app.state.container
        return DocumentService(
            session=session,
            settings=active_container.settings,
            storage=active_container.storage,
            ai_client=active_container.ai_client,
            transcriber=active_container.transcriber,
            cache=active_container.cache,
            vector_index=active_container.vector_index,
        )

    def get_current_user(
        request: Request,
        session: Session = Depends(get_session),
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        active_container = request.app.state.container
        try:
            return active_container.auth_manager.authenticate(
                session,
                authorization=authorization,
                api_key=x_api_key,
            )
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    def get_rate_limited_user(
        request: Request,
        current_user=Depends(get_current_user),
    ):
        active_container = request.app.state.container
        try:
            active_container.rate_limiter.check(scope=request.url.path, identifier=current_user.id)
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
        return current_user

    def throttle_identifier(request: Request) -> str:
        return request.client.host if request.client else "anonymous"

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.options("/{full_path:path}")
    async def preflight_handler():
        return {}

    @app.post(
        f"{container.settings.api_prefix}/auth/register",
        response_model=AuthResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def register(payload: AuthRequest, request: Request, session: Session = Depends(get_session)):
        active_container = request.app.state.container
        try:
            active_container.rate_limiter.check(scope="auth-register", identifier=throttle_identifier(request))
            user, access_token, api_key = active_container.auth_manager.register(
                session,
                username=payload.username,
                password=payload.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

        return AuthResponse(
            access_token=access_token,
            user=UserResponse.from_model(user),
            api_key=api_key,
        )

    @app.post(f"{container.settings.api_prefix}/auth/login", response_model=AuthResponse)
    def login(payload: AuthRequest, request: Request, session: Session = Depends(get_session)):
        active_container = request.app.state.container
        try:
            active_container.rate_limiter.check(scope="auth-login", identifier=throttle_identifier(request))
            user, access_token = active_container.auth_manager.login(
                session,
                username=payload.username,
                password=payload.password,
            )
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

        return AuthResponse(access_token=access_token, user=UserResponse.from_model(user))

    @app.get(f"{container.settings.api_prefix}/auth/me", response_model=UserResponse)
    def me(current_user=Depends(get_current_user)):
        return UserResponse.from_model(current_user)

    @app.post(f"{container.settings.api_prefix}/auth/api-key", response_model=ApiKeyResponse)
    def rotate_api_key(
        request: Request,
        current_user=Depends(get_rate_limited_user),
        session: Session = Depends(get_session),
    ):
        active_container = request.app.state.container
        api_key = active_container.auth_manager.rotate_api_key(session, current_user)
        return ApiKeyResponse(api_key=api_key)

    @app.get(f"{container.settings.api_prefix}/assets", response_model=list[AssetResponse])
    def list_assets(
        request: Request,
        current_user=Depends(get_current_user),
        service: DocumentService = Depends(get_document_service),
    ):
        return [asset_to_schema(asset, request) for asset in service.list_assets(current_user.id)]

    @app.post(f"{container.settings.api_prefix}/assets/upload", response_model=UploadResponse)
    def upload_asset(
        request: Request,
        file: UploadFile = File(...),
        current_user=Depends(get_rate_limited_user),
        service: DocumentService = Depends(get_document_service),
    ):
        try:
            asset = service.create_asset(current_user.id, file)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ProcessingError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        return UploadResponse(asset=asset_to_schema(asset, request))

    @app.get(
        f"{container.settings.api_prefix}/assets/{{asset_id}}",
        response_model=AssetResponse,
    )
    def get_asset(
        asset_id: str,
        request: Request,
        current_user=Depends(get_current_user),
        service: DocumentService = Depends(get_document_service),
    ):
        asset = service.get_asset(current_user.id, asset_id)
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
        return asset_to_schema(asset, request)

    @app.get(f"{container.settings.api_prefix}/assets/{{asset_id}}/content", name="get_asset_file")
    def get_asset_file(
        asset_id: str,
        current_user=Depends(get_current_user),
        service: DocumentService = Depends(get_document_service),
    ):
        asset = service.get_asset(current_user.id, asset_id)
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
        return FileResponse(asset.storage_path, filename=asset.original_filename, media_type=asset.mime_type)

    @app.post(f"{container.settings.api_prefix}/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        current_user=Depends(get_rate_limited_user),
        service: DocumentService = Depends(get_document_service),
    ):
        result = service.chat(current_user.id, payload.question, payload.asset_ids)
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            timestamp_matches=[TopicMatch.model_validate(item) for item in result.timestamp_matches],
        )

    @app.post(f"{container.settings.api_prefix}/chat/stream")
    def stream_chat(
        payload: ChatRequest,
        current_user=Depends(get_rate_limited_user),
        service: DocumentService = Depends(get_document_service),
    ):
        def event_stream() -> Iterator[str]:
            for event in service.stream_chat(current_user.id, payload.question, payload.asset_ids):
                yield json.dumps(event, separators=(",", ":")) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post(
        f"{container.settings.api_prefix}/assets/{{asset_id}}/topics",
        response_model=list[TopicMatch],
    )
    def topic_search(
        asset_id: str,
        payload: TopicQueryRequest,
        current_user=Depends(get_rate_limited_user),
        service: DocumentService = Depends(get_document_service),
    ):
        asset = service.get_asset(current_user.id, asset_id)
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
        return [TopicMatch.model_validate(item) for item in service.find_topic_matches(current_user.id, asset_id, payload.topic)]

    return app


def asset_to_schema(asset, request: Request) -> AssetResponse:
    return AssetResponse.from_model(
        asset,
        file_url=str(request.url_for("get_asset_file", asset_id=asset.id)),
        text_preview=truncate_text(asset.extracted_text, 240),
    )


app = create_app()
