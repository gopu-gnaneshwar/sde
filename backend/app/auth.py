from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.config import Settings


class AuthError(RuntimeError):
    pass


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str, *, iterations: int = 390_000) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "$".join(
        [
            "pbkdf2_sha256",
            str(iterations),
            _urlsafe_b64encode(salt),
            _urlsafe_b64encode(digest),
        ]
    )


def verify_password(password: str, encoded_password: str) -> bool:
    try:
        algorithm, iterations, salt, expected_digest = encoded_password.split("$", 3)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        _urlsafe_b64decode(salt),
        int(iterations),
    )
    return hmac.compare_digest(_urlsafe_b64encode(digest), expected_digest)


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def register(self, session: Session, username: str, password: str) -> tuple[models.User, str, str]:
        normalized_username = self._normalize_username(username)
        existing = session.scalar(select(models.User).where(models.User.username == normalized_username))
        if existing is not None:
            raise ValueError("Username is already registered.")

        api_key = self.generate_api_key()
        user = models.User(
            username=normalized_username,
            password_hash=hash_password(password),
            api_key_hash=hash_api_key(api_key),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user, self.create_access_token(user), api_key

    def login(self, session: Session, username: str, password: str) -> tuple[models.User, str]:
        normalized_username = self._normalize_username(username)
        user = session.scalar(select(models.User).where(models.User.username == normalized_username))
        if user is None or not verify_password(password, user.password_hash):
            raise AuthError("Invalid username or password.")
        return user, self.create_access_token(user)

    def authenticate(
        self,
        session: Session,
        *,
        authorization: str | None = None,
        api_key: str | None = None,
    ) -> models.User:
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise AuthError("Unsupported authorization scheme.")
            return self._get_user_from_token(session, token)

        if api_key:
            user = session.scalar(select(models.User).where(models.User.api_key_hash == hash_api_key(api_key)))
            if user is None:
                raise AuthError("Invalid API key.")
            return user

        raise AuthError("Authentication credentials were not provided.")

    def rotate_api_key(self, session: Session, user: models.User) -> str:
        api_key = self.generate_api_key()
        user.api_key_hash = hash_api_key(api_key)
        session.add(user)
        session.commit()
        session.refresh(user)
        return api_key

    def create_access_token(self, user: models.User) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.settings.access_token_expire_minutes)
        payload = {
            "sub": user.id,
            "username": user.username,
            "exp": int(expires_at.timestamp()),
        }

        encoded_header = _urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        encoded_payload = _urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signature = self._sign(f"{encoded_header}.{encoded_payload}")
        return f"{encoded_header}.{encoded_payload}.{signature}"

    def _get_user_from_token(self, session: Session, token: str) -> models.User:
        payload = self.decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise AuthError("Invalid access token.")

        user = session.get(models.User, user_id)
        if user is None:
            raise AuthError("User no longer exists.")
        return user

    def decode_access_token(self, token: str) -> dict[str, object]:
        try:
            encoded_header, encoded_payload, signature = token.split(".", 2)
        except ValueError as exc:
            raise AuthError("Invalid access token.") from exc

        message = f"{encoded_header}.{encoded_payload}"
        expected_signature = self._sign(message)
        if not hmac.compare_digest(signature, expected_signature):
            raise AuthError("Invalid access token signature.")

        payload = json.loads(_urlsafe_b64decode(encoded_payload))
        expires_at = int(payload.get("exp", 0))
        if expires_at <= int(datetime.now(timezone.utc).timestamp()):
            raise AuthError("Access token has expired.")
        return payload

    def generate_api_key(self) -> str:
        return f"mm_{secrets.token_urlsafe(24)}"

    def _sign(self, value: str) -> str:
        digest = hmac.new(
            self.settings.auth_secret_key.encode("utf-8"),
            value.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return _urlsafe_b64encode(digest)

    @staticmethod
    def _normalize_username(value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,31}", normalized):
            raise ValueError(
                "Username must be 3-32 characters and use only letters, numbers, dots, hyphens, or underscores."
            )
        return normalized
