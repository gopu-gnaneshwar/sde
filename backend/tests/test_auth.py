from __future__ import annotations

import json

import pytest

from app.auth import AuthError, AuthManager, _urlsafe_b64encode, hash_password, verify_password
from app.config import Settings
from app.database import build_engine, build_session_factory, init_db


def test_password_hash_round_trip():
    encoded = hash_password("supersecret1")
    assert verify_password("supersecret1", encoded)
    assert not verify_password("wrong-password", encoded)
    assert not verify_password("supersecret1", "invalid")
    assert not verify_password("supersecret1", encoded.replace("pbkdf2_sha256", "sha1", 1))


def test_auth_manager_register_login_and_api_key_lookup(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'auth.db'}",
        storage_dir=tmp_path / "data",
        auth_secret_key="test-secret",
    )
    engine = build_engine(settings.database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session = session_factory()
    manager = AuthManager(settings)

    user, access_token, api_key = manager.register(session, "tester", "supersecret1")
    assert user.username == "tester"
    assert access_token
    assert api_key.startswith("mm_")

    payload = manager.decode_access_token(access_token)
    assert payload["sub"] == user.id

    logged_in_user, login_token = manager.login(session, "tester", "supersecret1")
    assert logged_in_user.id == user.id
    assert manager.authenticate(session, authorization=f"Bearer {login_token}").id == user.id
    assert manager.authenticate(session, api_key=api_key).id == user.id

    rotated_api_key = manager.rotate_api_key(session, user)
    assert rotated_api_key.startswith("mm_")
    with pytest.raises(AuthError):
        manager.authenticate(session, api_key=api_key)

    session.close()


def test_auth_manager_rejects_bad_credentials(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'auth-errors.db'}",
        storage_dir=tmp_path / "data",
        auth_secret_key="test-secret",
    )
    engine = build_engine(settings.database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session = session_factory()
    manager = AuthManager(settings)
    manager.register(session, "tester", "supersecret1")

    with pytest.raises(AuthError):
        manager.login(session, "tester", "wrong-password")

    with pytest.raises(ValueError):
        manager.register(session, "invalid name", "supersecret1")

    with pytest.raises(AuthError):
        manager.authenticate(session)

    session.close()


def test_auth_manager_rejects_invalid_tokens_and_missing_users(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'auth-token.db'}",
        storage_dir=tmp_path / "data",
        auth_secret_key="test-secret",
        access_token_expire_minutes=1,
    )
    engine = build_engine(settings.database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session = session_factory()
    manager = AuthManager(settings)
    user, token, _ = manager.register(session, "tester", "supersecret1")

    with pytest.raises(AuthError):
        manager.authenticate(session, authorization="Basic abc123")

    with pytest.raises(AuthError):
        manager.decode_access_token("bad-token")

    header, payload, _ = token.split(".", 2)
    expired_payload = _urlsafe_b64encode(json.dumps({"sub": user.id, "exp": 0}).encode("utf-8"))
    expired_signature = manager._sign(f"{header}.{expired_payload}")
    with pytest.raises(AuthError):
        manager.decode_access_token(f"{header}.{expired_payload}.{expired_signature}")

    with pytest.raises(AuthError):
        manager.decode_access_token(f"{header}.{payload}.broken")

    missing_sub_payload = _urlsafe_b64encode(json.dumps({"exp": 9999999999}).encode("utf-8"))
    missing_sub_signature = manager._sign(f"{header}.{missing_sub_payload}")
    with pytest.raises(AuthError):
        manager._get_user_from_token(session, f"{header}.{missing_sub_payload}.{missing_sub_signature}")

    deleted_user_payload = _urlsafe_b64encode(json.dumps({"sub": "missing-user", "exp": 9999999999}).encode("utf-8"))
    deleted_user_signature = manager._sign(f"{header}.{deleted_user_payload}")
    with pytest.raises(AuthError):
        manager._get_user_from_token(session, f"{header}.{deleted_user_payload}.{deleted_user_signature}")

    with pytest.raises(ValueError):
        manager.register(session, "tester", "supersecret1")

    session.close()
