from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings


class CacheBackend(Protocol):
    def get_json(self, key: str) -> object | None:
        ...

    def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        ...

    def increment(self, key: str, *, ttl_seconds: int) -> int:
        ...


@dataclass(slots=True)
class _MemoryEntry:
    value: object
    expires_at: float


class InMemoryCache:
    def __init__(self):
        self._values: dict[str, _MemoryEntry] = {}
        self._lock = threading.Lock()

    def get_json(self, key: str) -> object | None:
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._values.pop(key, None)
                return None
            return entry.value

    def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        with self._lock:
            self._values[key] = _MemoryEntry(value=value, expires_at=time.monotonic() + ttl_seconds)

    def increment(self, key: str, *, ttl_seconds: int) -> int:
        with self._lock:
            now = time.monotonic()
            entry = self._values.get(key)
            if entry is None or entry.expires_at <= now:
                value = 1
            else:
                value = int(entry.value) + 1

            self._values[key] = _MemoryEntry(value=value, expires_at=now + ttl_seconds)
            return value


class RedisCache:
    def __init__(self, redis_url: str):
        import redis

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client.ping()

    def get_json(self, key: str) -> object | None:
        payload = self.client.get(key)
        return json.loads(payload) if payload else None

    def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.client.set(name=key, value=json.dumps(value), ex=ttl_seconds)

    def increment(self, key: str, *, ttl_seconds: int) -> int:
        value = int(self.client.incr(key))
        if value == 1:
            self.client.expire(key, ttl_seconds)
        return value


class RateLimitExceeded(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, cache: CacheBackend, *, limit: int, window_seconds: int):
        self.cache = cache
        self.limit = limit
        self.window_seconds = window_seconds

    def check(self, *, scope: str, identifier: str) -> None:
        current_count = self.cache.increment(
            f"rate-limit:{scope}:{identifier}",
            ttl_seconds=self.window_seconds,
        )
        if current_count > self.limit:
            raise RateLimitExceeded("Rate limit exceeded. Please wait and retry.")


def build_cache_backend(settings: Settings) -> CacheBackend:
    if not settings.redis_url:
        return InMemoryCache()

    try:
        return RedisCache(settings.redis_url)
    except Exception:
        # Local development should stay usable when Redis is temporarily unavailable.
        return InMemoryCache()
