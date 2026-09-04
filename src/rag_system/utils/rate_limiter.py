"""Production rate limiter: per-tenant token bucket with Redis-backed sliding window.

Two modes:
  - In-process (asyncio.Lock): suitable for single-process deployments
  - Redis (atomic Lua): suitable for multi-process / K8s deployments
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

import structlog

from src.rag_system.utils.exceptions import APIRateLimitError

logger = structlog.get_logger(__name__)


class TokenBucket:
    """Async token bucket — thread-safe via asyncio.Lock."""

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens/second
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + delta * self.refill_rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if self.refill_rate <= 0:
                    wait = min(remaining, 0.1)
                else:
                    wait = min((tokens - self._tokens) / self.refill_rate, remaining, 0.1)
                await asyncio.sleep(wait)

    @property
    def available_tokens(self) -> float:
        self._refill()
        return round(self._tokens, 2)


class AsyncRateLimiter:
    """Per-entity (tenant/API) async rate limiter.

    Maintains separate token buckets per tenant_id for fair-queuing.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
        global_rps: float = 50.0,
    ) -> None:
        self._rps = requests_per_second
        self._burst = burst_size
        self._global_bucket = TokenBucket(capacity=global_rps * 2, refill_rate=global_rps)
        self._tenant_buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, tenant_id: str) -> TokenBucket:
        if tenant_id not in self._tenant_buckets:
            self._tenant_buckets[tenant_id] = TokenBucket(
                capacity=self._burst, refill_rate=self._rps
            )
        return self._tenant_buckets[tenant_id]

    async def acquire(self, tenant_id: str = "default", timeout: float = 30.0) -> None:
        """Acquire one token for tenant, subject to global cap.

        Raises:
            APIRateLimitError: If token cannot be acquired within timeout.
        """
        async with self._lock:
            bucket = self._get_bucket(tenant_id)

        # Check global first
        global_ok = await self._global_bucket.acquire(timeout=timeout)
        if not global_ok:
            raise APIRateLimitError(
                "Global rate limit exceeded",
                retry_after=(
                    int(1.0 / self._global_bucket.refill_rate)
                    if self._global_bucket.refill_rate > 0
                    else None
                ),
            )

        # Check per-tenant
        tenant_ok = await bucket.acquire(timeout=timeout)
        if not tenant_ok:
            raise APIRateLimitError(
                f"Per-tenant rate limit exceeded for tenant={tenant_id}",
                retry_after=int(1.0 / self._rps) if self._rps > 0 else None,
                api_name="rag-query",
            )

    def stats(self, tenant_id: str = "default") -> dict:
        bucket = self._tenant_buckets.get(tenant_id)
        return {
            "tenant_id": tenant_id,
            "available_tokens": bucket.available_tokens if bucket else self._burst,
            "capacity": self._burst,
            "refill_rate_rps": self._rps,
            "global_available": self._global_bucket.available_tokens,
        }

    async def __aenter__(self) -> AsyncRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


# ── Sliding-window Redis rate limiter (distributed) ───────────────────────────

_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local ttl = window * 2

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, now .. '-' .. math.random())
    redis.call('EXPIRE', key, ttl)
    return 1
end
return 0
"""


class RedisRateLimiter:
    """Distributed sliding-window rate limiter backed by Redis.

    Suitable for multi-process K8s deployments where per-process buckets
    would allow tenants to exceed limits.
    """

    def __init__(
        self,
        redis_url: str,
        window_seconds: int = 60,
        max_requests: int = 600,
    ) -> None:
        self._url = redis_url
        self._window = window_seconds
        self._max_requests = max_requests
        self._client: Optional[object] = None
        self._script: Optional[object] = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(self._url)
            except Exception as exc:
                logger.warning("redis_rate_limiter_unavailable", error=str(exc))
        return self._client

    async def is_allowed(self, key: str) -> bool:
        client = self._get_client()
        if not client:
            return True  # Fail open
        try:
            now_ms = int(time.time() * 1000)
            result = await client.eval(
                _SLIDING_WINDOW_LUA,
                1,
                f"rl:{key}",
                now_ms,
                self._window * 1000,
                self._max_requests,
            )
            return bool(result)
        except Exception as exc:
            logger.warning("redis_rate_limit_check_failed", error=str(exc))
            return True  # Fail open

    async def check_or_raise(self, key: str) -> None:
        allowed = await self.is_allowed(key)
        if not allowed:
            raise APIRateLimitError(
                f"Rate limit exceeded for key={key}",
                retry_after=self._window,
            )
